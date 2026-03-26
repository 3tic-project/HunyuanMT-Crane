#!/usr/bin/env python3
"""Benchmark Crane's Qwen3 serving path over the OpenAI-compatible API.

This script is aimed at the Phase 0 / Phase 1 baseline workloads from
`docs/qwen3-performance-roadmap.md`. It uses only Python's standard library so
it can run directly on remote CUDA servers without extra dependencies.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_STATS_POLL_INTERVAL = 0.25
DEFAULT_GPU_POLL_INTERVAL = 0.25


@dataclass(frozen=True)
class RequestSpec:
    label: str
    user_prompt: str
    max_tokens: int
    system_prompt: str | None = None


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    concurrency: int
    requests: list[RequestSpec]


def repeat_words(prefix: str, word: str, count: int) -> str:
    count = max(count, 1)
    return f"{prefix}\n" + " ".join([word] * count)


def default_workloads() -> dict[str, Workload]:
    shared_system = repeat_words("System prompt:", "policy", 256)
    short_prompt = repeat_words("User prompt:", "short", 64)
    medium_prompt = repeat_words("User prompt:", "medium", 256)
    long_prompt_4k = repeat_words("User prompt:", "context", 4096)
    long_prompt_8k = repeat_words("User prompt:", "context", 8192)
    long_prompt_32k = repeat_words("User prompt:", "context", 32768)

    workloads = {
        "w1": Workload(
            name="w1",
            description="单请求短 prompt，短生成",
            concurrency=1,
            requests=[RequestSpec("w1-0", short_prompt, 32)],
        ),
        "w2": Workload(
            name="w2",
            description="单请求长 prompt，短生成",
            concurrency=1,
            requests=[RequestSpec("w2-0", long_prompt_4k, 32)],
        ),
        "w3": Workload(
            name="w3",
            description="8 并发，短 prompt，中等生成",
            concurrency=8,
            requests=[RequestSpec(f"w3-{i}", short_prompt, 128) for i in range(8)],
        ),
        "w4": Workload(
            name="w4",
            description="32 并发，共享 system prompt",
            concurrency=32,
            requests=[
                RequestSpec(f"w4-{i}", medium_prompt, 96, system_prompt=shared_system)
                for i in range(32)
            ],
        ),
        "w5_8k": Workload(
            name="w5_8k",
            description="长上下文 8K decode",
            concurrency=8,
            requests=[RequestSpec(f"w5_8k-{i}", long_prompt_8k, 64) for i in range(8)],
        ),
        "w5_32k": Workload(
            name="w5_32k",
            description="长上下文 32K decode",
            concurrency=4,
            requests=[RequestSpec(f"w5_32k-{i}", long_prompt_32k, 32) for i in range(4)],
        ),
        "w6": Workload(
            name="w6",
            description="高 batch decode，固定 decode tokens",
            concurrency=32,
            requests=[RequestSpec(f"w6-{i}", medium_prompt, 256) for i in range(32)],
        ),
        "w7": Workload(
            name="w7",
            description="一个超长 prefill 与一批活跃 decode 请求混跑",
            concurrency=8,
            requests=[
                RequestSpec("w7-long", long_prompt_8k, 96, system_prompt=shared_system),
                *[
                    RequestSpec(f"w7-short-{i}", short_prompt, 96, system_prompt=shared_system)
                    for i in range(7)
                ],
            ],
        ),
    }
    return workloads


def fetch_json(url: str, timeout: int) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = (len(values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return values[low]
    weight = idx - low
    return values[low] * (1.0 - weight) + values[high] * weight


def format_ms(ms: float) -> str:
    return f"{ms:.2f}"


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GiB"
    if num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.1f} MiB"
    if num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.1f} KiB"
    return f"{num_bytes} B"


class StatsSampler:
    def __init__(self, stats_url: str, timeout: int, interval_s: float) -> None:
        self.stats_url = stats_url
        self.timeout = timeout
        self.interval_s = interval_s
        self.max_kv_bytes = 0
        self.max_kv_pages = 0
        self.samples = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="stats-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = fetch_json(self.stats_url, self.timeout)
                self.samples += 1
                self.max_kv_bytes = max(
                    self.max_kv_bytes, int(snapshot.get("current_tracked_kv_bytes", 0))
                )
                self.max_kv_pages = max(
                    self.max_kv_pages, int(snapshot.get("current_estimated_kv_pages", 0))
                )
            except Exception:
                self.errors += 1
            self._stop.wait(self.interval_s)


class NvidiaSmiSampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.max_used_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = self._detect()

    def _detect(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return False
        return True

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if values:
                    self.samples += 1
                    self.max_used_mib = max(self.max_used_mib, max(values))
            except Exception:
                pass
            self._stop.wait(self.interval_s)


@dataclass
class RequestResult:
    label: str
    ok: bool
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    error: str | None = None


def stream_chat_completion(
    base_url: str,
    model: str,
    spec: RequestSpec,
    timeout: int,
) -> RequestResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    messages = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append({"role": "user", "content": spec.user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    ttft_ms: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                if data.startswith("error:"):
                    raise RuntimeError(data)

                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if ttft_ms is None and delta.get("content"):
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    finish_reason = choices[0].get("finish_reason") or finish_reason

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))

        end = time.perf_counter()
        return RequestResult(
            label=spec.label,
            ok=True,
            ttft_ms=ttft_ms if ttft_ms is not None else (end - start) * 1000.0,
            e2e_ms=(end - start) * 1000.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        end = time.perf_counter()
        return RequestResult(
            label=spec.label,
            ok=False,
            ttft_ms=0.0,
            e2e_ms=(end - start) * 1000.0,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason=None,
            error=str(exc),
        )


def diff_stats(before: dict[str, Any], after: dict[str, Any], keys: list[str]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key in keys:
        delta[key] = int(after.get(key, 0)) - int(before.get(key, 0))
    return delta


def run_workload(
    base_url: str,
    model: str,
    workload: Workload,
    timeout: int,
    stats_interval: float,
    gpu_interval: float,
) -> dict[str, Any]:
    stats_url = base_url.rstrip("/") + "/v1/stats"
    before_stats = fetch_json(stats_url, timeout)

    stats_sampler = StatsSampler(stats_url, timeout, stats_interval)
    gpu_sampler = NvidiaSmiSampler(gpu_interval)
    stats_sampler.start()
    gpu_sampler.start()

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workload.concurrency) as pool:
        futures = [
            pool.submit(stream_chat_completion, base_url, model, spec, timeout)
            for spec in workload.requests
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed_s = time.perf_counter() - started

    stats_sampler.stop()
    gpu_sampler.stop()
    after_stats = fetch_json(stats_url, timeout)

    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    ttft_values = [result.ttft_ms for result in successes]
    e2e_values = [result.e2e_ms for result in successes]
    prompt_tokens = sum(result.prompt_tokens for result in successes)
    completion_tokens = sum(result.completion_tokens for result in successes)

    stats_delta = diff_stats(
        before_stats,
        after_stats,
        [
            "total_prefill_chunks",
            "total_prefill_time_us",
            "total_prefill_chunk_time_us",
            "total_decode_steps",
            "total_decode_time_us",
            "total_batch_decode_setup_time_us",
            "total_batch_decode_extract_time_us",
            "total_batch_decode_mask_time_us",
            "total_decode_plan_time_us",
            "total_sampling_time_us",
            "total_decode_plan_cache_hits",
            "total_decode_plan_cache_misses",
            "total_h2d_metadata_bytes",
            "total_page_budget_denials",
            "total_completion_tokens",
            "total_prompt_tokens",
        ],
    )

    output = {
        "workload": workload.name,
        "description": workload.description,
        "requests": len(workload.requests),
        "successes": len(successes),
        "failures": len(failures),
        "elapsed_s": elapsed_s,
        "requests_per_s": (len(successes) / elapsed_s) if elapsed_s > 0 else 0.0,
        "aggregate_completion_tok_s": (completion_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": {
            "p50": percentile(ttft_values, 0.50),
            "p95": percentile(ttft_values, 0.95),
            "p99": percentile(ttft_values, 0.99),
        },
        "e2e_ms": {
            "p50": percentile(e2e_values, 0.50),
            "p95": percentile(e2e_values, 0.95),
            "p99": percentile(e2e_values, 0.99),
        },
        "stats_delta": stats_delta,
        "max_current_tracked_kv_bytes": stats_sampler.max_kv_bytes,
        "max_current_estimated_kv_pages": stats_sampler.max_kv_pages,
        "peak_gpu_used_mib": gpu_sampler.max_used_mib if gpu_sampler.available else None,
        "failed_requests": [asdict(result) for result in failures],
    }
    return output


def print_report(report: dict[str, Any]) -> None:
    print()
    print(f"[{report['workload']}] {report['description']}")
    print(
        f"requests={report['requests']} successes={report['successes']} failures={report['failures']} "
        f"elapsed={report['elapsed_s']:.2f}s req/s={report['requests_per_s']:.2f} "
        f"decode_tok/s={report['aggregate_completion_tok_s']:.2f}"
    )
    print(
        f"ttft_ms p50={format_ms(report['ttft_ms']['p50'])} "
        f"p95={format_ms(report['ttft_ms']['p95'])} "
        f"p99={format_ms(report['ttft_ms']['p99'])}"
    )
    print(
        f"e2e_ms  p50={format_ms(report['e2e_ms']['p50'])} "
        f"p95={format_ms(report['e2e_ms']['p95'])} "
        f"p99={format_ms(report['e2e_ms']['p99'])}"
    )
    print(
        f"prompt_tokens={report['prompt_tokens']} completion_tokens={report['completion_tokens']} "
        f"max_kv={format_bytes(report['max_current_tracked_kv_bytes'])} "
        f"max_pages={report['max_current_estimated_kv_pages']}"
    )
    if report["peak_gpu_used_mib"] is not None:
        print(f"peak_gpu_used={report['peak_gpu_used_mib']} MiB")

    stats_delta = report["stats_delta"]
    print(
        "stats_delta "
        f"prefill_chunks={stats_delta['total_prefill_chunks']} "
        f"decode_steps={stats_delta['total_decode_steps']} "
        f"setup_ms={stats_delta['total_batch_decode_setup_time_us'] / 1000:.2f} "
        f"extract_ms={stats_delta['total_batch_decode_extract_time_us'] / 1000:.2f} "
        f"mask_ms={stats_delta['total_batch_decode_mask_time_us'] / 1000:.2f} "
        f"plan_ms={stats_delta['total_decode_plan_time_us'] / 1000:.2f} "
        f"sampling_ms={stats_delta['total_sampling_time_us'] / 1000:.2f} "
        f"h2d_metadata={format_bytes(stats_delta['total_h2d_metadata_bytes'])}"
    )

    if report["failed_requests"]:
        print("failed_requests:")
        for item in report["failed_requests"]:
            print(f"  - {item['label']}: {item['error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Crane Qwen3 serving")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workload",
        action="append",
        choices=sorted(default_workloads().keys()),
        help="Run one or more predefined workloads. Defaults to w1,w3,w4,w7.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--stats-poll-interval", type=float, default=DEFAULT_STATS_POLL_INTERVAL)
    parser.add_argument("--gpu-poll-interval", type=float, default=DEFAULT_GPU_POLL_INTERVAL)
    parser.add_argument("--json-out", help="Write the full benchmark report to JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workloads = default_workloads()
    selected = args.workload or ["w1", "w3", "w4", "w7"]

    reports = []
    for name in selected:
        report = run_workload(
            args.base_url,
            args.model,
            workloads[name],
            args.timeout,
            args.stats_poll_interval,
            args.gpu_poll_interval,
        )
        reports.append(report)
        print_report(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "benchmark_id": str(uuid.uuid4()),
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "base_url": args.base_url,
                    "model": args.model,
                    "host": os.uname().nodename if hasattr(os, "uname") else None,
                    "reports": reports,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
