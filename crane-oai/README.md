# crane-oai

OpenAI & SGLang 兼容的高性能推理 API 服务器，基于 [Crane](../README.md) 框架构建，支持连续批处理（continuous batching）。

## 特性

- **OpenAI 兼容 API** — 完全兼容 OpenAI Chat Completions、Text Completions、Models、Tokenize/Detokenize 接口
- **SGLang 原生 API** — 支持 SGLang 风格的 `/generate`、`/model_info`、`/server_info` 等端点
- **连续批处理** — 专用推理线程，FIFO 调度器，自动管理 KV 缓存交换
- **多模型支持** — 自动检测并加载 Hunyuan Dense、Qwen 2.5、Qwen 3 等模型架构
- **流式响应** — SSE (Server-Sent Events) 实时流式 token 输出
- **跨平台加速** — CPU / CUDA / Apple Metal 自动选择

## 快速开始

### 构建

```bash
# CPU 版本
cargo build -p crane-oai --release

# CUDA GPU 版本
cargo build -p crane-oai --release --features cuda
```

### 启动服务器

```bash
# 基本用法 — 自动检测模型类型和设备
crane-oai --model-path /path/to/model

# 指定模型类型和端口
crane-oai --model-path /path/to/Qwen2.5-7B-Instruct \
    --model-type qwen25 \
    --port 8000

# 使用 GGUF 格式权重
crane-oai --model-path /path/to/model.gguf \
    --format gguf

# 强制使用 CPU
crane-oai --model-path /path/to/model --cpu
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | *必填* | 模型目录路径或 GGUF 文件路径 |
| `--model-type` | `auto` | 模型架构: `auto`, `hunyuan`, `qwen25`, `qwen3` |
| `--model-name` | 目录名 | API 响应中展示的模型名称 |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--port` | `8080` | 绑定端口 |
| `--cpu` | `false` | 强制使用 CPU（即使 GPU 可用） |
| `--max-concurrent` | `32` | 解码阶段最大并发序列数 |
| `--decode-tokens-per-seq` | `16` | 每个序列切换前解码的 token 数（越大 KV 交换越少） |
| `--format` | `auto` | 权重格式: `auto`, `safetensors`, `gguf` |

## API 端点

### OpenAI 兼容接口

#### `POST /v1/chat/completions`

Chat 对话补全，支持流式和非流式响应。

```bash
# 非流式
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'

# 流式 (SSE)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

**请求参数**:

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 是 | — | 模型名称 |
| `messages` | array | 是 | — | 消息列表 `[{role, content}]` |
| `max_tokens` | int | 否 | `512` | 最大生成 token 数 |
| `temperature` | float | 否 | `0.8` | 采样温度，0 表示贪心 |
| `top_p` | float | 否 | `0.95` | Nucleus 采样阈值 |
| `top_k` | int | 否 | `40` | Top-k 采样 |
| `repetition_penalty` | float | 否 | `1.05` | 重复惩罚系数 |
| `stream` | bool | 否 | `false` | 启用 SSE 流式响应 |
| `stream_options` | object | 否 | — | `{"include_usage": true}` 在最终 chunk 含 usage |
| `seed` | int | 否 | — | 随机种子 |

#### `POST /v1/completions`

文本补全（无 chat 模板）。

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 64
  }'
```

`prompt` 支持单个字符串或字符串数组（拼接处理）。

#### `GET /v1/models`

列出可用模型。

```bash
curl http://localhost:8080/v1/models
```

#### `GET /v1/models/:model_id`

获取指定模型信息。

```bash
curl http://localhost:8080/v1/models/Qwen2.5-7B-Instruct
```

#### `POST /v1/tokenize`

将文本分词为 token ID。

```bash
# 直接文本
curl http://localhost:8080/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# 通过 chat 模板
curl http://localhost:8080/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}]}'
```

#### `POST /v1/detokenize`

将 token ID 解码为文本。

```bash
curl http://localhost:8080/v1/detokenize \
  -H "Content-Type: application/json" \
  -d '{"tokens": [9707, 1917]}'
```

### SGLang 兼容接口

#### `POST /generate`

原生文本生成端点。

```bash
# 文本输入
curl http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The meaning of life is",
    "sampling_params": {
      "max_new_tokens": 128,
      "temperature": 0.8,
      "top_p": 0.95
    }
  }'

# Token ID 输入
curl http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_ids": [1, 2, 3],
    "sampling_params": {"max_new_tokens": 64},
    "stream": true
  }'
```

**`sampling_params` 字段**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_new_tokens` | int | `128` | 最大生成 token 数 |
| `temperature` | float | `0.8` | 采样温度 |
| `top_p` | float | `0.95` | Nucleus 采样 |
| `top_k` | int | `20` | Top-k 采样 |
| `repetition_penalty` | float | `1.0` | 重复惩罚 |
| `stop` | string/array | — | 停止字符串 |
| `stop_token_ids` | array | — | 停止 token ID |
| `seed` | int | — | 随机种子 |
| `n` | int | `1` | 并行生成数量 |

#### `GET /model_info`

返回模型元数据。

```bash
curl http://localhost:8080/model_info
```

```json
{
  "model_path": "/models/Qwen2.5-7B-Instruct",
  "model_type": "qwen25",
  "is_generation": true,
  "dtype": "F32",
  "device": "Metal(0)"
}
```

#### `GET /server_info`

返回服务器配置和实时引擎统计信息。

```bash
curl http://localhost:8080/server_info
```

```json
{
  "version": "0.1.0",
  "model_path": "/models/Qwen2.5-7B-Instruct",
  "model_type": "qwen25",
  "host": "0.0.0.0",
  "port": 8080,
  "max_concurrent": 32,
  "decode_tokens_per_seq": 16,
  "stats": {
    "total_requests": 42,
    "completed_requests": 40,
    "avg_decode_tokens_per_sec": 35.2,
    "avg_prefill_tokens_per_sec": 120.5,
    "active_sequences": 2,
    "waiting_sequences": 0
  }
}
```

#### `GET /health_generate`

深度健康检查 — 运行 1-token 推理探测，30 秒超时。

```bash
curl http://localhost:8080/health_generate
# {"status": "ok"}
```

#### `POST /flush_cache`

KV 缓存刷新（当前架构下 KV 缓存随序列生命周期自动管理，此端点为兼容性保留）。

#### `POST /abort_request`

取消进行中的请求。

```bash
curl http://localhost:8080/abort_request \
  -H "Content-Type: application/json" \
  -d '{"rid": "gen-xxxx-xxxx"}'
```

### 管理接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查，返回 `{"status": "ok"}` |
| `/v1/stats` | GET | 引擎统计快照（请求数、吞吐率、活跃序列等） |

## 架构

```
crane-oai/src/
├── main.rs              # CLI 入口、AppState、路由构建
├── openai_api.rs        # OpenAI 请求/响应类型定义
├── sglang_api.rs        # SGLang 原生 API 类型定义
├── chat_template.rs     # 聊天模板处理（Jinja / Hunyuan 硬编码）
├── handlers/
│   ├── mod.rs           # Handler 模块声明
│   ├── common.rs        # /health, /v1/stats
│   ├── openai.rs        # OpenAI 兼容端点处理器
│   ├── sglang.rs        # SGLang 兼容端点处理器
│   └── sse.rs           # SSE 流构建器
└── engine/
    ├── mod.rs           # InferenceEngine 核心循环（连续批处理）
    ├── types.rs         # EngineRequest / EngineResponse / EngineHandle
    ├── stats.rs         # 无锁原子统计计数器
    ├── sampling.rs      # Token 采样（Top-k/p、Gumbel-max、重复惩罚）
    ├── scheduler.rs     # FIFO 调度器（prefill 优先）
    ├── sequence.rs      # 序列生命周期管理
    ├── backend.rs       # ModelBackend trait 及各模型实现
    └── model_factory.rs # 模型自动检测与工厂函数
```

### 推理引擎

引擎运行在专用 OS 线程上，通过 `mpsc` channel 与 Tokio 异步 HTTP handler 通信：

```
HTTP Request → Handler → EngineHandle.submit() ──channel──→ InferenceEngine
                  ↑                                              │
                  └──── UnboundedReceiver<EngineResponse> ←──────┘
                         (Token / Finished / Error)
```

**调度策略**:
- **Prefill 优先** — 新请求优先处理 prefill 以尽快开始流式响应
- **批量 Decode** — 所有 running 序列在同一步骤中各解码一个 token
- **KV 缓存交换** — 当并发数超过模型批处理能力时，自动 swap-in/swap-out（Hunyuan 支持）
- **`max_running`** — 控制同时处于 decode 阶段的序列数，限制峰值 KV 缓存内存

### 模型后端

通过 `ModelBackend` trait 抽象，支持不同模型架构：

| 模型 | 批解码 | KV Swap | 格式 |
|------|--------|---------|------|
| Hunyuan Dense | ✅ | ✅ | Safetensors / GGUF |
| Qwen 2.5 | 顺序 | ❌ | Safetensors |
| Qwen 3 | 顺序 | ❌ | Safetensors |

模型类型可通过 `--model-type` 显式指定，或从 `config.json` 的 `model_type` / `architectures` 字段自动检测。

## 与 OpenAI SDK 配合使用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # crane-oai 不要求 API key
)

# 非流式
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 测试

```bash
# 运行所有单元测试（122 个）
cargo test -p crane-oai

# 运行特定模块测试
cargo test -p crane-oai engine::scheduler
cargo test -p crane-oai openai_api::tests
cargo test -p crane-oai sglang_api::tests
```

## License

MIT
