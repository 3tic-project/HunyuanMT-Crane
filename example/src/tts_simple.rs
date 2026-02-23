//! Qwen3-TTS Simple Example
//!
//! Demonstrates text-to-speech synthesis using crane-core's Qwen3-TTS model.
//!
//! # Model variants
//!
//! There are two model variants — make sure you use the correct one:
//!
//! | Model | Type | Speaker control |
//! |-------|------|----------------|
//! | `Qwen3-TTS-12Hz-0.6B-Base` | `base` | Voice cloning (needs reference audio) |
//! | `Qwen3-TTS-12Hz-0.6B-CustomVoice` | `custom_voice` | Predefined speakers (e.g. `serena`, `ryan`) |
//!
//! This example generates speech **without** a reference voice, so:
//! - **Base model**: speaker = `None` (model uses a default voice)
//! - **CustomVoice model**: speaker = one of the predefined names
//!
//! # Setup
//!
//! ```bash
//! # Download one of the models
//! huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
//!     --local-dir checkpoints/Qwen3-TTS-12Hz-0.6B-Base
//!
//! # or (for predefined speakers):
//! huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
//!     --local-dir checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice
//!
//! # Run
//! CRANE_TTS_DEBUG=1 cargo run --bin tts_simple --release
//! ```

fn main() -> anyhow::Result<()> {
    use crane_core::models::{DType, Device};

    // ── Choose model path ──────────────────────────────────────────
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "vendor/Qwen3-TTS-12Hz-0.6B-Base".into());

    let device = {
        #[cfg(feature = "cuda")]
        { Device::new_cuda(0).unwrap_or(Device::Cpu) }
        #[cfg(all(target_os = "macos", not(feature = "cuda")))]
        { Device::new_metal(0).unwrap_or(Device::Cpu) }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        { Device::Cpu }
    };
    let dtype = {
        #[cfg(feature = "cuda")]
        { DType::BF16 }
        #[cfg(not(feature = "cuda"))]
        { DType::F32 }
    };

    println!("Loading Qwen3-TTS from: {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = crane_core::models::qwen3_tts::Model::new(&model_path, &device, &dtype)?;

    // ── Detect model type and pick speaker ─────────────────────────
    let model_type = model.config.tts_model_type.as_deref().unwrap_or("base");
    println!("Model type: {model_type}");

    let speaker: Option<String> = match model_type {
        "custom_voice" => {
            // Use first available speaker from config
            let first = model.config.talker_config.spk_id
                .keys()
                .next()
                .cloned();
            println!("Available speakers: {:?}", model.config.talker_config.spk_id.keys().collect::<Vec<_>>());
            first
        }
        _ => None, // Base model: no predefined speaker
    };

    let examples: &[(&str, &str, &str)] = &[
        // (text, language, output_stem)
        ("今天天气真好，我们去公园吧！",                                    "chinese", "output_tts_zh"),
        ("Hello! I am Crane, an ultra-fast inference engine in Rust.", "english", "output_tts_en"),
    ];

    for (i, (text, lang, stem)) in examples.iter().enumerate() {
        println!("\n[{}/{}] lang={lang}  speaker={}", i + 1, examples.len(), speaker.as_deref().unwrap_or("(none)"));
        println!("  Text: {text}");

        let (audio, sr) = model.generate_speech(
            text,
            lang,
            speaker.as_deref(),
            2048,   // max codec tokens
            0.9,    // temperature (Python default)
            Some(1.0), // top_p (Python default)
            1.05,   // repetition_penalty (Python default)
        )?;

        let wav_path = format!("{stem}.wav");
        // Convert to PCM and write WAV
        let audio_f32 = audio.to_dtype(crane_core::models::DType::F32)?.flatten_all()?;
        let samples = audio_f32.to_vec1::<f32>()?;
        println!("  Generated {:.1}s of audio ({} samples @ {sr} Hz)", samples.len() as f32 / sr as f32, samples.len());

        write_wav(&wav_path, &samples, sr)?;
        println!("  Saved {wav_path}");
    }

    println!("\nDone!");
    Ok(())
}

/// Write a 16-bit mono WAV file from f32 samples.
fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    use std::io::Write;

    let num_samples = samples.len() as u32;
    let data_len = num_samples * 2; // 16-bit = 2 bytes per sample
    let mut f = std::fs::File::create(path)?;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;    // PCM
    f.write_all(&1u16.to_le_bytes())?;    // mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?;    // block align
    f.write_all(&16u16.to_le_bytes())?;   // bits per sample
    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;

    for &s in samples {
        let scaled = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&scaled.to_le_bytes())?;
    }

    Ok(())
}
