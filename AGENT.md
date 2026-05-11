# Project Guidance

- Keep this project pure Uya for model/runtime logic. Do not wrap Python,
  PyTorch, llama.cpp, or the official MiniCPM-o implementation as runtime
  dependencies.
- Target Linux x86_64 CPU first. GPU backends, WebRTC, and native media IO are
  future portability topics, not initial requirements.
- Prefer small, testable modules: binary helpers, GGUF/model readers,
  tokenizer, tensor storage, quantization kernels, Qwen3 decoder, vision tower,
  audio encoder, speech decoder/vocoder, sampler, and CLI.
- Be explicit about supported modality/state. An inspector, tokenizer, or
  encoder smoke test is not full omni generation.
- Avoid committing large model files, generated audio/video, or downloaded
  assets. Use external paths and tiny fixtures.
