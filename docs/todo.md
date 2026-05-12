# minicpm-o-uya TODO

## Phase 0: 项目基座

- [x] 新建 `minicpm-o-uya` 仓库目录。
- [x] 写 `README.md`。
- [x] 写 `AGENT.md` 项目约束。
- [x] 增加 `Makefile` scaffold。
- [x] 增加 `src/main.uya` CLI scaffold。
- [x] 增加 `tests/README.md`。
- [x] 写 `docs/design.md`。
- [x] 写 `docs/todo.md`。
- [x] 确认本机 Uya 编译器路径和最小 scaffold 可编译。
- [x] 增加 `.gitignore`，排除 build、模型、音视频、raw fixture。

验收标准：

- `git status` 能看到初始项目文件。
- `make build` 能生成 `build/minicpm-o-uya`，或明确记录当前 Uya 编译器缺失。
- `build/minicpm-o-uya --help` 能说明当前是 scaffold。
- 文档明确说明当前无真实推理能力。

## Phase 1: Binary 与 GGUF inspector

- [x] 实现 little-endian binary reader。
- [x] 实现 safe file seek/read。
- [x] 读取 GGUF magic/version/tensor_count/metadata_count。
- [x] 安全跳过 GGUF metadata value。
- [x] 支持 scalar/string/array metadata。
- [x] 读取 tensor name、n_dims、shape、ggml_type、relative offset。
- [x] 计算 data section alignment 和 absolute tensor offset。
- [x] 计算 tensor byte size。
- [x] 实现 `inspect <model.gguf>`。
- [x] 对坏 magic、短 header、短 metadata、短 tensor directory 给明确错误。

验收标准：

- [x] tiny GGUF fixture 可打印 header 和 tensor table。
- [x] 对 `.part` 截断文件不崩溃。
- [x] `inspect` 不读取大 tensor data。

## Phase 2: MiniCPM-o schema audit

- [x] 实现 `audit <model.gguf>`。
- [x] 统计 metadata key/value 类型分布。
- [x] 统计 tensor dtype 分布。
- [x] 统计 root/text/vision/audio/speech tensor 命名分支。
- [x] 识别 Qwen3 text backbone metadata。
- [x] 识别 vision tower metadata。
- [x] 识别 audio encoder metadata。
- [x] 识别 speech/vocoder metadata。
- [x] 识别 media placeholder/special token metadata。
- [x] 输出 unsupported layout 诊断，不假装可生成。
- [x] 增加 `MINICPM_O_GGUF=/path make minicpmo-audit` 文档 target。

验收标准：

- [x] 对真实或社区 MiniCPM-o GGUF 能输出 tensor 数、dtype 分布、分支计数。
- [x] 能定位缺失 tokenizer、缺 root tensor、未知 dtype、未知 modality 分支。
- [x] `audit` 全程不 mmap/read tensor data。

## Phase 3: Tokenizer 与 chat template

- [x] 解析 `tokenizer.ggml.tokens`。
- [x] 解析 token scores/token types。
- [x] 解析 BPE merges 或实际 tokenizer rank 表。
- [x] 支持 BOS/EOS/PAD/UNK。
- [x] 支持 added/special tokens。
- [x] 实现 token id -> piece。
- [x] 实现 decode tokens -> text。
- [x] 实现 encode text -> token ids。
- [x] 支持 Qwen/ChatML-style role 模板。
- [x] 支持 MiniCPM-o media placeholder token。
- [x] 实现 `piece`、`encode`、`decode`、`format-chat` CLI。
- [x] 增加英文、中文、符号、换行、特殊 token golden tests。

验收标准：

- [x] 固定 prompt token 序列与可信 tiny GGUF fixture 一致。
- [x] `format-chat` 输出与 ChatML-style 模板对齐。
- [x] 多模态 placeholder 不被普通 BPE 错分。

## Phase 4: Tensor runtime 与 mmap 权重

- [x] 定义 `TensorDType`。
- [x] 定义 `TensorView`。
- [x] 定义 `TensorWeightTable`。
- [x] 定义 shape/stride/byte-size helpers。
- [x] 将 GGUF tensor table 转为 weight table。
- [x] 支持只读 mmap 挂接 tensor data。
- [x] 实现 root tensor lookup。
- [x] 实现 per-layer tensor name builder。
- [x] 实现 scratch arena。
- [x] 实现 LLM KV cache layout。
- [x] 实现 vision/audio/speech cache skeleton。
- [x] 增加越界、shape mismatch、缺 data tests。

验收标准：

- [x] 不复制大 tensor 即可建立 `TensorView.data`。
- [x] 能按 tensor 名称查 dtype、shape、offset、data pointer。
- [x] 对未知 dtype 和 byte-size 不明的 tensor 明确报错。

## Phase 5: Reference kernels 基础

- [x] F32 vector fill/copy/add/mul/dot。
- [x] F16 load -> F32。
- [x] BF16 load -> F32。
- [x] RMSNorm。
- [x] LayerNorm。
- [x] RoPE。
- [x] Softmax/masked softmax。
- [x] Dense F32/F16/BF16 matvec。
- [x] SwiGLU/SiLU/GELU。
- [x] Conv1D reference。
- [x] Conv2D reference。
- [x] Tiny golden tests。

验收标准：

- [x] 每个 kernel 有小尺寸 deterministic golden。
- [x] NaN/Inf/空长度边界处理明确。
- [x] scalar reference 输出稳定。

## Phase 6: Quantization kernels

- [x] 根据 MiniCPM-o GGUF audit 选择首批 dtype。
- [x] 实现 Q8_0 dot。
- [x] 实现 Q4_K dot。
- [x] 实现 Q5_K dot。
- [x] 实现 Q6_K dot。
- [x] 实现必要 IQ dtype dot。
- [x] 实现 fused dequant + dot。
- [x] 支持 quant matvec row stride。
- [x] 与 scalar dequant reference 对照。

验收标准：

- [x] 支持真实模型中 text decoder 命中的首批 dtype（首批覆盖 Q8_0、Q4_K、Q5_K、Q6_K、IQ4_NL）。
- [x] quant kernel 误差在阈值内。
- [x] unsupported dtype 报 tensor name 和 dtype。

## Phase 7: Qwen3 text config 与权重绑定

- [x] 建立 `Qwen3Config`。
- [x] 从 GGUF metadata 推导 vocab/context/heads/layers/norm/rope。
- [x] 从 tensor shape fallback 推导必要 config。
- [x] 识别 token embedding。
- [x] 识别 output norm。
- [x] 识别 lm_head/output。
- [x] 识别每层 attention norm。
- [x] 识别 q/k/v/o projection。
- [x] 识别 FFN norm。
- [x] 识别 gate/up/down projection。
- [x] 支持 Qwen3 特有 q/k norm 或 rope 变体。
- [x] 对缺 tensor/坏 shape/坏 dtype 报 layer 和 tensor name。

验收标准：

- [x] tiny Qwen3-like GGUF 可完整绑定。
- [x] 真实 text-only GGUF 若 unsupported，可输出具体原因。
- [x] 不再只返回笼统 rc=9。

## Phase 8: Qwen3 text-only forward

- [x] token id -> embedding。
- [x] single-token decode path。
- [x] prompt prefill 逐 token path。
- [x] Q/K/V projection。
- [x] RoPE。
- [x] KV cache 写入。
- [x] causal attention。
- [x] output projection。
- [x] FFN SwiGLU。
- [x] final norm。
- [x] lm_head logits。
- [x] greedy sampler。
- [x] `generate <model.gguf> <prompt>` text-only。

验收标准：

- [x] tiny fixture 能输出 logits。
- [x] 固定 prompt greedy token 可复现。
- [x] 与 reference tiny fixture 的 deterministic logits/top-1 对齐。

## Phase 9: Sampler 与文本 CLI

- [x] greedy sampler。
- [x] temperature。
- [x] top-k。
- [x] top-p。
- [x] min-p 或 typical-p，如模型推荐需要。
- [x] repeat penalty。
- [x] seed 管理。
- [x] stop token/stop string。
- [x] `chat` text-only REPL。
- [x] 多轮对话 KV/cache 策略。

验收标准：

- [x] 同 seed 同参数输出一致。
- [x] `chat` 使用 chat template。
- [x] context 溢出有清楚错误或截断策略。

## Phase 10: Vision raw tensor smoke

- [x] 定义 raw image tensor fixture 格式。
- [x] 定义 `VisionConfig`。
- [x] 绑定 vision patch/embedding tensors。
- [x] 绑定 vision transformer tensors。
- [x] 绑定 projector/resampler tensors。
- [x] 实现 patch embedding。
- [x] 实现 vision transformer block。
- [x] 实现 projector 到 LLM hidden。
- [x] 实现 image embedding span 注入 LLM prefill。
- [x] 实现 `vision-smoke <model.gguf> <image.raw>`。

验收标准：

- [x] tiny vision fixture checksum 稳定。
- [x] image placeholder 数量与 embedding span 一致。
- [x] 同 prompt 加/不加图像 logits 有可观测差异。

## Phase 11: Vision preprocessing

- [x] 定义 minimal raw RGB 输入格式。
- [x] 实现 resize reference。
- [x] 实现 crop/pad。
- [x] 实现 normalize。
- [x] 实现 patch/tiling 策略。
- [x] 实现 position embedding interpolation。
- [x] 增加 image manifest。
- [x] 增加视频 frame raw sequence manifest。

验收标准：

- [x] 固定小图预处理输出 golden。
- [x] 与官方 processor 的 shape、tile 数、placeholder 数一致。
- [x] 不依赖 PNG/JPEG/MP4 解码库。

## Phase 12: Audio log-mel/encoder smoke

- [x] 定义 log-mel fixture 格式。
- [x] 定义 `AudioConfig`。
- [x] 绑定 audio conv/front-end tensors。
- [x] 绑定 audio transformer encoder tensors。
- [x] 绑定 audio projector tensors。
- [x] 实现 conv front-end。
- [x] 实现 audio transformer block。
- [x] 实现 projector 到 LLM hidden。
- [x] 实现 audio embedding span 注入 LLM prefill。
- [x] 实现 `audio-smoke <model.gguf> <audio.raw>`。

验收标准：

- [x] tiny log-mel fixture checksum 稳定。
- [x] 固定音频 prompt 可影响 logits。
- [x] 缺 audio 分支时错误明确。

## Phase 13: Audio preprocessing 与 streaming cache

- [x] 实现 PCM16/F32 raw 读取。
- [x] 实现 mono/downmix。
- [x] 实现 resample reference。
- [x] 实现 window function。
- [x] 实现 STFT reference。
- [x] 实现 mel filterbank。
- [x] 实现 log scaling。
- [x] 实现 streaming chunk buffer。
- [x] 实现 overlap/cache reuse。

验收标准：

- PCM -> mel 与 Python reference 对齐。
- streaming chunk 和一次性处理差异在阈值内。
- 长音频不会无限增长 scratch 内存。

## Phase 14: Speech token 输出

- [x] 识别 speech output 分支 schema。
- [x] 定义 `SpeechConfig`。
- [x] 绑定 speech decoder tensors。
- [x] 识别 codec/acoustic token vocab。
- [x] 实现 text -> speech token smoke。
- [x] 实现 speech token decode 中间表示。
- [x] 保存 speech token 序列用于对照。

验收标准：

- tiny speech decoder fixture 输出稳定 token。
- 真实模型若不能 vocoder，也能明确输出 speech token 或 unsupported reason。

## Phase 15: Vocoder/WAV 输出

- [x] 识别 vocoder/Vocos-like tensor schema。
- [x] 实现必要 conv/upsample/norm/activation kernel。
- [x] 实现 fixed speech token -> waveform。
- [x] 实现 WAV writer。
- [x] 实现 `speech-smoke` 输出 wav。
- [x] 增加 RMS/peak/range 验证。

验收标准：

- tiny vocoder fixture 输出 deterministic waveform。
- WAV header、采样率、样本数正确。
- 输出不含 NaN/Inf，幅度可控。

## Phase 16: Omni prompt compiler

- [x] 定义 `PromptEvent`。
- [x] 定义 `CompiledPrompt`。
- [x] 支持 text event。
- [x] 支持 image event。
- [x] 支持 video frame event。
- [x] 支持 audio chunk event。
- [x] 支持 speech request/control event。
- [x] 实现 event -> token/span 编译。
- [x] 实现 position id 和 attention mask policy。
- [x] 实现 `omni-smoke <manifest.json>`。

验收标准：

- manifest 中 text/image/audio 顺序保持正确。
- embedding span 与 placeholder 对齐。
- context 长度计算包含 text token 和 media embedding。

## Phase 17: 阻塞式 omni chat

- [x] 加载一次模型进入 REPL。
- [x] 支持文本输入。
- [x] 支持 manifest 图片输入。
- [x] 支持 manifest 音频输入。
- [x] 支持 text output。
- [x] 支持 optional speech output。
- [x] 多轮对话状态管理。
- [x] turn boundary 和 cache reset 策略。

验收标准：

- text-only 多轮不破坏 KV cache。
- text+image 单轮可生成文本。
- text+audio 单轮可生成文本。
- speech output 若未支持，给明确 unsupported，不影响 text output。

## Phase 18: Streaming/full-duplex runtime

- [x] 输入事件队列。
- [x] audio ring buffer。
- [x] vision/audio encoder worker state。
- [x] LLM decode step scheduler。
- [x] speech synthesis queue。
- [x] backpressure 策略。
- [x] cancellation/interrupt。
- [x] partial output callback。
- [x] `stream-chat` CLI prototype。

验收标准：

- 可处理 chunked audio manifest。
- 支持用户打断并清理 pending speech。
- decode、encode、vocoder 阶段耗时可观测。

## Phase 19: 性能与优化

- [x] 增加 `bench` CLI。
- [x] 报告 load time。
- [x] 报告 text prompt/decode tokens/s。
- [x] 报告 vision encode time。
- [x] 报告 audio encode time。
- [x] 报告 vocoder samples/s。
- [x] 报告峰值内存估算。
- [x] 优化 hot matvec。
- [x] 优化 KV cache locality。
- [x] 优化 prefill tiled path。
- [x] 优化 media encoder scratch reuse。

验收标准：

- benchmark 可在无外部模型时跑 tiny fixture。
- 真实模型 benchmark 通过环境变量手动启用。
- 优化路径与 reference 误差在阈值内。

## Phase 20: 文档、兼容性与发布

- [x] 完善 README usage。
- [x] 记录支持的 MiniCPM-o 版本和文件格式。
- [x] 记录支持 dtype 列表。
- [x] 记录 unsupported modality 列表。
- [x] 记录外部模型 smoke 命令。
- [x] 记录 golden 对照来源和误差阈值。
- [x] 记录性能结果。
- [x] 增加 troubleshooting。
- [x] 增加架构图或数据流图。

验收标准：

- 新用户能按文档完成 build、inspect、audit。
- 每个失败场景都有建议下一步。
- 文档不夸大当前能力。


## Phase 21: 真实 MiniCPM-o 4.5 audio-to-audio 对齐

目标：在纯 Uya runtime 中实现与本地 `llama.cpp-omni` 离线 audio-to-audio demo 等价的功能，而不是只跑 tiny smoke。完整链路必须支持：参考音色音频 + 用户语音问题 -> audio encoder -> Qwen3/MiniCPM-o LLM 生成文本/hidden states -> TTS audio tokens -> token2wav/HiFiGAN -> 24 kHz mono WAV 回答。

### 21.1 基线与输入协议

- [x] 下载并本地跑通官方 `openbmb/MiniCPM-o-4_5-gguf` 分文件 GGUF。
- [x] 编译并跑通 `llama.cpp-omni` 的 `llama-omni-cli` CPU 路径。
- [x] 记录正确离线测试协议：`prefix_0000.wav` 是 reference/system voice，`prefix_0001.wav` 起才是用户输入。
- [x] 生成 `outputs/nihao_case` 和 `outputs/complex_case2` 作为行为基线。
- [x] 记录错误用法风险：只传 `prefix_0000.wav` 会被当作参考音色，容易看起来像复述输入。
- [x] 增加不入库的外部 baseline manifest，记录模型路径、输入 wav、输出 wav、回答文本、运行日志路径。
- [x] 增加 `docs/audio2audio-real.md`，明确 baseline 命令、输入文件命名、输出目录和验收口径。

验收标准：

- `llama.cpp-omni` baseline 至少有两个固定用例：短问候和复杂性能对比问题。
- 每个 baseline 都保存用户输入文本、用户 wav、模型回答文本、回答 wav、拼接 wav、运行日志。
- 文档明确区分 reference audio、user audio、answer audio，不再把 reference 音频误当用户问题。

### 21.2 官方 GGUF audit 与 tensor alias

- [x] Uya `audio2audio-smoke` 支持 split GGUF 参数：`--audio-model`、`--speech-model`、`--vocoder-model`。
- [x] 为官方 `MiniCPM-o-4_5-Q4_K_M.gguf`、`audio/MiniCPM-o-4_5-audio-F16.gguf`、`tts/MiniCPM-o-4_5-tts-F16.gguf`、`tts/MiniCPM-o-4_5-projector-F16.gguf`、`token2wav-gguf/*.gguf` 生成 tensor/metadata inventory。
- [x] 增加 `audit-bundle` 或等价脚本，批量输出每个 GGUF 的 tensor count、dtype distribution、name prefix、shape summary。
- [x] 为 audio encoder 建立官方 tensor alias 表。
- [x] 为 TTS GGUF 建立 `emb_code`、`emb_text`、`projector_semantic`、`projector_spk`、`head_code` 绑定表。
- [x] 为 token2wav 建立 encoder、flow_matching、flow_extra、hifigan2、prompt_cache 绑定表。
- [x] 对缺失 tensor、未知 dtype、shape mismatch 输出具体分支名和候选 alias。

验收标准：

- Uya 能在不执行推理的情况下完整 audit 官方 MiniCPM-o 4.5 GGUF bundle。
- 每个官方 tensor 要么被分类到明确模块，要么被列为 unsupported with reason。
- audit 输出能回答“还差哪些 kernel/binding 才能跑真实 audio-to-audio”。

### 21.3 Qwen3 8B 真实 LLM forward

- [x] 将 Qwen3 forward 上限从 tiny/smoke cap 提升到 MiniCPM-o 4.5 需要的实际尺寸：`hidden=4096`、`layers=36`、`ffn=12288`、`vocab≈151748`、`ctx>=4096`。
- [x] 把大数组从栈上固定数组迁移到 heap/scratch arena，避免 8B 模型运行时栈爆。
- [x] 实现/验证目标官方路径命中的真实 matvec dtype；未命中 dtype 保持显式 unsupported。
  - [x] 运行时区分 parser 支持和 matvec 支持，避免静默错误输出。
  - [x] 实现 GGML 布局的 Q4_K/Q5_K/Q6_K fused matvec，并覆盖 Q4_K/Q5_K/Q6_K token embedding 行反量化。
  - [x] 官方 `MiniCPM-o-4_5-Q4_K_M.gguf` 可执行 text-only `generate --max-new-tokens 1`。
  - [x] 官方 `MiniCPM-o-4_5-Q4_K_M.gguf` 已验证只命中 `F32/Q4_K/Q6_K`；`Q8_K/IQ*` 仍保留 runtime 显式 unsupported，不静默执行。
- [ ] 支持 Qwen3/MiniCPM-o 4.5 的 rope、q/k norm、GQA、KV cache layout 和 chat template。
  - [x] 当前 forward 已包含 q/k norm、GQA 维度、KV cache 和预计算 RoPE 的基本执行路径，能跑官方模型 smoke。
  - [ ] 与 llama.cpp 对齐 RoPE scaling/chat template/stop token 细节。
- [x] 支持 prompt prefill 分块、decode step、sampler、stop token、hidden state capture。
  - [x] text-only prompt prefill、decode step、sampler 和 stop token smoke 可执行。
  - [x] `generate --dump-hidden` 已接出 prompt/generated token 的 LLM hidden state summary，供后续 TTS projector 消费。
- [x] 增加 text-only 对齐用例：`make text-real-align` 用同一 prompt 对照 Uya 与 llama.cpp `llama-completion` 的 greedy 1-token 输出。

验收标准：

- [x] Uya 能加载官方 `MiniCPM-o-4_5-Q4_K_M.gguf` 并完成 text-only prefill/decode smoke。
- 同一 text prompt 下，Uya 与 llama.cpp 的前若干 token 或 top-k logits 在可解释误差内。
- 运行时内存分配、KV cache bytes、load time、prefill/decode tokens/s 可观测。

### 21.4 Audio encoder 与用户语音注入

- [x] 支持真实 PCM/WAV 读取：16 kHz mono s16/f32，并明确拒绝不支持格式或自动转码路径。
  - [x] `audio-input-probe` 支持 RIFF/WAVE PCM16、RIFF/WAVE F32 和现有 UYAP PCM；非 16 kHz mono 明确报错，不自动转码。
- [ ] 对齐官方 audio preprocessing：window、STFT、mel、chunking、padding、streaming cache。
  - [x] `audio-real-preprocess-probe` 已按 llama.cpp-omni MiniCPM-o 路径输出参数 plan：`frame_size=400`、`filter_bins=201`、`hop_length=160`、`mel_bins=80`、100ms 对齐、center pad 200 samples/side、conv2 下采样和 pool(5,5) 后的 `encoder_positions`。
  - [x] `audio2audio-real --audit-only` 已在 ref/user 输入检查后追加 preprocessing plan，支持显式 ref/user 和 `prefix_0000.wav`/`prefix_0001.wav` 测试格式。
  - [x] `audio-bind` 已验证官方 audio GGUF 的 `filters` metadata：`n_mel=80`、`n_fft/filter_bins=201`、`filters=16080`。
  - [x] `audio-real-mel-probe` 已实现真实 WAV/UYAP -> periodic Hann -> DFT/STFT power -> GGUF `filters` mel filterbank -> log10 clamp/normalize，并输出 frames/elements/checksum/首值。
  - [x] `audio-real-mel-probe --dump-f32` 与 `tests/compare_audio_mel_alignment.py`/`make audio-real-mel-align` 已接好，可对比 `llama.cpp-omni` 的 `log_mel_spectrogram.json` dump。
  - [x] 已与本地 `llama.cpp-omni` `log_mel_spectrogram` dump 做数值误差对齐：`outputs/complex_case2/complex2_0000.wav` 为 `mean_abs=1.2707e-5`、`max_abs=1.4266e-3`，`complex2_0001.wav` 为 `mean_abs=1.2695e-5`、`max_abs=1.4004e-3`；当前默认阈值取 `mean_abs <= 2e-5`、`max_abs <= 2e-3`。
- [ ] 移除 tiny audio cap，支持真实用户语音长度和多 chunk 输入。
  - [x] 输入 probe 流式扫描真实 WAV/UYAP，不受 tiny mel cap 限制；`audio-real-encode-probe` 现已可跑真实 audio encoder forward，但当前仍保留 `480000 samples` 上限且没有多 chunk 调度。
- [ ] 绑定官方 audio encoder tensors，并实现对应 conv/transformer/projector forward。
  - [x] `audio-bind`/`audio2audio-real --audit-only` 已绑定官方 `encoder.conv*`、24 层 `encoder.blocks.*`、`encoder.ln_post.*`、`audio_projector.linear{1,2}.*`，共 371 个 tensor。
  - [x] `audio-real-encode-probe` 已实现官方 `conv + transformer + projector + pool(5,5)` forward，并输出 `mel_frames/conv_tokens/n_pos/n_embd/checksum/encode_ms`。
- [ ] 实现正确 prompt 协议：reference audio 进入 system prompt，user audio 进入 `<|im_start|>user` turn。
  - [x] `audio2audio-real --audit-only` 已区分 `ref_audio` 和 `user_audio`，并在协议日志中输出两者路径。
- [x] 支持 `prefix_0000.wav`/`prefix_0001.wav` 测试格式，也支持显式 `--ref-audio`/`--user-audio` 参数。
- [ ] 保存 audio embedding checksum、span count、n_pos、prefill timing 便于与 llama.cpp-omni 对照。
  - [x] 输入阶段已保存 sample checksum、duration、peak、rms；`audio-real-encode-probe` 与 `audio2audio-real --audit-only --encode-probe` 已可输出 embedding checksum、`n_pos`、`n_embd` 和 encode wall time。

验收标准：

- 同一 user wav 下，Uya audio embedding 的 shape/n_pos 与 llama.cpp-omni 记录一致或差异可解释。
- `0000=ref, 0001=user` 的复杂用例不再复述输入，而能进入回答路径。
- 缺 audio 分支或音频太短/太长时有清晰诊断。

### 21.5 LLM -> TTS audio token 生成

- [x] 实现 TTS GGUF 权重加载/绑定审计：`emb_code.0.weight`、`emb_text.weight`、`projector_semantic.*`、`projector_spk.*`、`head_code.0.weight`。
  - [x] `tts-bind`/`audio2audio-real --audit-only` 已绑定官方 TTS decoder 20 层 `blk.*`、embedding/projector/head，共 193 个 tensor。
- [ ] 实现 LLM hidden states 到 TTS conditioning embedding 的 projector path。
- [ ] 实现 TTS prefill/decode cache，与 audio_bos、audio_eos、audio token vocabulary 对齐。
- [ ] 实现 chunked text/hidden-state queue：LLM 每个文本 chunk 同步送入 TTS。
- [ ] 生成 audio token ids 并保存 `audio_tokens_chunk_*.txt/bin`，格式对齐 llama.cpp-omni。
- [ ] 增加 TTS token 级对照：相同 hidden/text chunk 下，audio token 前 N 个与 baseline 对比。

验收标准：

- Uya 能从 LLM 回答文本/hidden states 生成非空 audio token 序列。
- audio token 分片数量、EOS 行为、首 token/audio_bos 行为与 llama.cpp-omni 可对照。
- TTS 阶段输出耗时、token count、cache length 可观测。

### 21.6 Token2Wav/HiFiGAN 真实声码器

- [ ] 解析并绑定 `token2wav-gguf/encoder.gguf`。
- [ ] 解析并绑定 `token2wav-gguf/flow_matching.gguf`。
- [ ] 解析并绑定 `token2wav-gguf/flow_extra.gguf`。
- [ ] 解析并绑定 `token2wav-gguf/hifigan2.gguf`。
- [ ] 解析并应用 `token2wav-gguf/prompt_cache.gguf`，避免每次实时重算参考音色 cache。
- [ ] 实现 token2mel/flow matching 推理图需要的 attention、conv、norm、sampling/noise schedule。
- [ ] 实现 hifigan2 vocoder forward，输出 24 kHz mono PCM。
- [ ] 实现流式 WAV chunk 写入，并支持最终 concat 成完整回答 wav。
- [ ] 支持 CPU reference 后端；GPU/多线程优化另设后续任务。

验收标准：

- Uya 可把 TTS audio tokens 转成可播放 WAV。
- WAV header、采样率、通道数、duration、RMS/peak 合法。
- 与 llama.cpp-omni 同用例的回答音频时长、chunk 数、首响时间指标可对照。

### 21.7 `audio2audio-real` CLI

- [x] 新增真实 CLI，不复用 smoke 名称：`audio2audio-real`。
- [x] 支持显式分文件参数：`--llm`、`--audio`、`--tts`、`--projector`、`--token2wav-dir`。
- [x] 支持输入参数：`--ref-audio`、`--user-audio`/`--input-audio`、`--out`。
- [x] 支持 llama.cpp-omni 测试格式：`--test-prefix PREFIX --count N`，其中 `0000` 是 ref，`0001..` 是 user turn。
  - [x] 支持单轮 `--input-prefix PREFIX`，自动解析 `PREFIX_0000.wav` 为 ref、`PREFIX_0001.wav` 为 user。
  - [x] `--test-prefix PREFIX --count N` 会逐个 probe `PREFIX_0001.wav..PREFIX_%04u.wav`，用于多 user turn 输入审计。
- [ ] 输出回答文本、answer wav、turn wav、audio token chunks、timing log。
- [ ] 增加 `--text-only`、`--no-tts`、`--dump-hidden`、`--dump-embeddings` 诊断模式。
- [ ] 所有真实模型命令默认要求显式路径，不从仓库内隐式下载模型。

验收标准：

- 以下命令能生成回答 WAV：

```sh
build/minicpm-o-uya audio2audio-real \
  --llm models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --audio models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf \
  --tts models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  --projector models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  --token2wav-dir models/MiniCPM-o-4_5-gguf/token2wav-gguf \
  --ref-audio outputs/complex_case2/complex2_0000.wav \
  --user-audio outputs/complex_case2/complex2_0001.wav \
  --out outputs/uya_complex_answer.wav
```

- 输出目录包含 `answer.txt`、`answer.wav`、`turn.wav`、`timing.log`、必要 debug dumps。
- 错误用法如只传 `0000.wav` 时给出“这是 reference audio，不是 user audio”的诊断。

### 21.8 性能对齐与回归

- [ ] 为真实 audio-to-audio 增加 benchmark 指标：load time、audio prefill、LLM prefill、first audio response、total wall time、peak RSS、answer duration、RTF。
- [ ] 与 `llama.cpp-omni` 同用例对照，记录 CPU-only 基线。
- [ ] 增加长音频、短音频、静音、中文、英文、中英混合、复杂多项要求用例。
- [ ] 增加 deterministic smoke 保持无模型 CI 可跑，真实模型测试只在显式环境变量启用。
- [ ] 文档记录当前速度预期：第一版先正确，再优化，不承诺立即达到 llama.cpp-omni 性能。

验收标准：

- `make test` 仍只跑 tiny fixture。
- `MINICPM_O_REAL_BUNDLE=/path make audio-real-bind` 可手动跑官方 audio encoder bind-only。
- `MINICPM_O_REAL_BUNDLE=/path make tts-real-bind` 可手动跑官方 TTS bind-only。
- `MINICPM_O_REAL_BUNDLE=/path make audio2audio-real-audit` 可手动跑真实模型。
- 每个性能回归都有日志和指标，便于比较 Uya 与 llama.cpp-omni。


## 本机/外部模型约定

不要把模型权重放进仓库。使用环境变量指向外部路径：

```text
MINICPM_O_GGUF=/path/to/minicpm-o.gguf
MINICPM_O_TEXT_GGUF=/path/to/MiniCPM-o-4_5-Q4_K_M.gguf
MINICPM_O_AUDIO_GGUF=/path/to/audio/MiniCPM-o-4_5-audio-F16.gguf
MINICPM_O_TTS_GGUF=/path/to/tts/MiniCPM-o-4_5-tts-F16.gguf
MINICPM_O_PROJECTOR_GGUF=/path/to/tts/MiniCPM-o-4_5-projector-F16.gguf
MINICPM_O_TOKEN2WAV_DIR=/path/to/token2wav-gguf
MINICPM_O_REF_AUDIO=/path/to/ref_0000.wav
MINICPM_O_USER_AUDIO=/path/to/user_0001.wav
MINICPM_O_IMAGE_RAW=/path/to/image.raw
MINICPM_O_AUDIO_RAW=/path/to/audio.raw
MINICPM_O_MANIFEST=/path/to/omni-manifest.json
MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf
```

默认 `make test` 只跑 tiny fixture。真实模型 smoke 使用单独 target，且必须 documented。模型权重、`outputs/` 生成音频、`llama.cpp*` 工作树和下载缓存都不应提交。
