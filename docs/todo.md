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

- [ ] F32 vector fill/copy/add/mul/dot。
- [ ] F16 load -> F32。
- [ ] BF16 load -> F32。
- [ ] RMSNorm。
- [ ] LayerNorm。
- [ ] RoPE。
- [ ] Softmax/masked softmax。
- [ ] Dense F32/F16/BF16 matvec。
- [ ] SwiGLU/SiLU/GELU。
- [ ] Conv1D reference。
- [ ] Conv2D reference。
- [ ] Tiny golden tests。

验收标准：

- 每个 kernel 有小尺寸 deterministic golden。
- NaN/Inf/空长度边界处理明确。
- scalar reference 输出稳定。

## Phase 6: Quantization kernels

- [ ] 根据 MiniCPM-o GGUF audit 选择首批 dtype。
- [ ] 实现 Q8_0 dot。
- [ ] 实现 Q4_K dot。
- [ ] 实现 Q5_K dot。
- [ ] 实现 Q6_K dot。
- [ ] 实现必要 IQ dtype dot。
- [ ] 实现 fused dequant + dot。
- [ ] 支持 quant matvec row stride。
- [ ] 与 scalar dequant reference 对照。

验收标准：

- 支持真实模型中 text decoder 命中的首批 dtype。
- quant kernel 误差在阈值内。
- unsupported dtype 报 tensor name 和 dtype。

## Phase 7: Qwen3 text config 与权重绑定

- [ ] 建立 `Qwen3Config`。
- [ ] 从 GGUF metadata 推导 vocab/context/heads/layers/norm/rope。
- [ ] 从 tensor shape fallback 推导必要 config。
- [ ] 识别 token embedding。
- [ ] 识别 output norm。
- [ ] 识别 lm_head/output。
- [ ] 识别每层 attention norm。
- [ ] 识别 q/k/v/o projection。
- [ ] 识别 FFN norm。
- [ ] 识别 gate/up/down projection。
- [ ] 支持 Qwen3 特有 q/k norm 或 rope 变体。
- [ ] 对缺 tensor/坏 shape/坏 dtype 报 layer 和 tensor name。

验收标准：

- tiny Qwen3-like GGUF 可完整绑定。
- 真实 text-only GGUF 若 unsupported，可输出具体原因。
- 不再只返回笼统 rc=9。

## Phase 8: Qwen3 text-only forward

- [ ] token id -> embedding。
- [ ] single-token decode path。
- [ ] prompt prefill 逐 token path。
- [ ] Q/K/V projection。
- [ ] RoPE。
- [ ] KV cache 写入。
- [ ] causal attention。
- [ ] output projection。
- [ ] FFN SwiGLU。
- [ ] final norm。
- [ ] lm_head logits。
- [ ] greedy sampler。
- [ ] `generate <model.gguf> <prompt>` text-only。

验收标准：

- tiny fixture 能输出 logits。
- 固定 prompt greedy token 可复现。
- 与官方/参考实现 top-k logits 对齐到文档阈值。

## Phase 9: Sampler 与文本 CLI

- [ ] greedy sampler。
- [ ] temperature。
- [ ] top-k。
- [ ] top-p。
- [ ] min-p 或 typical-p，如模型推荐需要。
- [ ] repeat penalty。
- [ ] seed 管理。
- [ ] stop token/stop string。
- [ ] `chat` text-only REPL。
- [ ] 多轮对话 KV/cache 策略。

验收标准：

- 同 seed 同参数输出一致。
- `chat` 使用 chat template。
- context 溢出有清楚错误或截断策略。

## Phase 10: Vision raw tensor smoke

- [ ] 定义 raw image tensor fixture 格式。
- [ ] 定义 `VisionConfig`。
- [ ] 绑定 vision patch/embedding tensors。
- [ ] 绑定 vision transformer tensors。
- [ ] 绑定 projector/resampler tensors。
- [ ] 实现 patch embedding。
- [ ] 实现 vision transformer block。
- [ ] 实现 projector 到 LLM hidden。
- [ ] 实现 image embedding span 注入 LLM prefill。
- [ ] 实现 `vision-smoke <model.gguf> <image.raw>`。

验收标准：

- tiny vision fixture checksum 稳定。
- image placeholder 数量与 embedding span 一致。
- 同 prompt 加/不加图像 logits 有可观测差异。

## Phase 11: Vision preprocessing

- [ ] 定义 minimal raw RGB 输入格式。
- [ ] 实现 resize reference。
- [ ] 实现 crop/pad。
- [ ] 实现 normalize。
- [ ] 实现 patch/tiling 策略。
- [ ] 实现 position embedding interpolation。
- [ ] 增加 image manifest。
- [ ] 增加视频 frame raw sequence manifest。

验收标准：

- 固定小图预处理输出 golden。
- 与官方 processor 的 shape、tile 数、placeholder 数一致。
- 不依赖 PNG/JPEG/MP4 解码库。

## Phase 12: Audio log-mel/encoder smoke

- [ ] 定义 log-mel fixture 格式。
- [ ] 定义 `AudioConfig`。
- [ ] 绑定 audio conv/front-end tensors。
- [ ] 绑定 audio transformer encoder tensors。
- [ ] 绑定 audio projector tensors。
- [ ] 实现 conv front-end。
- [ ] 实现 audio transformer block。
- [ ] 实现 projector 到 LLM hidden。
- [ ] 实现 audio embedding span 注入 LLM prefill。
- [ ] 实现 `audio-smoke <model.gguf> <audio.raw>`。

验收标准：

- tiny log-mel fixture checksum 稳定。
- 固定音频 prompt 可影响 logits。
- 缺 audio 分支时错误明确。

## Phase 13: Audio preprocessing 与 streaming cache

- [ ] 实现 PCM16/F32 raw 读取。
- [ ] 实现 mono/downmix。
- [ ] 实现 resample reference。
- [ ] 实现 window function。
- [ ] 实现 STFT reference。
- [ ] 实现 mel filterbank。
- [ ] 实现 log scaling。
- [ ] 实现 streaming chunk buffer。
- [ ] 实现 overlap/cache reuse。

验收标准：

- PCM -> mel 与 Python reference 对齐。
- streaming chunk 和一次性处理差异在阈值内。
- 长音频不会无限增长 scratch 内存。

## Phase 14: Speech token 输出

- [ ] 识别 speech output 分支 schema。
- [ ] 定义 `SpeechConfig`。
- [ ] 绑定 speech decoder tensors。
- [ ] 识别 codec/acoustic token vocab。
- [ ] 实现 text -> speech token smoke。
- [ ] 实现 speech token decode 中间表示。
- [ ] 保存 speech token 序列用于对照。

验收标准：

- tiny speech decoder fixture 输出稳定 token。
- 真实模型若不能 vocoder，也能明确输出 speech token 或 unsupported reason。

## Phase 15: Vocoder/WAV 输出

- [ ] 识别 vocoder/Vocos-like tensor schema。
- [ ] 实现必要 conv/upsample/norm/activation kernel。
- [ ] 实现 fixed speech token -> waveform。
- [ ] 实现 WAV writer。
- [ ] 实现 `speech-smoke` 输出 wav。
- [ ] 增加 RMS/peak/range 验证。

验收标准：

- tiny vocoder fixture 输出 deterministic waveform。
- WAV header、采样率、样本数正确。
- 输出不含 NaN/Inf，幅度可控。

## Phase 16: Omni prompt compiler

- [ ] 定义 `PromptEvent`。
- [ ] 定义 `CompiledPrompt`。
- [ ] 支持 text event。
- [ ] 支持 image event。
- [ ] 支持 video frame event。
- [ ] 支持 audio chunk event。
- [ ] 支持 speech request/control event。
- [ ] 实现 event -> token/span 编译。
- [ ] 实现 position id 和 attention mask policy。
- [ ] 实现 `omni-smoke <manifest.json>`。

验收标准：

- manifest 中 text/image/audio 顺序保持正确。
- embedding span 与 placeholder 对齐。
- context 长度计算包含 text token 和 media embedding。

## Phase 17: 阻塞式 omni chat

- [ ] 加载一次模型进入 REPL。
- [ ] 支持文本输入。
- [ ] 支持 manifest 图片输入。
- [ ] 支持 manifest 音频输入。
- [ ] 支持 text output。
- [ ] 支持 optional speech output。
- [ ] 多轮对话状态管理。
- [ ] turn boundary 和 cache reset 策略。

验收标准：

- text-only 多轮不破坏 KV cache。
- text+image 单轮可生成文本。
- text+audio 单轮可生成文本。
- speech output 若未支持，给明确 unsupported，不影响 text output。

## Phase 18: Streaming/full-duplex runtime

- [ ] 输入事件队列。
- [ ] audio ring buffer。
- [ ] vision/audio encoder worker state。
- [ ] LLM decode step scheduler。
- [ ] speech synthesis queue。
- [ ] backpressure 策略。
- [ ] cancellation/interrupt。
- [ ] partial output callback。
- [ ] `stream-chat` CLI prototype。

验收标准：

- 可处理 chunked audio manifest。
- 支持用户打断并清理 pending speech。
- decode、encode、vocoder 阶段耗时可观测。

## Phase 19: 性能与优化

- [ ] 增加 `bench` CLI。
- [ ] 报告 load time。
- [ ] 报告 text prompt/decode tokens/s。
- [ ] 报告 vision encode time。
- [ ] 报告 audio encode time。
- [ ] 报告 vocoder samples/s。
- [ ] 报告峰值内存估算。
- [ ] 优化 hot matvec。
- [ ] 优化 KV cache locality。
- [ ] 优化 prefill tiled path。
- [ ] 优化 media encoder scratch reuse。

验收标准：

- benchmark 可在无外部模型时跑 tiny fixture。
- 真实模型 benchmark 通过环境变量手动启用。
- 优化路径与 reference 误差在阈值内。

## Phase 20: 文档、兼容性与发布

- [ ] 完善 README usage。
- [ ] 记录支持的 MiniCPM-o 版本和文件格式。
- [ ] 记录支持 dtype 列表。
- [ ] 记录 unsupported modality 列表。
- [ ] 记录外部模型 smoke 命令。
- [ ] 记录 golden 对照来源和误差阈值。
- [ ] 记录性能结果。
- [ ] 增加 troubleshooting。
- [ ] 增加架构图或数据流图。

验收标准：

- 新用户能按文档完成 build、inspect、audit。
- 每个失败场景都有建议下一步。
- 文档不夸大当前能力。

## 本机/外部模型约定

不要把模型权重放进仓库。使用环境变量指向外部路径：

```text
MINICPM_O_GGUF=/path/to/minicpm-o.gguf
MINICPM_O_TEXT_GGUF=/path/to/qwen3-text.gguf
MINICPM_O_IMAGE_RAW=/path/to/image.raw
MINICPM_O_AUDIO_RAW=/path/to/audio.raw
MINICPM_O_MANIFEST=/path/to/omni-manifest.json
```

默认 `make test` 只跑 tiny fixture。真实模型 smoke 使用单独 target，且必须 documented。
