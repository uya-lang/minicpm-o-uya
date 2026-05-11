# minicpm-o-uya 纯 Uya 实现设计文档

## 1. 项目目标

`minicpm-o-uya` 的目标是用纯 Uya 实现一个 CPU-first 的 MiniCPM-o
端侧推理 runtime。项目不是把 OpenBMB/MiniCPM-o 的 Python/PyTorch 代码逐行翻译，
也不是调用 llama.cpp-omni 或官方 server；而是按模型推理链路重新实现文件读取、
tokenizer、张量视图、量化 kernel、多模态编码、LLM decode、语音输出、采样和 CLI。

初期目标是可验证、可分阶段落地：

1. 先能读取模型文件并做 schema audit。
2. 再跑通 text-only Qwen3 decoder。
3. 然后接视觉输入。
4. 再接音频输入。
5. 最后接语音输出和流式 omni chat。

任何阶段都必须明确说明当前支持的 modality、dtype、布局和限制；不能把
`inspect/audit/encode` 误称为完整 MiniCPM-o 推理。

## 2. 参考对象和边界

当前主要参考目标是 MiniCPM-o 4.5。公开模型卡描述它是端到端多模态模型，核心组件包括：

- 文本骨干：Qwen3-8B。
- 视觉编码：SigLIP2 类视觉 encoder。
- 音频输入：Whisper-medium 类 audio encoder。
- 语音输出：CosyVoice2 类 speech generation，加 vocoder/codec 路径。
- 多模态能力：图像、视频、文本、音频输入，以及文本/语音输出。
- 端侧参考：官方提供 llama.cpp-omni 作为低资源设备推理路线参考。

`minicpm-o-uya` 的边界：

- 可以阅读官方 Python、配置、模型卡、转换脚本和 llama.cpp-omni 行为作为参考。
- 不把 Python、PyTorch、Transformers、llama.cpp、ONNX Runtime、FFmpeg、WebRTC 作为运行依赖。
- 不在仓库里保存大模型、原始数据集、大音视频样本或下载产物。
- 初期接受外部工具预处理出的 raw tensor 作为图像/音频 smoke fixture，避免把媒体解码和模型推理混在第一阶段。
- 允许设计未来 FFI/系统 IO 边界，但默认实现路径必须是纯 Uya。

## 3. 当前机器与运行约束

默认开发目标：

- OS/arch：Linux x86_64。
- CPU-first：先 scalar reference，再加 x86_64 SIMD fast path。
- 内存友好：大权重使用只读 mmap，避免逐 tensor 复制和全量 dequant。
- 磁盘友好：模型路径来自外部，不进入 git。
- 可测试：每个模块先用 tiny fixture 和 golden 值验证。

多模态运行时会遇到额外约束：

- 视觉输入有 resize、normalize、patch/tiling、位置编码和 projector。
- 视频输入需要帧抽取和时间维采样；初期不实现视频解码，只实现 raw frame 序列。
- 音频输入需要重采样、mel filterbank、分块和 encoder state；初期先吃 raw PCM 或 log-mel fixture。
- 语音输出涉及 codec token、声码器和 streaming buffer；初期先输出 acoustic/vocoder 中间结果或 tiny wav fixture。
- full-duplex 需要调度器、环形缓冲、KV cache 增量更新和 backpressure；这是最后阶段。

## 4. 总体架构

建议模块划分：

```text
src/main.uya                 CLI 入口
src/minicpmo/binary.uya      little-endian 读取、文件工具
src/minicpmo/gguf.uya        GGUF/metadata/tensor directory loader
src/minicpmo/config.uya      MiniCPM-o/Qwen3/SigLIP/Whisper/CosyVoice 配置
src/minicpmo/tokenizer.uya   tokenizer、special token、chat template
src/minicpmo/tensor.uya      dtype、TensorView、WeightTable、mmap 权重
src/minicpmo/kernels.uya     F32/F16/BF16/quant matvec、norm、softmax、conv
src/minicpmo/qwen3.uya       Qwen3 text decoder forward
src/minicpmo/vision.uya      SigLIP2 vision tower 和 image projector
src/minicpmo/audio.uya       PCM/log-mel、Whisper-style audio encoder
src/minicpmo/speech.uya      CosyVoice-style speech decoder、codec/vocoder
src/minicpmo/omni.uya        多模态 token 拼接、状态机、stream scheduler
src/minicpmo/runtime.uya     Session、KV cache、modality cache、generate/chat
src/minicpmo/sampler.uya     greedy、temperature、top-k/top-p、repeat penalty
src/minicpmo/bench.uya       text/vision/audio/speech benchmark
```

CLI 初始命令：

```text
--help
inspect <model.gguf>
audit <model.gguf>
tensor <model.gguf> <name>
view <model.gguf> <name>
piece <model.gguf> <token-id>
encode <model.gguf> <text>
decode <model.gguf> <token-ids...>
format-chat <model.gguf> <text>
generate <model.gguf> <text>
vision-smoke <model.gguf> <image.raw>
audio-smoke <model.gguf> <audio.raw>
speech-smoke <model.gguf> <text-or-tokens>
omni-smoke <model.gguf> <manifest.json>
chat <model.gguf>
bench
```

## 5. 模型文件与权重策略

### 5.1 支持格式优先级

第一优先级是 GGUF，因为它适合端侧 runtime：metadata、tokenizer、tensor directory 和
权重 data 都在单文件中，便于 audit 和 mmap。若官方或社区 GGUF 不完整，则分阶段支持：

1. `inspect/audit` 任意 GGUF：只读 header、metadata、tensor directory。
2. Qwen3 text-only GGUF：要求具备 tokenizer、root tensors 和 decoder layers。
3. MiniCPM-o multimodal GGUF：要求视觉/audio/speech 分支 tensor 命名可识别。
4. safetensors/hf-index：只作为后续转换或离线验证输入，不作为初始 runtime 格式。

### 5.2 GGUF loader 设计

GGUF loader 要记录：

- 文件 magic/version、tensor_count、metadata_count、alignment、data_offset、file_size。
- 所有 metadata key/value 的 type、长度、offset、必要采样。
- 完整 tensor table：name、dtype、shape、n_dims、relative offset、absolute offset、byte size。
- tokenizer metadata：tokens、merges、scores、token types、BOS/EOS/UNK/PAD、chat template。
- 模型 metadata：architecture、block_count、context_length、embedding_length、head_count、rope、norm eps。
- 多模态 metadata：vision/audio/speech 分支配置，若 GGUF 使用自定义 key，需要纳入 alias 表。
- 截断诊断：区分 header、metadata、tensor directory、tensor data 哪一段截断。

`audit` 不读取 tensor data；`load runtime` 才 mmap tensor data。

### 5.3 权重内存策略

- 权重默认只读 mmap。
- `TensorView.data` 指向 mmap 中的 tensor data，不复制。
- 不在 load 阶段做全量 dequant。
- 激活使用 arena/scratch buffer，按层复用。
- KV cache、vision cache、audio cache、speech cache 是长期状态，单独分配。
- 对大 vocab logits 提供 scratch reuse 和 top-k partial scan，避免额外大分配。

## 6. Tokenizer 与 prompt 模板

MiniCPM-o text backbone 使用 Qwen3 系 tokenizer 语义时，需要覆盖：

- tokenizer vocab 与 merge/rank 表。
- added tokens 与 special tokens。
- BOS/EOS/PAD/UNK。
- ChatML/Qwen-style role token、system/user/assistant/tool 片段。
- 多模态占位 token：image、video、audio、speech、media placeholder。
- `tokenizer.chat_template` 或官方 processor 的等价格式化规则。
- encode/decode roundtrip。
- allow/disallow special token 策略。

实现策略：

1. 先实现 GGUF tokenizer 元数据加载。
2. 对 Qwen3 tokenizer 做固定 prompt golden：英文、中文、符号、换行、特殊 token。
3. 实现 `format-chat`，输出纯文本模板结果。
4. 增加 multimodal placeholder formatter，例如 `<image>`、`<audio>` 或 GGUF 中真实 special token。
5. 对齐官方 processor 的 token 序列，而不是只比较可读字符串。

## 7. Tensor、dtype 与 kernel 策略

### 7.1 DType 覆盖

基础 dtype：

- F32、F16、BF16。
- I8/I16/I32/I64，用于索引、token、位置、mask 等。
- Q8_0、Q4_K、Q5_K、Q6_K、IQ 系列，根据实际 GGUF dtype 逐步补齐。
- 若 MiniCPM-o GGUF 使用自定义 quantization，需要先 audit 结构，再实现 reference kernel。

### 7.2 Kernel 分层

先实现正确性：

- scalar F32 vector ops。
- F16/BF16 load 转 F32。
- RMSNorm/LayerNorm。
- RoPE/Qwen3 RoPE scaling。
- Softmax/masked softmax。
- Dense matvec。
- Quant block dot。
- SwiGLU/SiLU/GELU。
- Conv1D/Conv2D reference，用于 vision/audio/speech 分支。
- Mel filterbank 和 STFT reference，如果选择在 Uya 内实现音频预处理。

再实现优化：

- AVX2/F16C/BF16 可用时的 dot fast path。
- fused dequant + dot。
- KV cache cache-friendly row access。
- Qwen3 prefill batch matmul 的简化 tiled path。
- Vision patch embedding/attention 的 contiguous layout。
- Audio streaming encoder 的 block reuse。

所有 optimized kernel 都必须保留 scalar reference 对照。

## 8. Text-only Qwen3 decoder

Text-only 是第一条真正 generation 路径。目标是先支持 MiniCPM-o 中的 Qwen3-8B text backbone，
不接图像、音频和语音输出。

### 8.1 配置

`Qwen3Config` 应包含：

- vocab_size。
- hidden_size。
- intermediate_size。
- num_hidden_layers。
- num_attention_heads。
- num_key_value_heads。
- head_dim。
- max_position_embeddings/context_length。
- rope_theta、rope scaling。
- rms_norm_eps。
- tie_word_embeddings。
- sliding window 或 attention mask 变种。
- dtype/quant policy。

### 8.2 权重绑定

需要识别 Qwen/Qwen3 常见 tensor 命名：

- token embedding。
- final norm。
- lm_head/output。
- per-layer input norm / post attention norm。
- q/k/v/o projection。
- gate/up/down projection。
- optional q/k norm。
- optional rope scaling metadata。

权重绑定必须报具体错误：缺 tensor、shape 不匹配、dtype 不支持、层号不连续。

### 8.3 Forward 流程

单 token decode：

1. token id -> embedding。
2. 对每层执行 attention norm。
3. Q/K/V projection。
4. RoPE。
5. 写 KV cache。
6. causal attention。
7. output projection + residual。
8. FFN norm。
9. SwiGLU MLP。
10. residual。
11. final norm。
12. lm_head -> logits。
13. sampler -> next token。

Prefill 初期可以逐 token，后续再 batch/tiled。

### 8.4 验证

- tiny Qwen-like fixture logits。
- 固定 prompt greedy token 序列。
- 与 Transformers/llama.cpp-omni 同 prompt top-k logits 对照。
- dtype reference 与 quant path 误差阈值。

## 9. 视觉输入：SigLIP2 + projector

视觉路径目标是把图片/视频帧编码成 LLM 可消费的视觉 embedding 或 media token 序列。

### 9.1 输入策略

分三档：

1. Raw tensor smoke：输入已 resize/normalize 的 `float32 CHW` 或 `NHWC`。
2. Minimal image preprocess：实现 resize、center crop、normalize、patchify。
3. Full image/video preprocess：支持动态分辨率、tiling、frame sampling、视频 manifest。

初期不实现 PNG/JPEG/MP4 解码；使用外部工具生成 raw fixture。

### 9.2 Vision tower

需要实现：

- patch embedding 或 conv stem。
- vision transformer blocks。
- attention、MLP、norm。
- position embedding / 2D position interpolation。
- pooling/select tokens。
- projector/resampler，把 vision hidden 映射到 Qwen3 hidden。

### 9.3 注入 LLM

两种可能布局都要支持 audit：

- processor 生成 media placeholder token，runtime 在 embedding 阶段替换对应 token embedding。
- 模型将 vision embeddings 拼到文本 embedding 序列前/中间。

实现时建立 `MultimodalPrompt`：

```text
segments = [text tokens, image embedding spans, audio embedding spans, text tokens]
```

LLM prefill 接收 embedding 序列而不仅是 token id 序列。

### 9.4 验证

- 固定 raw image tensor 的 vision embedding checksum。
- tiny SigLIP-like transformer fixture。
- image placeholder 数量与 embedding span 长度一致。
- 与官方 processor/llama.cpp-omni 对齐首轮 logits top-k。

## 10. 音频输入：Whisper-style encoder

音频输入路径目标是把用户语音编码成 LLM 可消费的 audio embedding。

### 10.1 输入策略

分三档：

1. log-mel fixture：绕过 STFT，直接测试 encoder。
2. raw PCM fixture：实现重采样、分帧、窗函数、STFT、mel filterbank、log scaling。
3. streaming PCM：环形缓冲、chunk overlap、增量 encoder state。

### 10.2 Audio encoder

需要实现：

- conv front-end。
- positional embedding。
- transformer encoder blocks。
- norm/pooling/downsample。
- projector 到 Qwen3 hidden。
- VAD 或 turn boundary 只作为后续功能，不放在首版 correctness 路径。

### 10.3 与 LLM 集成

类似视觉路径，构造 audio embedding span 并注入 LLM prefill。需要记录：

- audio span 对应原始时间范围。
- streaming chunk index。
- 是否允许跨 chunk cache。
- 音频输入和文本输入的交错顺序。

### 10.4 验证

- tiny log-mel encoder golden。
- PCM -> mel golden。
- 固定音频 prompt 的 audio embedding checksum。
- 与官方 processor 对齐 token/embedding span 长度。

## 11. 语音输出：CosyVoice-style decoder 与 vocoder

语音输出是 MiniCPM-o 与普通 VLM/LLM 的核心差异之一。纯 Uya 实现必须分层推进。

### 11.1 输出分层

1. Text-only 输出：只生成文字。
2. Speech token 输出：生成 codec/acoustic token，不合成 waveform。
3. Vocoder smoke：把固定 speech token 转成 waveform。
4. Full speech：LLM 或 speech decoder 生成 token，再 vocoder 输出 wav。
5. Streaming speech：边生成边合成，支持 chunk buffer。

### 11.2 Speech 模块

根据实际模型文件确认后实现：

- speech tokenizer/codec token 表。
- acoustic decoder transformer/flow matching/其他结构。
- speaker embedding 或 style prompt。
- length regulator / duration / prosody 控制。
- vocoder，例如 Vocos 类路径。
- wav writer，初期只写 PCM WAV，不做实时播放。

### 11.3 验证

- 固定 speech token -> mel/acoustic hidden checksum。
- vocoder tiny fixture waveform checksum。
- 输出 wav header 正确。
- 与官方示例的短句音频长度、采样率、RMS 范围对齐。

## 12. Omni runtime 与状态管理

完整 runtime 不只是多个 encoder 相加，还需要一个统一 session。

### 12.1 Session 状态

`RuntimeSession` 应包含：

- GGUF info 与 weight table。
- tokenizer 与 chat template。
- Qwen3 model/config/weights。
- vision model/config/weights。
- audio model/config/weights。
- speech model/config/weights。
- LLM KV cache。
- multimodal embedding cache。
- audio streaming cache。
- speech output buffer。
- sampler state。
- turn/session 状态。

### 12.2 Prompt 表示

使用统一结构表示多模态输入：

```text
PromptEvent {
    kind: text | image | video_frame | audio_chunk | speech_request | control,
    data_ref: path | bytes | tensor_view | token_span,
    timestamp_ms,
    flags,
}
```

编译成：

```text
CompiledPrompt {
    token_ids,
    embedding_spans,
    position_ids,
    attention_mask_policy,
    media_metadata,
}
```

### 12.3 Streaming 调度

最后阶段需要：

- 输入队列：text/image/audio chunks。
- 预处理队列：vision/audio encoder。
- LLM decode 队列。
- speech synthesis 队列。
- backpressure：输出太慢时暂停 decode 或降采样。
- cancellation：用户打断时清理未完成 speech/audio 状态。

初期 CLI 可以是阻塞式；full-duplex 另设 `stream-chat` 或 `omni-chat`。

## 13. 错误处理策略

所有失败必须可定位：

- 文件打不开：路径和 errno 语义。
- GGUF 截断：header/metadata/tensor directory/tensor data。
- 缺 metadata：具体 key。
- 缺 tensor：具体 tensor name 和 layer。
- shape mismatch：expected/actual。
- dtype unsupported：tensor name + dtype。
- modality unsupported：当前命令、模型分支、缺失模块。
- 内存不足：分配对象、尺寸。
- context 溢出：当前 token/embedding length 与 n_ctx。

CLI 返回码建议：

- `0` 成功。
- `1` 参数错误。
- `2` 文件/内存错误。
- `3` 文件截断或解析失败。
- `4` tokenizer 错误。
- `5` tensor/schema 错误。
- `6` dtype/kernel 未支持。
- `7` modality 未支持。
- `8` runtime forward 失败。
- `9` 明确 unsupported layout。

## 14. 验证策略

### 14.1 单元测试

- binary endian helper。
- GGUF metadata/tensor fixture。
- tokenizer fixture。
- tensor shape/stride/byte size。
- dtype block size。
- F16/BF16 转换。
- RMSNorm、RoPE、Softmax、SwiGLU。
- matvec/quant dot。
- conv/STFT/mel。
- Qwen3 tiny block。
- vision tiny block。
- audio tiny block。
- speech tiny block。

### 14.2 集成测试

- `inspect` 对 tiny GGUF。
- `audit` 对外部 MiniCPM-o GGUF。
- `encode/decode` golden。
- `format-chat` golden。
- text-only generation smoke。
- image raw tensor smoke。
- audio raw/log-mel smoke。
- speech token/vocoder smoke。
- omni manifest smoke。

### 14.3 外部模型 smoke

外部模型路径通过环境变量传入：

```sh
MINICPM_O_GGUF=/path/to/model.gguf make minicpmo-audit
MINICPM_O_TEXT_GGUF=/path/to/text.gguf make qwen3-smoke
MINICPM_O_OMNI_GGUF=/path/to/omni.gguf make omni-smoke
```

这些 target 不应成为默认 `make test`，除非模型 fixture 很小。

## 15. 性能路线

性能优先级：

1. 正确性：小模型/fixture 数值对齐。
2. 可运行：text-only 真实模型能生成。
3. 可交互：图片 + 文本 prompt 可接受延迟。
4. 音频可用：短音频输入可处理。
5. 语音可用：短句 TTS 可输出 wav。
6. 流式：低延迟 full-duplex。

优化方向：

- mmap 权重和 scratch arena。
- fused dequant-dot。
- KV cache contiguous layout。
- prefill batch/tiled matmul。
- vision/audio encoder 的 block-level reuse。
- speech vocoder chunk synthesis。
- benchmark 分开报告 load、prefill、decode、vision encode、audio encode、vocoder。

## 16. 里程碑定义

项目成功不是一次性完成 full omni，而是每个阶段可验证：

- M0：项目骨架和设计文档完整。
- M1：GGUF audit 能准确识别 MiniCPM-o schema。
- M2：Qwen3 text-only 真实 GGUF 可生成文本。
- M3：图片 raw tensor 可影响 logits/generation。
- M4：音频 raw/log-mel 可影响 logits/generation。
- M5：speech token/vocoder 可输出 wav。
- M6：text+image+audio -> text/speech 的阻塞式 omni chat。
- M7：streaming/full-duplex omni chat。

## 17. 初期非目标

- 不做训练或微调。
- 不实现官方 Web UI。
- 不把 PyTorch/Transformers 作为 runtime fallback。
- 不承诺第一版支持视频文件直接输入。
- 不承诺第一版支持实时麦克风/扬声器。
- 不承诺第一版速度达到官方或 llama.cpp-omni。
- 不在仓库保存大权重或媒体样本。

## 18. 参考资料

- OpenBMB/MiniCPM-o: https://github.com/OpenBMB/MiniCPM-o
- MiniCPM-o README 中 MiniCPM-o 4.5 说明和 llama.cpp-omni 端侧路径。
- Qwen3、SigLIP2、Whisper、CosyVoice2 的公开实现和论文只作为行为参考；本项目实现必须保持纯 Uya runtime 边界。
