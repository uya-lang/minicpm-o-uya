# GPU 后端 TODO

## 目标

为 `minicpm-o-uya` 增加一个最小但可演进的 GPU 后端。

当前策略不是一口气把整个运行时 GPU 化，而是按收益和风险排序：

1. 先立住后端 / 设备抽象，默认 CPU 路径不变。
2. 先打通 Qwen3 纯文本 decode 的 CUDA 热路径。
3. 用统一的 `bench` 对比 `--backend cpu` 和 `--backend cuda`。
4. 纯文本路径跑出稳定收益后，再迁 `audio` / TTS / omni 相关路径。

当前仓库已经完成的前置工作：

- `BackendKind` / `Qwen3Device` 已进入 `src/minicpmo/qwen3.uya`。
- `generate` / `chat` / `bench` 已支持 `--backend cpu|cuda`。
- 默认后端仍是 `cpu`。
- `cuda` 当前会明确报 `not implemented yet`。
- `bench` 的真实 text 模式已输出 `backend=...`。

---

## 阶段 0：约束与边界确认

- [ ] 明确第一版 GPU 后端只覆盖 `Qwen3` 纯文本 decode。
- [ ] 明确第一版不覆盖 vision / audio encoder / speech / token2wav / vocoder。
- [ ] 明确第一版允许 `prefill` 和 `decode` 覆盖范围不同，但 `bench` 输出必须说清楚。
- [ ] 明确 GPU 后端的最小支持环境：Linux + NVIDIA CUDA。
- [ ] 明确首版是否要求多卡；若不要求，统一固定 `device.index = 0`。
- [ ] 明确首版错误策略：GPU 不可用、kernel launch 失败、shape unsupported、dtype unsupported 都必须给出清楚报错。

验收标准：

- [ ] 文档明确写出“第一版支持什么、不支持什么”。
- [ ] CLI 遇到 unsupported backend/path 时，不回退成静默 CPU。

---

## 阶段 1：后端模块骨架

- [ ] 新建独立 GPU 后端模块，建议路径如 `src/minicpmo/gpu.uya` 或 `src/minicpmo/cuda.uya`。
- [ ] 定义 GPU 端基础错误码，或统一的布尔返回约定。
- [ ] 定义 GPU context / device handle 的初始化与关闭。
- [ ] 定义 GPU buffer 抽象：device ptr、bytes、owner、dtype。
- [ ] 定义 host <-> device 拷贝 helper。
- [ ] 定义最小日志与 debug 开关。
- [ ] 将 Qwen3 中与后端相关的 helper 尽量收敛到后端模块，而不是散落在模型逻辑里。

验收标准：

- [ ] 可以单独执行一次 `cuda init -> alloc -> memcpy -> free -> close` smoke。
- [ ] 无 GPU 或 CUDA 不可用时，报错路径稳定。

---

## 阶段 2：Qwen3 运行时设备化

- [ ] 扩展 `Qwen3Runtime`，让后端不只停留在枚举层，而是能持有后端特定状态。
- [ ] 为 `Qwen3Runtime` 增加后端特定的 alloc/free 分支。
- [ ] 设计 GPU 版 KV cache 布局，保证与 CPU 版逻辑一致。
- [ ] 设计 GPU 版 RoPE cache 布局，优先复用 CPU 端预计算思路。
- [ ] 设计 `logits` / `hidden` / `scratch` 的 GPU 生命周期。
- [ ] 明确 `qwen3_open_runtime()` 和 `qwen3_close_runtime()` 在 GPU 模式下的资源责任边界。

验收标准：

- [ ] `qwen3_open_runtime()` 在 `backend=cuda` 下至少能完成运行时分配骨架，或在尚未接入 kernel 前给出清楚的 unsupported 点。
- [ ] `qwen3_close_runtime()` 在 GPU 模式下不泄漏资源。

---

## 阶段 3：Matvec 分发

这是第一版最关键的切点。

- [ ] 将 `qwen3_matvec_view()` 改成后端分发。
- [ ] 保留现有 CPU 路径为 `backend=cpu`。
- [ ] 新增 GPU matvec 入口，只覆盖 Qwen3 decode 实际命中的 dtype。
- [ ] 第一批优先覆盖 `F32`。
- [ ] 第一批优先覆盖 `F16`。
- [ ] 第一批优先覆盖 `BF16`。
- [ ] 视实际模型命中情况决定量化优先级：`Q8_0`、`Q4_K`、`Q5_K`、`Q6_K`、`IQ4_NL`。
- [ ] 为每种 GPU matvec 准备与 CPU reference 的误差对照测试。
- [ ] 明确不支持 dtype 时要打印 tensor name 和 dtype。

建议优先级：

1. 先做 dense `F16` / `BF16` / `F32` matvec。
2. 再做模型最常命中的 quant matvec。
3. 最后再考虑更广的 quant 覆盖。

验收标准：

- [ ] `qwen3_matvec_view()` 在 CPU / CUDA 下输出 shape 一致。
- [ ] 小尺寸 golden 与 CPU reference 对齐在阈值内。
- [ ] 出现 unsupported dtype 时不会 silent fallback。

---

## 阶段 4：纯文本 decode 热路径闭环

在 matvec 分发之后，把 decode 真正跑通。

- [ ] 梳理 `qwen3_forward_embedding_with_rope()` 中哪些步骤留在 CPU，哪些迁到 GPU。
- [ ] 第一版优先让 `q/k/v/o projection` GPU 化。
- [ ] 第一版优先让 `gate/up/down projection` GPU 化。
- [ ] 第一版优先让 `output logits matvec` GPU 化。
- [ ] 评估 `rms_norm` 是否需要同时迁到 GPU；若先留在 CPU，评估拷贝开销。
- [ ] 评估 `qwen3_attention()` 是否首版必须 GPU 化；若暂时保留 CPU，需要测拷贝成本是否可接受。
- [ ] 优先保证 single-token decode 的端到端闭环可运行。
- [ ] 明确 `prefill` 是否首版只复用 decode path，还是另做批处理优化。

建议策略：

- 第一阶段可以接受“attention 暂留 CPU、matvec 上 GPU”的过渡版本，但 benchmark 必须诚实。
- 若 CPU <-> GPU 往返吞掉收益，则下一步优先 GPU 化 attention，而不是扩展新 modality。

验收标准：

- [ ] `generate --backend cuda` 在 tiny fixture 上可运行。
- [ ] greedy 输出与 `--backend cpu` 一致，或在允许误差范围内 top-1 一致。
- [ ] 无 context overflow / 资源泄漏 / 崩溃。

---

## 阶段 5：Attention 后端

如果阶段 4 的 matvec-only GPU 方案收益不够，就该做 attention。

- [ ] 为 `qwen3_attention()` 增加后端分发。
- [ ] 优先只做 causal single-token decode attention。
- [ ] 支持 KV cache 留在 GPU 端，不在每个 token 上往返 host/device。
- [ ] 明确 `head_count` / `kv_head_count` / `head_dim` 的支持范围。
- [ ] 增加 small-shape CPU 对照。
- [ ] 增加极限上下文和边界测试。

验收标准：

- [ ] GPU attention 与 CPU reference 在误差阈值内。
- [ ] decode 路径不再需要每个 token 都拷回完整 K/V。
- [ ] `text_decode` bench 比阶段 4 明显改善，或至少能解释瓶颈去向。

---

## 阶段 6：Bench 与性能口径

- [ ] 保持 `bench --backend cpu` 和 `bench --backend cuda` 的输出结构一致。
- [ ] 在 `bench` 输出中明确当前 GPU 优化覆盖面，例如 `hot_matvec=cuda`。
- [ ] 在 `bench` 输出中明确 `attention=cpu` / `attention=cuda`。
- [ ] 在 `bench` 输出中明确 `kv_cache=device` / `kv_cache=host`。
- [ ] 增加 first-token / decode-only / prefill-only 的统计口径说明。
- [ ] 固定一个 tiny fixture `bench`，继续用于功能 smoke。
- [ ] 为真实纯文本模型准备 benchmark 手工命令模板。

建议额外输出：

- [ ] `backend_init_ms`
- [ ] `host_to_device_bytes`
- [ ] `device_to_host_bytes`
- [ ] `first_token_us`
- [ ] `decode_tokens/s`

验收标准：

- [ ] `bench` 输出足以判断收益来自哪里，而不是只看到总 `tokens/s`。
- [ ] CPU / CUDA 可直接用同一命令对比。

---

## 阶段 7：权重与缓存驻留策略

- [ ] 决定哪些权重首版常驻 GPU。
- [ ] 决定是否按层懒上传，还是一次性上传 text decoder 全部权重。
- [ ] 记录显存占用估算公式。
- [ ] 明确运行时 scratch 与 persistent buffer 的边界。
- [ ] 明确超显存时的报错策略。

候选策略：

1. 小模型 / tiny fixture：全量常驻。
2. 大模型首版：先只考虑 text decoder 常驻。
3. 真遇到显存瓶颈，再讨论 layer streaming；不要过早复杂化。

验收标准：

- [ ] 显存使用可估算。
- [ ] 不出现重复上传导致的明显性能劣化。

---

## 阶段 8：测试补齐

- [ ] 为后端解析增加 CLI 测试。
- [ ] 为 `--backend cuda` unsupported 路径增加 fixture 检查。
- [ ] 为 GPU matvec 增加 kernel 级 smoke。
- [ ] 为 tiny Qwen3 生成增加 CPU / CUDA 一致性测试。
- [ ] 为 `bench` 输出增加 backend 字段断言。
- [ ] 若引入 quant GPU kernel，为每个 dtype 增加对照误差测试。

验收标准：

- [ ] 无 GPU 环境下，CPU 测试依然全绿。
- [ ] 有 GPU 环境时，能额外打开一组 CUDA smoke / compare 测试。

---

## 阶段 9：文档与开发者体验

- [ ] 在 `README.md` 增加后端说明。
- [ ] 在 `tests/README.md` 增加 `--backend cpu|cuda` 示例。
- [ ] 记录 CUDA 依赖、驱动要求和构建方式。
- [ ] 记录当前支持的 dtype / 模型范围 / 已知限制。
- [ ] 记录性能测量建议命令。

验收标准：

- [ ] 新协作者能看文档跑通 CPU，对 CUDA 支持范围也不会误解。

---

## 阶段 10：扩到多模态前的门槛

只有在 Qwen3 纯文本 CUDA 路径满足下面条件后，才建议扩 vision / audio / speech：

- [ ] `generate --backend cuda` 在 tiny fixture 上稳定。
- [ ] 真实纯文本模型的 decode 性能达到“值得继续”的水平。
- [ ] `bench` 足以解释瓶颈位置。
- [ ] 运行时资源管理稳定。
- [ ] GPU kernel 的错误路径和 unsupported 路径清晰。

达到门槛后，再按顺序考虑：

1. `audio2audio` 路径里复用的 Qwen3 LLM 部分。
2. `audio encoder`。
3. `TTS decoder`。
4. `token2wav` / `vocoder`。

---

## 非目标

以下内容不属于第一版 GPU 后端：

- 一次性 GPU 化整个 MiniCPM-o 全链路。
- 一开始就支持所有 quant dtype。
- 一开始就支持 full-duplex omni streaming。
- 为了 GPU 后端重写整个 tensor runtime 抽象。
- 静默回退到 CPU，却伪装成 GPU 成功。

---

## 推荐推进顺序

最稳的实际开发顺序：

1. 后端模块骨架。
2. 运行时设备化。
3. matvec 分发。
4. 纯文本 decode 闭环。
5. `bench` 的 CPU / CUDA 对比。
6. attention GPU 化。
7. 权重驻留 / quant 扩展。
8. 多模态复用路径。
