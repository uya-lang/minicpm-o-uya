#!/usr/bin/env python3
import struct
from pathlib import Path


GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_STRING = 8
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_ARRAY = 9
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def metadata_string(key: str, value: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(value)


def metadata_u32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<II", GGUF_TYPE_UINT32, value)


def metadata_i32(key: str, value: int) -> bytes:
    return gguf_string(key) + struct.pack("<Ii", GGUF_TYPE_INT32, value)


def metadata_f32(key: str, value: float) -> bytes:
    return gguf_string(key) + struct.pack("<If", GGUF_TYPE_FLOAT32, value)


def metadata_u32_array(key: str, values: list[int]) -> bytes:
    payload = gguf_string(key)
    payload += struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_UINT32, len(values))
    for value in values:
        payload += struct.pack("<I", value)
    return payload


def metadata_i32_array(key: str, values: list[int]) -> bytes:
    payload = gguf_string(key)
    payload += struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_INT32, len(values))
    for value in values:
        payload += struct.pack("<i", value)
    return payload


def metadata_f32_array(key: str, values: list[float]) -> bytes:
    payload = gguf_string(key)
    payload += struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32, len(values))
    for value in values:
        payload += struct.pack("<f", value)
    return payload


def metadata_string_array(key: str, values: list[str]) -> bytes:
    payload = gguf_string(key)
    payload += struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING, len(values))
    for value in values:
        payload += gguf_string(value)
    return payload


def tensor_info(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    payload = gguf_string(name)
    payload += struct.pack("<I", len(dims))
    for dim in dims:
        payload += struct.pack("<Q", dim)
    payload += struct.pack("<IQ", ggml_type, offset)
    return payload


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_size(dims: list[int], ggml_type: int) -> int:
    elements = 1
    for dim in dims:
        elements *= dim
    if ggml_type == GGML_TYPE_F32:
        return elements * 4
    if ggml_type == GGML_TYPE_F16:
        return elements * 2
    raise ValueError(f"unsupported fixture ggml type {ggml_type}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = [
        metadata_string("general.architecture", "minicpmo-qwen3"),
        metadata_string("general.name", "tiny phase2 audit fixture"),
        metadata_u32("general.alignment", 32),
        metadata_u32("qwen3.context_length", 32),
        metadata_u32("qwen3.embedding_length", 8),
        metadata_u32("qwen3.feed_forward_length", 16),
        metadata_u32("qwen3.block_count", 1),
        metadata_u32("qwen3.attention.head_count", 2),
        metadata_u32("qwen3.attention.head_count_kv", 1),
        metadata_u32("qwen3.rope.dimension_count", 4),
        metadata_f32("qwen3.attention.layer_norm_rms_epsilon", 0.000001),
        metadata_f32("qwen3.rope.freq_base", 1000000.0),
        metadata_u32("vision.siglip2.block_count", 1),
        metadata_u32("audio.whisper.block_count", 1),
        metadata_string("speech.cosyvoice2.kind", "codec-vocoder"),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array("tokenizer.ggml.tokens", [
            "<unk>", "<s>", "</s>", "<pad>", "hello", " ", "world", "!",
            "你好", "，", "世界", "\n", "<image>", "<audio>", "<video>", "<|im_start|>", "<|im_end|>", "user", "assistant",
        ]),
        metadata_f32_array("tokenizer.ggml.scores", [0.0, 0.0, 0.0, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9, -1.0]),
        metadata_i32_array("tokenizer.ggml.token_type", [3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1]),
        metadata_string_array("tokenizer.ggml.merges", ["h e", "he l", "hel lo", "w o", "wo r", "wor ld"]),
        metadata_u32("tokenizer.ggml.unknown_token_id", 0),
        metadata_u32("tokenizer.ggml.bos_token_id", 1),
        metadata_u32("tokenizer.ggml.eos_token_id", 2),
        metadata_u32("tokenizer.ggml.padding_token_id", 3),
        metadata_string("tokenizer.chat_template", "minicpmo.chatml"),
        metadata_u32("minicpmo.media.image_token_id", 12),
        metadata_i32("test.scalar_i32", -7),
        metadata_u32_array("test.array_u32", [1, 2, 3, 5, 8]),
    ]

    text_tensor_specs = [
        ("token_embd.weight", [8, 19], GGML_TYPE_F16),
        ("output_norm.weight", [8], GGML_TYPE_F32),
        ("output.weight", [8, 19], GGML_TYPE_F16),
        ("blk.0.attn_norm.weight", [8], GGML_TYPE_F32),
        ("blk.0.attn_q.weight", [8, 8], GGML_TYPE_F32),
        ("blk.0.attn_k.weight", [8, 4], GGML_TYPE_F32),
        ("blk.0.attn_v.weight", [8, 4], GGML_TYPE_F32),
        ("blk.0.attn_output.weight", [8, 8], GGML_TYPE_F32),
        ("blk.0.attn_q_norm.weight", [4], GGML_TYPE_F32),
        ("blk.0.attn_k_norm.weight", [4], GGML_TYPE_F32),
        ("blk.0.ffn_norm.weight", [8], GGML_TYPE_F32),
        ("blk.0.ffn_gate.weight", [8, 16], GGML_TYPE_F16),
        ("blk.0.ffn_up.weight", [8, 16], GGML_TYPE_F16),
        ("blk.0.ffn_down.weight", [16, 8], GGML_TYPE_F16),
    ]
    branch_tensor_specs = [
        ("vision.patch_embd.weight", [4, 4], GGML_TYPE_F16),
        ("audio.whisper.encoder.weight", [4, 4], GGML_TYPE_F16),
        ("speech.cosyvoice2.decoder.weight", [4, 4], GGML_TYPE_F16),
        ("vocoder.proj.weight", [4, 4], GGML_TYPE_F16),
    ]
    tensor_specs = text_tensor_specs + branch_tensor_specs

    tensors = []
    tensor_data = bytearray()
    offset = 0
    for index, (name, dims, ggml_type) in enumerate(tensor_specs):
        offset = align_up(offset, 32)
        if len(tensor_data) < offset:
            tensor_data.extend(b"\x00" * (offset - len(tensor_data)))
        size = tensor_size(dims, ggml_type)
        tensors.append(tensor_info(name, dims, ggml_type, offset))
        tensor_data.extend(((index * 37 + i) & 0xFF) for i in range(size))
        offset += size

    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    directory = header + b"".join(metadata) + b"".join(tensors)
    data_start = align_up(len(directory), 32)
    padding = b"\x00" * (data_start - len(directory))
    fixture = directory + padding + bytes(tensor_data)

    (out_dir / "tiny.gguf").write_bytes(fixture)
    (out_dir / "tiny.gguf.part").write_bytes(fixture[:48])
    (out_dir / "tiny_data_truncated.gguf").write_bytes(fixture[:-17])

    bad_tensor = tensor_info("mystery.branch.weight", [4], 63, 0)
    bad_directory = struct.pack("<IIQQ", 0x46554747, 3, 1, 0) + bad_tensor
    bad_data_start = align_up(len(bad_directory), 32)
    bad_fixture = bad_directory + (b"\x00" * (bad_data_start - len(bad_directory))) + b"bad!"
    (out_dir / "bad_schema.gguf").write_bytes(bad_fixture)

    missing_q_specs = [spec for spec in text_tensor_specs if spec[0] != "blk.0.attn_q.weight"]
    missing_q_tensors = []
    missing_q_data = bytearray()
    offset = 0
    for index, (name, dims, ggml_type) in enumerate(missing_q_specs):
        offset = align_up(offset, 32)
        if len(missing_q_data) < offset:
            missing_q_data.extend(b"\x00" * (offset - len(missing_q_data)))
        size = tensor_size(dims, ggml_type)
        missing_q_tensors.append(tensor_info(name, dims, ggml_type, offset))
        missing_q_data.extend(((index * 19 + i) & 0xFF) for i in range(size))
        offset += size
    missing_q_directory = struct.pack("<IIQQ", 0x46554747, 3, len(missing_q_tensors), len(metadata)) + b"".join(metadata) + b"".join(missing_q_tensors)
    missing_q_data_start = align_up(len(missing_q_directory), 32)
    missing_q_fixture = missing_q_directory + (b"\x00" * (missing_q_data_start - len(missing_q_directory))) + bytes(missing_q_data)
    (out_dir / "tiny_qwen3_missing_q.gguf").write_bytes(missing_q_fixture)


if __name__ == "__main__":
    main()
