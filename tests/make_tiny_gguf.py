#!/usr/bin/env python3
import struct
from pathlib import Path


GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_STRING = 8
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


def metadata_u32_array(key: str, values: list[int]) -> bytes:
    payload = gguf_string(key)
    payload += struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_UINT32, len(values))
    for value in values:
        payload += struct.pack("<I", value)
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
        metadata_u32("qwen3.block_count", 1),
        metadata_u32("qwen3.attention.head_count", 2),
        metadata_u32("vision.siglip2.block_count", 1),
        metadata_u32("audio.whisper.block_count", 1),
        metadata_string("speech.cosyvoice2.kind", "codec-vocoder"),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array("tokenizer.ggml.tokens", ["<unk>", "hello", "<image>", "<audio>", "<video>"]),
        metadata_string("tokenizer.chat_template", "<|im_start|>user <image> <audio><|im_end|>"),
        metadata_u32("minicpmo.media.image_token_id", 2),
        metadata_i32("test.scalar_i32", -7),
        metadata_u32_array("test.array_u32", [1, 2, 3, 5, 8]),
    ]

    tensor_specs = [
        ("token_embd.weight", [8, 4], GGML_TYPE_F16),
        ("output.weight", [4, 8], GGML_TYPE_F16),
        ("blk.0.attn_q.weight", [4, 2], GGML_TYPE_F32),
        ("vision.patch_embd.weight", [4, 4], GGML_TYPE_F16),
        ("audio.whisper.encoder.weight", [4, 4], GGML_TYPE_F16),
        ("speech.cosyvoice2.decoder.weight", [4, 4], GGML_TYPE_F16),
        ("vocoder.proj.weight", [4, 4], GGML_TYPE_F16),
    ]

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

    bad_tensor = tensor_info("mystery.branch.weight", [4], 63, 0)
    bad_directory = struct.pack("<IIQQ", 0x46554747, 3, 1, 0) + bad_tensor
    bad_data_start = align_up(len(bad_directory), 32)
    bad_fixture = bad_directory + (b"\x00" * (bad_data_start - len(bad_directory))) + b"bad!"
    (out_dir / "bad_schema.gguf").write_bytes(bad_fixture)


if __name__ == "__main__":
    main()
