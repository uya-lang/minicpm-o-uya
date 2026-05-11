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


def f16_bits(value: float) -> int:
    if value == 0.0:
        return 0
    sign = 0
    if value < 0:
        sign = 1
        value = -value
    exp = 0
    while value >= 2.0:
        value *= 0.5
        exp += 1
    while value < 1.0:
        value *= 2.0
        exp -= 1
    exp_bits = exp + 15
    frac = int((value - 1.0) * 1024.0 + 0.5)
    if frac >= 1024:
        frac = 0
        exp_bits += 1
    return (sign << 15) | (exp_bits << 10) | frac


def pack_tensor_values(name: str, dims: list[int], ggml_type: int) -> bytes:
    elements = 1
    for dim in dims:
        elements *= dim
    values = []
    if name == "token_embd.weight":
        for token in range(dims[1]):
            for row in range(dims[0]):
                target_row = token % dims[0]
                if token == 11:
                    target_row = 4
                values.append(1.0 if row == target_row else 0.0)
    elif name == "output_norm.weight":
        values = [1.0] * elements
    elif name == "output.weight":
        for vocab in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if row == (vocab % dims[0]) else 0.0)
    elif name.endswith("attn_norm.weight") or name.endswith("ffn_norm.weight"):
        values = [1.0] * elements
    elif name.endswith("attn_q.weight"):
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if row == col else 0.0)
    elif name.endswith("attn_k.weight") or name.endswith("attn_v.weight"):
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if row == col else 0.0)
    elif name.endswith("attn_output.weight"):
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if row == col else 0.0)
    elif name.endswith("attn_q_norm.weight") or name.endswith("attn_k_norm.weight"):
        values = [1.0] * elements
    elif name.endswith("ffn_gate.weight"):
        values = [0.0] * elements
    elif name.endswith("ffn_up.weight"):
        values = [0.0] * elements
    elif name.endswith("ffn_down.weight"):
        values = [0.0] * elements
    elif name == "vision.patch_embd.weight":
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(0.5 if (row % dims[1]) == col else 0.0)
    elif name == "vision.patch_embd.bias":
        values = [0.0] * elements
    elif name == "vision.position_embd.weight":
        for pos in range(dims[0]):
            for col in range(dims[1]):
                values.append(0.05 * (pos + 1) * (col + 1))
    elif name == "vision.output_norm.weight":
        values = [1.0] * elements
    elif name == "vision.resampler.weight":
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if row == col else 0.0)
    elif name == "vision.projector.weight":
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if (col % dims[0]) == row else 0.0)
    elif name.startswith("vision.blk.") and (name.endswith("attn_norm.weight") or name.endswith("ffn_norm.weight")):
        values = [1.0] * elements
    elif name.startswith("vision.blk.") and (name.endswith("attn_q.weight") or name.endswith("attn_k.weight") or name.endswith("attn_v.weight") or name.endswith("attn_output.weight")):
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if row == col else 0.0)
    elif name.startswith("vision.blk.") and (name.endswith("ffn_gate.weight") or name.endswith("ffn_up.weight") or name.endswith("ffn_down.weight")):
        values = [0.0] * elements
    elif name == "audio.conv1.weight":
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(0.75 if row == col else 0.0)
    elif name == "audio.conv1.bias":
        values = [0.0] * elements
    elif name == "audio.position_embd.weight":
        for pos in range(dims[0]):
            for col in range(dims[1]):
                values.append(0.03 * (pos + 1) * (col + 1))
    elif name == "audio.output_norm.weight":
        values = [1.0] * elements
    elif name == "audio.projector.weight":
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if (col % dims[0]) == row else 0.0)
    elif name.startswith("audio.blk.") and (name.endswith("attn_norm.weight") or name.endswith("ffn_norm.weight")):
        values = [1.0] * elements
    elif name.startswith("audio.blk.") and (name.endswith("attn_q.weight") or name.endswith("attn_k.weight") or name.endswith("attn_v.weight") or name.endswith("attn_output.weight")):
        for col in range(dims[1]):
            for row in range(dims[0]):
                values.append(1.0 if row == col else 0.0)
    elif name.startswith("audio.blk.") and (name.endswith("ffn_gate.weight") or name.endswith("ffn_up.weight") or name.endswith("ffn_down.weight")):
        values = [0.0] * elements
    else:
        values = [0.0] * elements
    if ggml_type == GGML_TYPE_F32:
        return b"".join(struct.pack("<f", value) for value in values)
    if ggml_type == GGML_TYPE_F16:
        return b"".join(struct.pack("<H", f16_bits(value)) for value in values)
    raise ValueError(f"unsupported ggml type {ggml_type}")


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
        metadata_u32("vision.image_width", 2),
        metadata_u32("vision.image_height", 2),
        metadata_u32("vision.channel_count", 3),
        metadata_u32("vision.patch_size", 2),
        metadata_u32("vision.preprocess.target_width", 2),
        metadata_u32("vision.preprocess.target_height", 2),
        metadata_u32("vision.preprocess.crop_width", 2),
        metadata_u32("vision.preprocess.crop_height", 2),
        metadata_u32("vision.preprocess.tile_width", 2),
        metadata_u32("vision.preprocess.tile_height", 2),
        metadata_f32("vision.preprocess.mean_r", 0.5),
        metadata_f32("vision.preprocess.mean_g", 0.5),
        metadata_f32("vision.preprocess.mean_b", 0.5),
        metadata_f32("vision.preprocess.std_r", 0.5),
        metadata_f32("vision.preprocess.std_g", 0.5),
        metadata_f32("vision.preprocess.std_b", 0.5),
        metadata_u32("vision.embedding_length", 4),
        metadata_u32("vision.siglip2.block_count", 1),
        metadata_u32("vision.attention.head_count", 1),
        metadata_u32("vision.feed_forward_length", 8),
        metadata_u32("vision.projector.output_length", 8),
        metadata_u32("audio.log_mel.frame_count", 1),
        metadata_u32("audio.log_mel.bin_count", 4),
        metadata_u32("audio.embedding_length", 4),
        metadata_u32("audio.whisper.block_count", 1),
        metadata_u32("audio.attention.head_count", 1),
        metadata_u32("audio.feed_forward_length", 8),
        metadata_u32("audio.projector.output_length", 8),
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
        metadata_u32("minicpmo.media.audio_token_id", 13),
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
        ("vision.patch_embd.weight", [12, 4], GGML_TYPE_F32),
        ("vision.patch_embd.bias", [4], GGML_TYPE_F32),
        ("vision.position_embd.weight", [4, 4], GGML_TYPE_F32),
        ("vision.blk.0.attn_norm.weight", [4], GGML_TYPE_F32),
        ("vision.blk.0.attn_q.weight", [4, 4], GGML_TYPE_F32),
        ("vision.blk.0.attn_k.weight", [4, 4], GGML_TYPE_F32),
        ("vision.blk.0.attn_v.weight", [4, 4], GGML_TYPE_F32),
        ("vision.blk.0.attn_output.weight", [4, 4], GGML_TYPE_F32),
        ("vision.blk.0.ffn_norm.weight", [4], GGML_TYPE_F32),
        ("vision.blk.0.ffn_gate.weight", [4, 8], GGML_TYPE_F32),
        ("vision.blk.0.ffn_up.weight", [4, 8], GGML_TYPE_F32),
        ("vision.blk.0.ffn_down.weight", [8, 4], GGML_TYPE_F32),
        ("vision.output_norm.weight", [4], GGML_TYPE_F32),
        ("vision.resampler.weight", [4, 4], GGML_TYPE_F32),
        ("vision.projector.weight", [4, 8], GGML_TYPE_F32),
        ("audio.conv1.weight", [4, 4], GGML_TYPE_F32),
        ("audio.conv1.bias", [4], GGML_TYPE_F32),
        ("audio.position_embd.weight", [1, 4], GGML_TYPE_F32),
        ("audio.blk.0.attn_norm.weight", [4], GGML_TYPE_F32),
        ("audio.blk.0.attn_q.weight", [4, 4], GGML_TYPE_F32),
        ("audio.blk.0.attn_k.weight", [4, 4], GGML_TYPE_F32),
        ("audio.blk.0.attn_v.weight", [4, 4], GGML_TYPE_F32),
        ("audio.blk.0.attn_output.weight", [4, 4], GGML_TYPE_F32),
        ("audio.blk.0.ffn_norm.weight", [4], GGML_TYPE_F32),
        ("audio.blk.0.ffn_gate.weight", [4, 8], GGML_TYPE_F32),
        ("audio.blk.0.ffn_up.weight", [4, 8], GGML_TYPE_F32),
        ("audio.blk.0.ffn_down.weight", [8, 4], GGML_TYPE_F32),
        ("audio.output_norm.weight", [4], GGML_TYPE_F32),
        ("audio.projector.weight", [4, 8], GGML_TYPE_F32),
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
        tensor_data.extend(pack_tensor_values(name, dims, ggml_type))
        offset += size

    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    directory = header + b"".join(metadata) + b"".join(tensors)
    data_start = align_up(len(directory), 32)
    padding = b"\x00" * (data_start - len(directory))
    fixture = directory + padding + bytes(tensor_data)

    (out_dir / "tiny.gguf").write_bytes(fixture)
    (out_dir / "tiny.gguf.part").write_bytes(fixture[:48])
    (out_dir / "tiny_data_truncated.gguf").write_bytes(fixture[:-17])

    bpe_metadata = [
        metadata_string("general.architecture", "tokenizer-only"),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string_array("tokenizer.ggml.tokens", [
            "<unk>", "h", "e", "he", "l", "hel", "lo", "hello",
            "Ġ", "w", "o", "wo", "r", "wor", "ld", "world", "Ġworld",
            "ä", "½", "ł", "ä½ł",
        ]),
        metadata_string_array("tokenizer.ggml.merges", [
            "h e", "he l", "l o", "hel lo",
            "w o", "wo r", "l d", "wor ld", "Ġ world",
            "ä ½", "ä½ ł",
        ]),
        metadata_u32("tokenizer.ggml.unknown_token_id", 0),
    ]
    bpe_header = struct.pack("<IIQQ", 0x46554747, 3, 0, len(bpe_metadata))
    (out_dir / "tiny_bpe.gguf").write_bytes(bpe_header + b"".join(bpe_metadata))

    rgb_pixels = bytes([
        0, 64, 128, 32, 96, 160, 64, 128, 192,
        96, 160, 224, 128, 192, 255, 160, 224, 32,
    ])
    rgb_frame_b = bytes((value + 17) % 256 for value in rgb_pixels)
    tiny_rgb = struct.pack("<IIIIII", 0x47525955, 1, 3, 2, 3, 0) + rgb_pixels
    tiny_rgb_b = struct.pack("<IIIIII", 0x47525955, 1, 3, 2, 3, 0) + rgb_frame_b
    (out_dir / "tiny_rgb.raw").write_bytes(tiny_rgb)
    tiny_image_values = [
        -1.0, -0.7490196078431373, -0.24705882352941178, 0.0039215686274509665,
        -0.4980392156862745, -0.24705882352941178, 0.2549019607843137, 0.5058823529411764,
        0.0039215686274509665, 0.2549019607843137, 0.7568627450980392, 1.0,
    ]
    tiny_image = struct.pack("<IIIIII", 0x49415955, 1, 2, 2, 3, 0)
    tiny_image += b"".join(struct.pack("<f", value) for value in tiny_image_values)
    (out_dir / "tiny_image.raw").write_bytes(tiny_image)
    tiny_image_manifest = struct.pack("<IIIIIII", 0x4D495955, 1, 1, 2, 2, 2, 2) + tiny_rgb
    (out_dir / "tiny_image.uyim").write_bytes(tiny_image_manifest)
    tiny_video_manifest = struct.pack("<IIIIIII", 0x4D565955, 1, 2, 2, 2, 2, 2) + tiny_rgb + tiny_rgb_b
    (out_dir / "tiny_video.uyvm").write_bytes(tiny_video_manifest)
    tiny_audio = struct.pack("<IIIII", 0x4D415955, 1, 1, 4, 0)
    tiny_audio += b"".join(struct.pack("<f", value) for value in [0.2, -0.4, 0.6, -0.8])
    (out_dir / "tiny_audio.raw").write_bytes(tiny_audio)
    stereo_pcm = [
        (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8),
        (0.8, 0.6), (0.6, 0.4), (0.4, 0.2), (0.2, 0.0),
        (-0.2, -0.4), (-0.4, -0.6), (-0.6, -0.8), (-0.8, -0.6),
        (-0.6, -0.4), (-0.4, -0.2), (-0.2, 0.0), (0.0, 0.2),
    ]
    tiny_pcm = struct.pack("<IIIIII", 0x50415955, 1, 8000, 2, len(stereo_pcm), 1)
    for left, right in stereo_pcm:
        tiny_pcm += struct.pack("<ff", left, right)
    (out_dir / "tiny_audio.pcm").write_bytes(tiny_pcm)
    tiny_omni = """{
  "events": [
    {"kind": "text", "text": "hello "},
    {"kind": "image", "path": "tiny_rgb.raw"},
    {"kind": "text", "text": "world"},
    {"kind": "video_frame", "path": "tiny_rgb.raw"},
    {"kind": "audio_chunk", "path": "tiny_audio.raw"},
    {"kind": "speech_request", "voice": "tiny"},
    {"kind": "control", "name": "end_turn"}
  ]
}
"""
    (out_dir / "tiny_omni.json").write_text(tiny_omni)
    tiny_stream = """{
  "events": [
    {"kind": "text", "text": "hello"},
    {"kind": "audio_chunk", "path": "tiny_audio_000.raw"},
    {"kind": "audio_chunk", "path": "tiny_audio_001.raw"},
    {"kind": "speech_request", "voice": "tiny"},
    {"kind": "control", "name": "interrupt"},
    {"kind": "text", "text": "world"}
  ]
}
"""
    (out_dir / "tiny_stream.json").write_text(tiny_stream)

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
        missing_q_data.extend(pack_tensor_values(name, dims, ggml_type))
        offset += size
    missing_q_directory = struct.pack("<IIQQ", 0x46554747, 3, len(missing_q_tensors), len(metadata)) + b"".join(metadata) + b"".join(missing_q_tensors)
    missing_q_data_start = align_up(len(missing_q_directory), 32)
    missing_q_fixture = missing_q_directory + (b"\x00" * (missing_q_data_start - len(missing_q_directory))) + bytes(missing_q_data)
    (out_dir / "tiny_qwen3_missing_q.gguf").write_bytes(missing_q_fixture)

    missing_audio_specs = [spec for spec in tensor_specs if not spec[0].startswith("audio.")]
    missing_audio_tensors = []
    missing_audio_data = bytearray()
    offset = 0
    for index, (name, dims, ggml_type) in enumerate(missing_audio_specs):
        offset = align_up(offset, 32)
        if len(missing_audio_data) < offset:
            missing_audio_data.extend(b"\x00" * (offset - len(missing_audio_data)))
        size = tensor_size(dims, ggml_type)
        missing_audio_tensors.append(tensor_info(name, dims, ggml_type, offset))
        missing_audio_data.extend(pack_tensor_values(name, dims, ggml_type))
        offset += size
    missing_audio_directory = struct.pack("<IIQQ", 0x46554747, 3, len(missing_audio_tensors), len(metadata)) + b"".join(metadata) + b"".join(missing_audio_tensors)
    missing_audio_data_start = align_up(len(missing_audio_directory), 32)
    missing_audio_fixture = missing_audio_directory + (b"\x00" * (missing_audio_data_start - len(missing_audio_directory))) + bytes(missing_audio_data)
    (out_dir / "tiny_audio_missing.gguf").write_bytes(missing_audio_fixture)


if __name__ == "__main__":
    main()
