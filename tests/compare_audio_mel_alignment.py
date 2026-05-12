#!/usr/bin/env python3
import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


UYA_MEL_DUMP_MAGIC = 0x4C4D5955
UYA_MEL_DUMP_VERSION = 1
FNV32_INIT = 2166136261
FNV32_PRIME = 16777619


def run_cmd(cmd, timeout_s):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout


def load_uya_dump(path):
    data = Path(path).read_bytes()
    if len(data) < 16:
        raise ValueError(f"Uya mel dump too short: {len(data)} bytes")
    magic, version, frames, mel_bins = struct.unpack("<IIII", data[:16])
    if magic != UYA_MEL_DUMP_MAGIC:
        raise ValueError(f"bad Uya mel dump magic: 0x{magic:08x}")
    if version != UYA_MEL_DUMP_VERSION:
        raise ValueError(f"unsupported Uya mel dump version: {version}")
    count = frames * mel_bins
    expected_size = 16 + count * 4
    if len(data) != expected_size:
        raise ValueError(f"bad Uya mel dump size: got={len(data)} expected={expected_size}")
    if count == 0:
        return frames, mel_bins, []
    values = list(struct.unpack(f"<{count}f", data[16:]))
    return frames, mel_bins, values


def load_llama_json(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list):
        raise ValueError("llama mel dump JSON must be a flat list")
    floats = [float(value) for value in values]
    if not floats:
        raise ValueError("llama mel dump JSON is empty")
    if len(floats) % 80 != 0:
        raise ValueError(f"llama mel dump element count {len(floats)} is not divisible by 80")
    return len(floats) // 80, 80, floats


def float32_checksum(values):
    checksum = FNV32_INIT
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        checksum = ((checksum ^ bits) * FNV32_PRIME) & 0xFFFFFFFF
    return checksum


def compare_values(uya_values, llama_values, frames):
    if len(uya_values) != len(llama_values):
        raise ValueError(f"element count mismatch: uya={len(uya_values)} llama={len(llama_values)}")
    max_abs = 0.0
    max_index = 0
    l1 = 0.0
    for idx, (uya_value, llama_value) in enumerate(zip(uya_values, llama_values)):
        diff = abs(uya_value - llama_value)
        l1 += diff
        if diff > max_abs:
            max_abs = diff
            max_index = idx
    mean_abs = l1 / len(uya_values)
    mel_index = max_index // frames
    frame_index = max_index % frames
    return {
        "l1": l1,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "max_index": max_index,
        "max_mel": mel_index,
        "max_frame": frame_index,
        "uya_at_max": uya_values[max_index],
        "llama_at_max": llama_values[max_index],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Uya real-audio mel dump with llama.cpp-omni log_mel_spectrogram.json.")
    parser.add_argument("--uya", required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--llama-json", required=True)
    parser.add_argument("--dump-path")
    parser.add_argument("--keep-dump", action="store_true")
    parser.add_argument("--max-abs-threshold", type=float, default=2e-3)
    parser.add_argument("--mean-abs-threshold", type=float, default=2e-5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    dump_path = args.dump_path
    cleanup_dump = False
    if dump_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="minicpm-o-uya-mel-", suffix=".uyml", delete=False)
        dump_path = tmp.name
        tmp.close()
        cleanup_dump = not args.keep_dump

    uya_cmd = [
        args.uya,
        "audio-real-mel-probe",
        args.audio_model,
        args.audio,
        "--dump-f32",
        dump_path,
    ]

    print(
        "audio-mel-align config: "
        f"audio_model={args.audio_model} audio={args.audio} "
        f"llama_json={args.llama_json} dump_path={dump_path}"
    )
    rc_uya, out_uya = run_cmd(uya_cmd, args.timeout)
    if rc_uya != 0:
        print(out_uya)
        print(f"audio-mel-align: FAIL uya_rc={rc_uya}")
        return 1

    try:
        uya_frames, uya_mel_bins, uya_values = load_uya_dump(dump_path)
        llama_frames, llama_mel_bins, llama_values = load_llama_json(args.llama_json)
    except ValueError as exc:
        print(f"audio-mel-align: FAIL parse={exc}")
        print("--- Uya output ---")
        print(out_uya)
        return 1
    finally:
        if cleanup_dump:
            Path(dump_path).unlink(missing_ok=True)

    print(
        "audio-mel-align uya: "
        f"frames={uya_frames} mel_bins={uya_mel_bins} "
        f"elements={len(uya_values)} checksum=0x{float32_checksum(uya_values):08x}"
    )
    print(
        "audio-mel-align llama: "
        f"frames={llama_frames} mel_bins={llama_mel_bins} "
        f"elements={len(llama_values)} checksum=0x{float32_checksum(llama_values):08x}"
    )

    if uya_frames != llama_frames or uya_mel_bins != llama_mel_bins:
        print(
            "audio-mel-align: FAIL shape mismatch "
            f"uya={uya_frames}x{uya_mel_bins} llama={llama_frames}x{llama_mel_bins}"
        )
        return 1

    stats = compare_values(uya_values, llama_values, uya_frames)
    print(
        "audio-mel-align diff: "
        f"l1={stats['l1']:.9f} mean_abs={stats['mean_abs']:.9f} max_abs={stats['max_abs']:.9f} "
        f"max_index={stats['max_index']} mel={stats['max_mel']} frame={stats['max_frame']} "
        f"uya={stats['uya_at_max']:.9f} llama={stats['llama_at_max']:.9f}"
    )

    if stats["max_abs"] > args.max_abs_threshold:
        print(
            "audio-mel-align: FAIL "
            f"max_abs={stats['max_abs']:.9f} exceeds threshold={args.max_abs_threshold:.9f}"
        )
        return 1
    if stats["mean_abs"] > args.mean_abs_threshold:
        print(
            "audio-mel-align: FAIL "
            f"mean_abs={stats['mean_abs']:.9f} exceeds threshold={args.mean_abs_threshold:.9f}"
        )
        return 1

    print("audio-mel-align: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
