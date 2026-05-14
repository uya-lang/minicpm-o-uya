#!/usr/bin/env python3
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


UYA_MEL_DUMP_MAGIC = 0x4C4D5955
UYA_MEL_DUMP_VERSION = 1


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
        raise ValueError(f"short uya dump: {path}")
    magic, version, frames, mel_bins = struct.unpack("<IIII", data[:16])
    if magic != UYA_MEL_DUMP_MAGIC:
        raise ValueError(f"bad uya dump magic=0x{magic:08x}")
    if version != UYA_MEL_DUMP_VERSION:
        raise ValueError(f"bad uya dump version={version}")
    count = frames * mel_bins
    need = 16 + count * 4
    if len(data) != need:
        raise ValueError(f"bad uya dump size got={len(data)} expected={need}")
    values = []
    if count:
        values = list(struct.unpack(f"<{count}f", data[16:]))
    return frames, mel_bins, values


def compare_values(a_values, b_values, frames):
    if len(a_values) != len(b_values):
        raise ValueError(f"element mismatch a={len(a_values)} b={len(b_values)}")
    max_abs = 0.0
    max_index = 0
    l1 = 0.0
    for idx, (a_value, b_value) in enumerate(zip(a_values, b_values)):
        diff = abs(a_value - b_value)
        l1 += diff
        if diff > max_abs:
            max_abs = diff
            max_index = idx
    mean_abs = l1 / len(a_values) if a_values else 0.0
    mel_index = max_index // frames if frames else 0
    frame_index = max_index % frames if frames else 0
    return {
        "l1": l1,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "max_index": max_index,
        "mel_index": mel_index,
        "frame_index": frame_index,
        "a_value": a_values[max_index] if a_values else 0.0,
        "b_value": b_values[max_index] if b_values else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare real audio mel one-shot output with streamed chunk output from Uya."
    )
    parser.add_argument("--uya", required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--chunk-samples", type=int, default=1600)
    parser.add_argument("--max-abs-threshold", type=float, default=2e-6)
    parser.add_argument("--mean-abs-threshold", type=float, default=2e-7)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="minicpm-o-uya-real-stream-") as tmp_dir:
        oneshot_path = str(Path(tmp_dir) / "oneshot.uyml")
        stream_path = str(Path(tmp_dir) / "stream.uyml")

        oneshot_cmd = [
            args.uya,
            "audio-real-mel-probe",
            args.audio_model,
            args.audio,
            "--dump-f32",
            oneshot_path,
        ]
        stream_cmd = [
            args.uya,
            "audio-real-mel-probe",
            args.audio_model,
            args.audio,
            "--stream-chunk-samples",
            str(args.chunk_samples),
            "--dump-f32",
            stream_path,
        ]

        print(
            "audio-real-stream-align config: "
            f"audio={args.audio} chunk_samples={args.chunk_samples} "
            f"audio_model={args.audio_model}"
        )

        rc_oneshot, out_oneshot = run_cmd(oneshot_cmd, args.timeout)
        print(out_oneshot, end="")
        if rc_oneshot != 0:
            print(f"audio-real-stream-align: FAIL oneshot_rc={rc_oneshot}")
            return 1

        rc_stream, out_stream = run_cmd(stream_cmd, args.timeout)
        print(out_stream, end="")
        if rc_stream != 0:
            print(f"audio-real-stream-align: FAIL stream_rc={rc_stream}")
            return 1

        try:
            one_frames, one_bins, one_values = load_uya_dump(oneshot_path)
            stream_frames, stream_bins, stream_values = load_uya_dump(stream_path)
        except ValueError as exc:
            print(f"audio-real-stream-align: FAIL parse={exc}")
            return 1

        print(
            "audio-real-stream-align shape: "
            f"oneshot={one_frames}x{one_bins} streamed={stream_frames}x{stream_bins}"
        )
        if (one_frames, one_bins) != (stream_frames, stream_bins):
            print("audio-real-stream-align: FAIL shape mismatch")
            return 1

        stats = compare_values(one_values, stream_values, one_frames)
        print(
            "audio-real-stream-align diff: "
            f"l1={stats['l1']:.9f} mean_abs={stats['mean_abs']:.9f} max_abs={stats['max_abs']:.9f} "
            f"max_index={stats['max_index']} mel={stats['mel_index']} frame={stats['frame_index']} "
            f"oneshot={stats['a_value']:.9f} streamed={stats['b_value']:.9f}"
        )

        if stats["max_abs"] > args.max_abs_threshold:
            print(
                "audio-real-stream-align: FAIL "
                f"max_abs={stats['max_abs']:.9f} exceeds threshold={args.max_abs_threshold:.9f}"
            )
            return 1
        if stats["mean_abs"] > args.mean_abs_threshold:
            print(
                "audio-real-stream-align: FAIL "
                f"mean_abs={stats['mean_abs']:.9f} exceeds threshold={args.mean_abs_threshold:.9f}"
            )
            return 1

        print("audio-real-stream-align: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
