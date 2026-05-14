#!/usr/bin/env python3
import argparse
import json
import wave
from pathlib import Path


def wav_info(path: Path):
    with wave.open(str(path), "rb") as handle:
        return {
            "sample_rate": handle.getframerate(),
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "frames": handle.getnframes(),
        }


def read_wav_pcm16(path: Path):
    info = wav_info(path)
    if info["channels"] != 1 or info["sample_width"] != 2:
        raise ValueError(f"unsupported wav format for {path}: {info}")
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    return info, frames


def write_wav_pcm16(path: Path, sample_rate: int, frames: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def build_case(name, category, ref_audio, user_audio, note):
    ref_info = wav_info(ref_audio)
    user_info = wav_info(user_audio) if user_audio else None
    return {
        "name": name,
        "category": category,
        "ref_audio": str(ref_audio),
        "user_audio": str(user_audio) if user_audio else None,
        "ref_duration_ms": ref_info["frames"] * 1000.0 / ref_info["sample_rate"],
        "user_duration_ms": (
            user_info["frames"] * 1000.0 / user_info["sample_rate"] if user_info else None
        ),
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a local audio2audio real-path case matrix with reusable WAV inputs."
    )
    parser.add_argument(
        "--output",
        default="outputs/baselines/audio2audio_case_matrix.json",
        help="Output JSON manifest path",
    )
    parser.add_argument(
        "--derived-dir",
        default="outputs/case_matrix",
        help="Directory for generated short/long/silence WAV files",
    )
    parser.add_argument(
        "--complex-ref",
        default="outputs/complex_case2/complex2_0000.wav",
    )
    parser.add_argument(
        "--complex-user",
        default="outputs/complex_case2/complex2_0001.wav",
    )
    parser.add_argument(
        "--mixed-user",
        default="MiniCPM-o/assets/input_examples/chi-english-1.wav",
    )
    parser.add_argument(
        "--english-user",
        default="MiniCPM-o/assets/input_examples/fast-pace.wav",
    )
    parser.add_argument(
        "--english-alt-user",
        default="MiniCPM-o/assets/input_examples/exciting-emotion.wav",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    derived_dir = Path(args.derived_dir)
    complex_ref = Path(args.complex_ref)
    complex_user = Path(args.complex_user)
    mixed_user = Path(args.mixed_user)
    english_user = Path(args.english_user)
    english_alt_user = Path(args.english_alt_user)

    complex_info, complex_frames = read_wav_pcm16(complex_user)
    sample_rate = complex_info["sample_rate"]
    one_second_bytes = sample_rate * 2
    four_seconds_bytes = sample_rate * 2 * 4

    short_user = derived_dir / "short_trim_0001.wav"
    write_wav_pcm16(short_user, sample_rate, complex_frames[:one_second_bytes])

    long_user = derived_dir / "long_loop_0001.wav"
    long_frames = complex_frames[:four_seconds_bytes] * 2
    write_wav_pcm16(long_user, sample_rate, long_frames)

    silence_user = derived_dir / "silence_0001.wav"
    silence_frames = b"\x00\x00" * (sample_rate * 2)
    write_wav_pcm16(silence_user, sample_rate, silence_frames)

    cases = [
        build_case(
            "complex_multirequirements",
            "complex",
            complex_ref,
            complex_user,
            "Existing two-file baseline; Chinese multi-step performance question.",
        ),
        build_case(
            "short_trimmed",
            "short",
            complex_ref,
            short_user,
            "First 1 second trimmed from complex user audio for short-input regressions.",
        ),
        build_case(
            "long_looped",
            "long",
            complex_ref,
            long_user,
            "Looped 4-second slice from complex user audio to stress longer streaming input.",
        ),
        build_case(
            "silence_guard",
            "silence",
            complex_ref,
            silence_user,
            "Synthetic 2-second 16 kHz mono silence to verify empty/quiet user audio handling.",
        ),
    ]

    optional_inputs = [
        ("english_fastpace", "english", english_user, "Repository example WAV intended for fast English speech."),
        ("english_excited", "english", english_alt_user, "Repository example WAV intended for expressive English speech."),
        ("mixed_zh_en", "mixed", mixed_user, "Repository example WAV intended for Chinese-English mixed speech."),
    ]
    for name, category, user_path, note in optional_inputs:
        if user_path.exists():
            info = wav_info(user_path)
            if info["channels"] == 1 and info["sample_width"] == 2:
                cases.append(build_case(name, category, complex_ref, user_path, note))

    manifest = {
        "schema": "minicpm-o-uya.audio2audio_case_matrix.v1",
        "note": "Local reusable case inventory for real audio2audio regressions.",
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"derived_dir={derived_dir}")
    for case in cases:
        print(
            "case-matrix "
            f"name={case['name']} category={case['category']} "
            f"user_ms={case['user_duration_ms']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
