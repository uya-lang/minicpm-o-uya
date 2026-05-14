#!/usr/bin/env python3
import argparse
import json
import wave
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wav_info(path: Path):
    if not path.exists():
        return None
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
    return {
        "path": str(path),
        "frames": frames,
        "sample_rate": rate,
        "channels": channels,
        "duration_ms": frames * 1000.0 / rate if rate else 0.0,
    }


def load_text(path: Path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def find_case(manifest, case_name):
    for case in manifest.get("cases", []):
        if case.get("name") == case_name:
            return case
    raise ValueError(f"missing baseline case {case_name}")


def select_uya_turn(report):
    turns = report.get("turns", [])
    if not turns:
        raise ValueError("uya report has no turns")
    return turns[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Compare one Uya audio2audio report JSON against a local llama.cpp-omni baseline manifest entry."
    )
    parser.add_argument("--uya-report", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--case-name", required=True)
    args = parser.parse_args()

    report = load_json(Path(args.uya_report))
    baseline_manifest = load_json(Path(args.baseline_manifest))
    baseline_case = find_case(baseline_manifest, args.case_name)
    uya_turn = select_uya_turn(report)

    baseline_answer_text = load_text(Path(baseline_case["answer_text"]))
    baseline_answer_wav = wav_info(Path(baseline_case["answer_audio"]))
    baseline_turn_wav = wav_info(Path(baseline_case["turn_audio"]))

    uya_answer_wav = uya_turn.get("answer_wav")
    uya_turn_wav = uya_turn.get("turn_wav")
    uya_answer_chars = uya_turn.get("answer_text_chars", 0)

    print(
        "audio2audio-baseline case: "
        f"name={baseline_case['name']} protocol={baseline_case['protocol']}"
    )
    print(
        "audio2audio-baseline answer-text: "
        f"baseline_chars={len(baseline_answer_text)} uya_chars={uya_answer_chars} "
        f"delta={uya_answer_chars - len(baseline_answer_text)}"
    )

    if baseline_answer_wav and uya_answer_wav:
        print(
            "audio2audio-baseline answer-wav: "
            f"baseline_ms={baseline_answer_wav['duration_ms']:.3f} "
            f"uya_ms={uya_answer_wav['duration_ms']:.3f} "
            f"delta_ms={uya_answer_wav['duration_ms'] - baseline_answer_wav['duration_ms']:.3f}"
        )
    if baseline_turn_wav and uya_turn_wav:
        print(
            "audio2audio-baseline turn-wav: "
            f"baseline_ms={baseline_turn_wav['duration_ms']:.3f} "
            f"uya_ms={uya_turn_wav['duration_ms']:.3f} "
            f"delta_ms={uya_turn_wav['duration_ms'] - baseline_turn_wav['duration_ms']:.3f}"
        )

    timing = uya_turn.get("timing", {})
    print(
        "audio2audio-baseline timing: "
        f"total_ms={timing.get('total_ms')} "
        f"first_audio_ms={timing.get('first_audio_ms')} "
        f"rtf={timing.get('rtf')}"
    )
    print("audio2audio-baseline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
