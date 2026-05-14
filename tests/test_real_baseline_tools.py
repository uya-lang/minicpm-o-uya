#!/usr/bin/env python3
import json
import os
import stat
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_TTS = REPO_ROOT / "tests" / "compare_tts_token_alignment.py"
RECORD_TTS = REPO_ROOT / "tests" / "record_tts_seeded_baseline.py"
COMPARE_AUDIO2AUDIO = REPO_ROOT / "tests" / "compare_audio2audio_baseline.py"


def write_wav(path: Path, frames: int, sample_rate: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


class RealBaselineToolTests(unittest.TestCase):
    def test_seeded_tts_manifest_and_strict_compare(self):
        with tempfile.TemporaryDirectory(prefix="minicpm-o-uya-test-tts-") as tmp:
            root = Path(tmp)
            llm_debug_dir = root / "llm_debug"
            ref_dir = root / "ref"
            manifest_path = root / "tts_baselines.json"
            summary_path = root / "align_summary.json"
            for index in range(2):
                chunk_dir = llm_debug_dir / f"chunk_{index}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                (chunk_dir / "llm_token_ids.txt").write_text("1,2,3\n", encoding="utf-8")
            ref_dir.mkdir(parents=True, exist_ok=True)
            (ref_dir / "audio_tokens_chunk_0.txt").write_text("11,22,33\n", encoding="utf-8")
            (ref_dir / "audio_tokens_chunk_1.txt").write_text("44,55\n", encoding="utf-8")

            record = subprocess.run(
                [
                    "python3",
                    str(RECORD_TTS),
                    "--output-manifest",
                    str(manifest_path),
                    "--baseline-name",
                    "seeded_case",
                    "--llm-debug-dir",
                    str(llm_debug_dir),
                    "--ref-dir",
                    str(ref_dir),
                    "--seed",
                    "7",
                    "--temperature",
                    "0",
                    "--greedy",
                    "--command",
                    "llama-omni-single-test-audio --seed 7 --temp 0",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            self.assertIn("tts-seeded-baseline-recorded", record.stdout)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            baseline = manifest["baselines"][0]
            self.assertEqual(baseline["name"], "seeded_case")
            self.assertEqual(baseline["seed"], 7)
            self.assertTrue(baseline["greedy"])
            self.assertTrue(baseline["deterministic_oracle"])
            self.assertEqual(baseline["chunk_count"], 2)
            self.assertFalse(baseline["allow_max_audio_tokens_stop"])

            fake_uya = root / "fake_uya.py"
            fake_uya.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import os",
                        "import sys",
                        "from pathlib import Path",
                        "def main():",
                        "    argv = sys.argv[1:]",
                        "    out_dir = Path(argv[argv.index('--out-dir') + 1])",
                        "    ref_dir = Path(os.environ['REF_DIR'])",
                        "    out_dir.mkdir(parents=True, exist_ok=True)",
                        "    for path in sorted(ref_dir.glob('audio_tokens_chunk_*.txt')):",
                        "        text = path.read_text(encoding='utf-8')",
                        "        if os.environ.get('MISMATCH') == '1' and path.name == 'audio_tokens_chunk_0.txt':",
                        "            text = '99,22,33\\n'",
                        "        (out_dir / path.name).write_text(text, encoding='utf-8')",
                        "    return 0",
                        "if __name__ == '__main__':",
                        "    raise SystemExit(main())",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_uya.chmod(fake_uya.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env["REF_DIR"] = str(ref_dir)
            ok = subprocess.run(
                [
                    "python3",
                    str(COMPARE_TTS),
                    "--uya",
                    str(fake_uya),
                    "--tts-model",
                    str(root / "tts.gguf"),
                    "--projector-model",
                    str(root / "projector.gguf"),
                    "--baseline-manifest",
                    str(manifest_path),
                    "--baseline-name",
                    "seeded_case",
                    "--output-json",
                    str(summary_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            self.assertIn("tts-token-align: PASS", ok.stdout)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["pass"])
            self.assertTrue(summary["require_exact"])
            self.assertTrue(summary["greedy"])
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(summary["count"], 2)

            env["MISMATCH"] = "1"
            mismatch = subprocess.run(
                [
                    "python3",
                    str(COMPARE_TTS),
                    "--uya",
                    str(fake_uya),
                    "--tts-model",
                    str(root / "tts.gguf"),
                    "--projector-model",
                    str(root / "projector.gguf"),
                    "--baseline-manifest",
                    str(manifest_path),
                    "--baseline-name",
                    "seeded_case",
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("FAIL exact mismatch", mismatch.stdout)

    def test_audio2audio_baseline_compare_writes_json(self):
        with tempfile.TemporaryDirectory(prefix="minicpm-o-uya-test-a2a-") as tmp:
            root = Path(tmp)
            baseline_dir = root / "baseline"
            summary_path = root / "compare.json"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            answer_text = baseline_dir / "answer.txt"
            answer_text.write_text("你好，基线回答。", encoding="utf-8")
            baseline_answer_wav = baseline_dir / "answer.wav"
            baseline_turn_wav = baseline_dir / "turn.wav"
            write_wav(baseline_answer_wav, frames=24000)
            write_wav(baseline_turn_wav, frames=36000)

            baseline_manifest = root / "audio2audio_baselines.json"
            baseline_manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "complex_case2",
                                "protocol": "0000=reference voice, 0001=user audio",
                                "answer_text": "baseline/answer.txt",
                                "answer_audio": "baseline/answer.wav",
                                "turn_audio": "baseline/turn.wav",
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            uya_report = root / "uya_report.json"
            uya_report.write_text(
                json.dumps(
                    {
                        "turns": [
                            {
                                "prefix": "turn_0001",
                                "answer_text_chars": 8,
                                "answer_wav": {
                                    "duration_ms": 900.0,
                                    "sample_rate": 24000,
                                    "channels": 1,
                                },
                                "turn_wav": {
                                    "duration_ms": 1400.0,
                                    "sample_rate": 24000,
                                    "channels": 1,
                                },
                                "timing": {
                                    "total_ms": 12345,
                                    "first_audio_ms": 6789,
                                    "rtf": 2.5,
                                    "peak_rss_kb": 345678,
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "python3",
                    str(COMPARE_AUDIO2AUDIO),
                    "--uya-report",
                    str(uya_report),
                    "--baseline-manifest",
                    str(baseline_manifest),
                    "--case-name",
                    "complex_case2",
                    "--output-json",
                    str(summary_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("audio2audio-baseline: PASS", proc.stdout)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["pass"])
            self.assertEqual(summary["case"]["name"], "complex_case2")
            self.assertEqual(summary["timing"]["total_ms"], 12345)
            self.assertAlmostEqual(summary["answer_wav"]["delta_ms"], -100.0)
            self.assertAlmostEqual(summary["turn_wav"]["delta_ms"], -100.0)


if __name__ == "__main__":
    unittest.main()
