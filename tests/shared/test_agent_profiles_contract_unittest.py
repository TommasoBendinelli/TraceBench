from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import benchmark_utils


class TestAgentProfilesContract(unittest.TestCase):
    def test_agent_registry_matches_appendix_agent_ids(self) -> None:
        self.assertEqual(
            [str(entry["agent_id"]) for entry in benchmark_utils.AGENTIC_PROFILES],
            [
                "gpt_5_5_codex_high",
                "gpt_5_5_low_codex_low",
                "gpt_5_3_spark_codex_high",
                "gemini_3_1_pro_high",
                "gemini_3_1_pro_low",
                "claude_4_opus_high",
                "claude_4_opus_low",
                "minimax-m2.7",
            ],
        )

    def test_gemini_profiles_use_runtime_aliases(self) -> None:
        profiles = benchmark_utils.AGENTIC_PROFILES_BY_ID

        self.assertEqual(
            profiles["gemini_3_1_pro_high"]["model_name"],
            "gemini_3_1_pro_high",
        )
        self.assertEqual(
            profiles["gemini_3_1_pro_low"]["model_name"],
            "gemini_3_1_pro_low",
        )

    def test_gpt_5_5_profiles_use_official_standard_token_pricing(self) -> None:
        profiles = benchmark_utils.AGENTIC_PROFILES_BY_ID
        expected = {
            "input_token": 5.0,
            "cached_token": 0.5,
            "completion_token": 30.0,
        }

        self.assertEqual(profiles["gpt_5_5_codex_high"]["cost"], expected)
        self.assertEqual(profiles["gpt_5_5_low_codex_low"]["cost"], expected)

    def test_opencode_platform_uses_documented_openrouter_env_var(self) -> None:
        payload = benchmark_utils._load_platform_registry()

        self.assertEqual(
            payload["opencode"]["api_key_environment_variable"],
            "OPENROUTER_API_KEY",
        )

    def test_platform_registry_uses_documented_environmental_key_and_default_tools(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            platform_path = Path(tmp_dir) / "platform.json"
            platform_path.write_text(
                json.dumps(
                    {
                        "codex": {
                            "name": "codex",
                            "npm_package": "@openai/codex@0.118",
                            "api_key_environment_variable": "OPENAI_API_KEY",
                            "default_tools": {},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(benchmark_utils, "_PLATFORM_REGISTRY_PATH", platform_path):
                payload = benchmark_utils._load_platform_registry()

        self.assertEqual(
            payload["codex"]["api_key_environment_variable"],
            "OPENAI_API_KEY",
        )
        self.assertEqual(payload["codex"]["default_tools"], {})

    def test_platform_registry_rejects_legacy_misspelled_environmental_key(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            platform_path = Path(tmp_dir) / "platform.json"
            platform_path.write_text(
                json.dumps(
                    {
                        "codex": {
                            "name": "codex",
                            "npm_package": "@openai/codex@0.118",
                            "required_enviromental_variable": "OPENAI_API_KEY",
                            "default_tools": {},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(benchmark_utils, "_PLATFORM_REGISTRY_PATH", platform_path):
                with self.assertRaisesRegex(ValueError, "api_key_environment_variable"):
                    benchmark_utils._load_platform_registry()

    def test_platform_registry_requires_default_tools(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            platform_path = Path(tmp_dir) / "platform.json"
            platform_path.write_text(
                json.dumps(
                    {
                        "codex": {
                            "name": "codex",
                            "npm_package": "@openai/codex@0.118",
                            "api_key_environment_variable": "OPENAI_API_KEY",
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(benchmark_utils, "_PLATFORM_REGISTRY_PATH", platform_path):
                with self.assertRaisesRegex(ValueError, "default_tools"):
                    benchmark_utils._load_platform_registry()

    def test_platform_registry_requires_api_key_variable_to_be_a_string(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            platform_path = Path(tmp_dir) / "platform.json"
            platform_path.write_text(
                json.dumps(
                    {
                        "codex": {
                            "name": "codex",
                            "npm_package": "@openai/codex@0.118",
                            "api_key_environment_variable": ["OPENAI_API_KEY"],
                            "default_tools": {},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(benchmark_utils, "_PLATFORM_REGISTRY_PATH", platform_path):
                with self.assertRaisesRegex(TypeError, "must be a string"):
                    benchmark_utils._load_platform_registry()

    def test_agent_registry_exposes_documented_platform_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            platform_path = Path(tmp_dir) / "platform.json"
            agent_path = Path(tmp_dir) / "agents.json"
            platform_path.write_text(
                json.dumps(
                    {
                        "codex": {
                            "name": "codex",
                            "npm_package": "@openai/codex@0.118",
                            "api_key_environment_variable": "OPENAI_API_KEY",
                            "default_tools": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            agent_path.write_text(
                json.dumps(
                    [
                        {
                            "agent_id": "gpt_5_5_codex_high",
                            "platform_id": "codex",
                            "cost": {
                                "input_token": 2.5,
                                "cached_token": 0.25,
                                "completion_token": 15.0,
                            },
                            "support_images": True,
                            "available_tools": {},
                            "model_name": "gpt-5.5",
                            "paper_facing_name": "gpt-5.5 (high)",
                            "reasoning": "high",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(benchmark_utils, "_PLATFORM_REGISTRY_PATH", platform_path),
                mock.patch.object(benchmark_utils, "_AGENT_REGISTRY_PATH", agent_path),
            ):
                payload = benchmark_utils._load_agent_registry()

        self.assertEqual(
            payload[0]["api_key_environment_variable"],
            "OPENAI_API_KEY",
        )
        self.assertEqual(payload[0]["paper_facing_name"], "gpt-5.5 (high)")
        self.assertEqual(payload[0]["agentic_platform"]["default_tools"], {})

    def test_agent_registry_rejects_empty_paper_facing_name(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            platform_path = Path(tmp_dir) / "platform.json"
            agent_path = Path(tmp_dir) / "agents.json"
            platform_path.write_text(
                json.dumps(
                    {
                        "codex": {
                            "name": "codex",
                            "npm_package": "@openai/codex@0.118",
                            "api_key_environment_variable": "OPENAI_API_KEY",
                            "default_tools": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            agent_path.write_text(
                json.dumps(
                    [
                        {
                            "agent_id": "gpt_5_5_codex_high",
                            "platform_id": "codex",
                            "cost": {
                                "input_token": 2.5,
                                "cached_token": 0.25,
                                "completion_token": 15.0,
                            },
                            "support_images": True,
                            "available_tools": {},
                            "model_name": "gpt-5.5",
                            "paper_facing_name": " ",
                            "reasoning": "high",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(benchmark_utils, "_PLATFORM_REGISTRY_PATH", platform_path),
                mock.patch.object(benchmark_utils, "_AGENT_REGISTRY_PATH", agent_path),
            ):
                with self.assertRaisesRegex(ValueError, "paper_facing_name"):
                    benchmark_utils._load_agent_registry()

    def test_agent_registry_rejects_duplicate_paper_facing_name(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            platform_path = Path(tmp_dir) / "platform.json"
            agent_path = Path(tmp_dir) / "agents.json"
            platform_path.write_text(
                json.dumps(
                    {
                        "codex": {
                            "name": "codex",
                            "npm_package": "@openai/codex@0.118",
                            "api_key_environment_variable": "OPENAI_API_KEY",
                            "default_tools": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            base_profile = {
                "platform_id": "codex",
                "cost": {
                    "input_token": 2.5,
                    "cached_token": 0.25,
                    "completion_token": 15.0,
                },
                "support_images": True,
                "available_tools": {},
                "model_name": "gpt-5.5",
                "paper_facing_name": "gpt-5.5",
                "reasoning": "high",
            }
            agent_path.write_text(
                json.dumps(
                    [
                        {"agent_id": "gpt_5_5_codex_high", **base_profile},
                        {"agent_id": "gpt_5_5_codex_low", **base_profile},
                    ]
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(benchmark_utils, "_PLATFORM_REGISTRY_PATH", platform_path),
                mock.patch.object(benchmark_utils, "_AGENT_REGISTRY_PATH", agent_path),
            ):
                with self.assertRaisesRegex(ValueError, "Duplicate paper_facing_name"):
                    benchmark_utils._load_agent_registry()


if __name__ == "__main__":
    unittest.main()
