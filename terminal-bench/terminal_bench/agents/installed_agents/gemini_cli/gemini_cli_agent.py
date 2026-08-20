import json
import os
import shlex
import textwrap
import time
from pathlib import Path
from typing import Any

from terminal_bench.agents.agent_name import AgentName
from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.utils.logger import logger
from terminal_bench.utils.subscription_errors import (
    SubscriptionError,
    has_quota_exceeded,
)

_TSENV_GEMINI_BASE_MODEL = "gemini-3.1-pro-preview"
_TSENV_GEMINI_ALIAS_THINKING_LEVELS = {
    "gemini_3_1_pro_high": "HIGH",
    "gemini_3_1_pro_low": "LOW",
}


class GeminiCliAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return AgentName.GEMINI_CLI.value

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name.split("/")[-1]
        self._version = kwargs.get("version", "latest")
        self._hint = kwargs.get("hint", "")
        self._logger = logger.getChild(__name__)
        self._auth_metadata = {
            "agent_name": self.name(),
            "mode_name": "api",
            "was_gemini_api_key_used": True,
            "was_gemini_api_auth_used": False,
        }

    @property
    def _env(self) -> dict[str, str]:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise KeyError("GEMINI_API_KEY must be set")
        return {"GEMINI_API_KEY": api_key}

    @property
    def _install_agent_script_path(self) -> os.PathLike:
        return self._get_templated_script_path("gemini-cli-setup.sh.j2")

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped_instruction = shlex.quote(instruction)
        return [
            TerminalCommand(
                command=(
                    "gemini "
                    f"-p {escaped_instruction} "
                    "-y "
                    "--output-format json "
                    f"-m {self._model_name}"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]

    def _get_gemini_tmp_dir(self, session: TmuxSession) -> str:
        container_home = self._get_container_home(session)
        return f"{container_home}/.gemini/tmp"

    def _persist_session_file(
        self, session: TmuxSession, session_file_path: str | None
    ) -> None:
        """
        Copy the session JSON into the mounted agent-logs directory for debugging.
        """
        if not session_file_path:
            return

        destination_dir = self.CONTAINER_AGENT_LOGS_PATH
        destination_path = f"{destination_dir}/{Path(session_file_path).name}"

        copy_cmd = (
            f"mkdir -p {shlex.quote(destination_dir)} && "
            f"cp -f {shlex.quote(session_file_path)} {shlex.quote(destination_path)}"
        )
        result = session.container.exec_run(["bash", "-lc", copy_cmd])
        if result.exit_code != 0:
            self._logger.warning(
                "Failed to persist Gemini session file to agent logs: %s",
                session_file_path,
            )

    def _read_latest_session_data(
        self,
        session: TmuxSession,
        gemini_tmp_dir: str,
        earliest_timestamp: float | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Read the newest Gemini CLI session JSON created after earliest_timestamp.
        Falls back to None when no session file matches.
        """
        
        timestamp_filter = ""
        if earliest_timestamp is not None:
            timestamp_filter = (
                f"        if mtime < {earliest_timestamp}:\n"
                "            continue\n"
            )

        script = textwrap.dedent(
            f"""
            python - <<'PY'
            import glob
            import json
            import os
            import sys

            base = os.path.expanduser({gemini_tmp_dir!r})
            paths = glob.glob(
                os.path.join(base, "**", "chats", "session-*.json"), recursive=True
            )

            latest = None
            for path in paths:
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
{timestamp_filter}                if latest is None or mtime > latest[0]:
                    latest = (mtime, path)

            if latest is None:
                sys.exit(1)

            path = latest[1]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(json.dumps({{"path": path, "data": data}}))
            PY
            """
        )

        result = session.container.exec_run(["bash", "-lc", script])
        if result.exit_code != 0:
            return None, None

        output = result.output.decode(errors="ignore").strip()
        if not output:
            return None, None

        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode Gemini session payload: %s", output)
            return None, None

        data = payload.get("data")
        path = payload.get("path")
        return data if isinstance(data, dict) else None, path

    def _get_latest_session_data(
        self,
        session: TmuxSession,
        started_at: float,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Fetch the newest Gemini session created after the command started.
        If none are found, fall back to the newest session overall.
        """
        gemini_tmp_dir = self._get_gemini_tmp_dir(session)

        data, path = self._read_latest_session_data(
            session=session,
            gemini_tmp_dir=gemini_tmp_dir,
            earliest_timestamp=started_at,
        )

        if data is None:
            data, path = self._read_latest_session_data(
                session=session,
                gemini_tmp_dir=gemini_tmp_dir,
                earliest_timestamp=None,
            )

        if path:
            self._logger.debug("Using Gemini session file: %s", path)

        return data, path

    @staticmethod
    def _get_output_delta(before: str, after: str) -> str:
        if after.startswith(before):
            return after[len(before) :]
        return after

    def _run_gemini_commands(
        self,
        session: TmuxSession,
        commands: list[TerminalCommand],
        env: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        
        if env:
            env_prefix = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in env.items()
            )
            env_prefix = f"{env_prefix}"
            updated_commands = []
            for command in commands:
                command_data = command.model_dump()
                command_data["command"] = f"{env_prefix} {command.command}"
                updated_commands.append(TerminalCommand(**command_data))
            commands = updated_commands
        output_before = session.capture_pane(capture_entire=True)
        started_at = time.time()
        for command in commands:
            session.send_command(command)
        output_after = session.capture_pane(capture_entire=True)
        output_delta =  output_after[len(output_before):]
        session_data, session_file_path = self._get_latest_session_data(
            session=session,
            started_at=started_at,
        )
        self._persist_session_file(session, session_file_path)
        return session_data, output_delta

    @staticmethod
    def _extract_usage_from_session(session_data: dict[str, Any]) -> dict[str, int] | None:
        """
        Sum token usage across Gemini responses from the saved session JSON.
        """
        if not isinstance(session_data, dict):
            return None

        total_input = 0
        total_cached = 0
        total_output = 0

        for message in session_data.get("messages", []):
            tokens = message.get("tokens") or {}
            try:
                total_input += int(tokens.get("input", 0) or 0)
                total_cached += int(tokens.get("cached", 0) or 0)
                total_output += int(tokens.get("output", 0) or 0)
            except (TypeError, ValueError):
                continue

        if total_input == 0 and total_cached == 0 and total_output == 0:
            return None

        return {
            "input_tokens": total_input,
            "cached_input_tokens": total_cached,
            "output_tokens": total_output,
        }

    def _get_container_home(self, session: TmuxSession) -> str:
        result = session.container.exec_run(["bash", "-lc", "echo -n $HOME"])

        if result.exit_code == 0:
            home_dir = result.output.decode(errors="ignore").strip()
            if home_dir:
                return home_dir

        return "/root"

    def _ensure_gemini_settings_in_container(self, session: TmuxSession) -> None:
        container_home = self._get_container_home(session)
        settings_path = f"{container_home}/.gemini/settings.json"
        aliases_json = json.dumps(_TSENV_GEMINI_ALIAS_THINKING_LEVELS)
        base_model_json = json.dumps(_TSENV_GEMINI_BASE_MODEL)
        script = textwrap.dedent(
            f"""
            python - <<'PY'
            import json
            from pathlib import Path

            path = Path({settings_path!r})
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {{}}
            else:
                data = {{}}
            path.parent.mkdir(parents=True, exist_ok=True)

            model = data.get("model")
            if not isinstance(model, dict):
                model = {{}}
            data["model"] = model
            model["skipNextSpeakerCheck"] = True

            model_configs = data.get("modelConfigs")
            if not isinstance(model_configs, dict):
                model_configs = {{}}
            data["modelConfigs"] = model_configs
            custom_aliases = model_configs.get("customAliases")
            if not isinstance(custom_aliases, dict):
                custom_aliases = {{}}
            model_configs["customAliases"] = custom_aliases

            base_model = {base_model_json}
            for alias, thinking_level in {aliases_json}.items():
                custom_aliases[alias] = {{
                    "extends": "chat-base-3",
                    "modelConfig": {{
                        "model": base_model,
                        "generateContentConfig": {{
                            "thinkingConfig": {{
                                "thinkingLevel": thinking_level,
                            }},
                        }},
                    }},
                }}

            path.write_text(json.dumps(data), encoding="utf-8")
            PY
            """
        )
        result = session.container.exec_run(["bash", "-lc", script])
        if result.exit_code != 0:
            raise RuntimeError("Failed to update Gemini settings.json aliases.")

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        
        self._logging_dir = logging_dir

        session.copy_to_container(
            self._install_agent_script_path,
            container_dir="/installed-agent",
            container_filename="install-agent.sh",
        )

        env_setup_content = self._create_env_setup_file()
        session.container.exec_run(
            [
                "sh",
                "-c",
                (
                    f"echo {shlex.quote(env_setup_content)} > "
                    "/installed-agent/setup-env.sh"
                ),
            ]
        )

        session.send_keys(
            [
                "source /installed-agent/setup-env.sh",
                "Enter",
            ],
            block=True,
            max_timeout_sec=float("inf"),
        )

       
       
        installation_failed = self._run_install_script(session)
        if installation_failed:
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.AGENT_INSTALLATION_FAILED,
            )
        self._ensure_gemini_settings_in_container(session)

        rendered_instruction = self._render_instruction(instruction)
        run_agent_commands = self._run_agent_commands(rendered_instruction)
        session_data, output_delta = self._run_gemini_commands(
            session=session,
            commands=run_agent_commands,
            env=self._env,
        )
        if has_quota_exceeded(output_delta):
            raise SubscriptionError(
                "Gemini CLI returned a subscription/quota error response."
            )
        usage = self._extract_usage_from_session(session_data) if session_data else None

        if usage is None:
            self._logger.warning(
                "Gemini session JSON missing or did not include token usage; "
                "falling back to zeroed metrics."
            )
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
            )

        return AgentResult(
            total_input_tokens=usage["input_tokens"],
            total_input_cached_tokens=usage["cached_input_tokens"],
            total_output_tokens=usage["output_tokens"],
        )
