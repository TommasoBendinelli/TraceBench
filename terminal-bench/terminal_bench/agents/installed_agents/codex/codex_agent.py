import json
import os
import shlex
import textwrap
import time
from pathlib import Path

from terminal_bench.agents.agent_name import AgentName
from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.utils.logger import logger


class CodexAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return AgentName.CODEX.value

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name.split("/")[-1]
        self._reasoning = kwargs.get("reasoning")
        self._version = kwargs.get("version", "latest")
        self._hint = kwargs.get("hint", "")
        self._logger = logger.getChild(__name__)

    @property
    def _env(self) -> dict[str, str]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise KeyError("OPENAI_API_KEY must be set")
        return {"OPENAI_API_KEY": api_key}

    @property
    def _install_agent_script_path(self) -> Path:
        return self._get_templated_script_path("codex-setup.sh.j2")

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped_instruction = shlex.quote(instruction)
        reasoning_flag = ""
        if self._reasoning:
            reasoning_flag = f"--config model_reasoning_effort={shlex.quote(str(self._reasoning))} "
        return [
            TerminalCommand(
                command=(
                    "codex exec "
                    "--sandbox danger-full-access "
                    "--skip-git-repo-check "
                    "--json "
                    f"--model {self._model_name} "
                    f"{reasoning_flag}"
                    "-- "  # end of flags
                    f"{escaped_instruction}"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            )
        ]

    def _get_container_home(self, session: TmuxSession) -> str:
        result = session.container.exec_run(["bash", "-lc", "echo -n $HOME"])

        if result.exit_code == 0:
            home_dir = result.output.decode(errors="ignore").strip()
            if home_dir:
                return home_dir

        return "/root"

    def _get_codex_sessions_dir(self, session: TmuxSession) -> str:
        container_home = self._get_container_home(session)
        return f"{container_home}/.codex/sessions"

    def _read_latest_session_file(
        self,
        session: TmuxSession,
        sessions_dir: str,
        earliest_timestamp: float | None,
    ) -> str | None:
        """
        Return the newest Codex session jsonl path, optionally filtered by mtime.
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

            base = os.path.expanduser({sessions_dir!r})
            paths = glob.glob(
                os.path.join(base, "**", "rollout-*.jsonl"), recursive=True
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

            print(json.dumps({{"path": latest[1]}}))
            PY
            """
        )

        result = session.container.exec_run(["bash", "-lc", script])
        if result.exit_code != 0:
            return None

        output = result.output.decode(errors="ignore").strip()
        if not output:
            return None

        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            self._logger.warning("Failed to decode Codex session payload: %s", output)
            return None

        path = payload.get("path")
        return path if isinstance(path, str) else None

    def _get_latest_session_file(
        self,
        session: TmuxSession,
        started_at: float,
    ) -> str | None:
        """
        Fetch the newest Codex session created after the command started.
        If none are found, fall back to the newest session overall.
        """
        sessions_dir = self._get_codex_sessions_dir(session)

        session_file = self._read_latest_session_file(
            session=session,
            sessions_dir=sessions_dir,
            earliest_timestamp=started_at,
        )

        if session_file is None:
            session_file = self._read_latest_session_file(
                session=session,
                sessions_dir=sessions_dir,
                earliest_timestamp=None,
            )

        if session_file:
            self._logger.debug("Using Codex session file: %s", session_file)

        return session_file

    def _persist_session_file(self, session: TmuxSession, session_file_path: str | None):
        """
        Copy the session jsonl file into the mounted agent-logs directory.
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
                "Failed to persist Codex session file to agent logs: %s",
                session_file_path,
            )

    @staticmethod
    def _parse_usage_from_output(output: str) -> dict[str, int] | None:
        """
        Parse Codex CLI JSON output to extract usage numbers.
        We scan line by line to avoid tripping on prompts or partial lines.
        """
        usage: dict[str, int] | None = None

        for line in output.splitlines():
            stripped = line.strip()
            if not stripped or not stripped.startswith("{"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if payload.get("type") == "turn.completed" and isinstance(
                payload.get("usage"), dict
            ):
                usage = payload["usage"]

        return usage

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        # Keep the same installation flow as AbstractInstalledAgent, but capture
        # Codex JSON output to populate token usage.
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

        rendered_instruction = self._render_instruction(instruction)
        run_agent_commands = self._run_agent_commands(rendered_instruction)

        command_started_at = time.time()
        for command in run_agent_commands:
            session.send_command(command)

        if os.environ.get("BREAKPOINT_DOCKER") == "1":
            breakpoint()
        session_file_path = self._get_latest_session_file(
            session=session,
            started_at=command_started_at,
        )
        self._persist_session_file(session, session_file_path)

        codex_output = session.capture_pane(capture_entire=True)
        usage = self._parse_usage_from_output(codex_output)

        if usage is None:
            self._logger.warning(
                "Codex CLI output did not contain usage; falling back to zeroed metrics."
            )
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
            )

        total_input_tokens = usage.get("input_tokens", 0)
        total_input_cached_tokens = usage.get("cached_input_tokens", 0)
        return AgentResult(
            total_input_tokens=total_input_tokens,
            total_input_cached_tokens=total_input_cached_tokens,
            total_output_tokens=usage.get("output_tokens", 0),
        )
