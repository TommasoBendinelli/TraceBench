from pathlib import Path

from terminal_bench.agents.agent_name import AgentName
from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.terminus_1 import Terminus
from terminal_bench.agents.terminus_2 import Terminus2
from terminal_bench.terminal.tmux_session import TmuxSession


class _TerminusTimeSeriesSchema(Terminus):
    """
    Terminus-1 variant tuned for SimBench UCR/UCR Anomaly time-series tasks.
    Uses a domain-specific prompt and slightly lower temperature for more
    deterministic command sequencing.
    """

    PROMPT_TEMPLATE_PATH = (
        Path(__file__).parent / "prompt-templates/terminus-timeseries.txt"
    )

    def __init__(
        self,
        model_name: str,
        max_episodes: int = 80,
        api_base: str | None = None,
        temperature: float = 0.35,
        custom_llm_provider: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            max_episodes=max_episodes,
            api_base=api_base,
            temperature=temperature,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )


class _TerminusTimeSeriesPlain(Terminus2):
    """Terminus-2 variant tuned for time-series tasks with JSON/XML parsing."""

    PROMPT_TEMPLATE_JSON_PATH = (
        Path(__file__).parent / "prompt-templates/terminus-timeseries-json-plain.txt"
    )
    PROMPT_TEMPLATE_XML_PATH = (
        Path(__file__).parent / "prompt-templates/terminus-timeseries-xml-plain.txt"
    )

    def _get_prompt_template_path(self) -> Path:
        if self._parser_name == "json":
            return self.PROMPT_TEMPLATE_JSON_PATH
        if self._parser_name == "xml":
            return self.PROMPT_TEMPLATE_XML_PATH
        raise ValueError(
            f"Unknown parser_name: {self._parser_name}. Use 'json' or 'xml'."
        )


class _TerminusTimeSeriesPlainXmlThink(Terminus2):
    """Terminus-2 XML variant with an optional <think> block in the response."""

    PROMPT_TEMPLATE_XML_THINK_PATH = (
        Path(__file__).parent
        / "prompt-templates/terminus-timeseries-xml-plain-think.txt"
    )

    def __init__(
        self,
        model_name: str,
        max_episodes: int = 80,
        api_base: str | None = None,
        temperature: float = 0.35,
        custom_llm_provider: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            max_episodes=max_episodes,
            parser_name="xml",
            api_base=api_base,
            temperature=temperature,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

    def _get_prompt_template_path(self) -> Path:
        return self.PROMPT_TEMPLATE_XML_THINK_PATH


class TerminusTimeSeries(BaseAgent):
    """
    Time-series Terminus agent with optional XML/JSON parsing.

    - Default: schema-based JSON (Terminus-1 prompt).
    - parser_name=json/xml: use Terminus-2 plain parsing with time-series prompts.
    """

    def __init__(
        self,
        model_name: str,
        max_episodes: int = 80,
        api_base: str | None = None,
        temperature: float = 0.35,
        custom_llm_provider: str | None = None,
        parser_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._parser_name = parser_name

        if parser_name is None:
            self._delegate = _TerminusTimeSeriesSchema(
                model_name=model_name,
                max_episodes=max_episodes,
                api_base=api_base,
                temperature=temperature,
                custom_llm_provider=custom_llm_provider,
                **kwargs,
            )
        else:
            if parser_name not in {"json", "xml"}:
                raise ValueError("parser_name must be 'json', 'xml', or None.")
            self._delegate = _TerminusTimeSeriesPlain(
                model_name=model_name,
                max_episodes=max_episodes,
                parser_name=parser_name,
                api_base=api_base,
                temperature=temperature,
                custom_llm_provider=custom_llm_provider,
                **kwargs,
            )

    @staticmethod
    def name() -> str:
        return AgentName.TERMINUS_TS.value

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        return self._delegate.perform_task(instruction, session, logging_dir)


class TerminusTimeSeriesXmlThink(BaseAgent):
    """Time-series Terminus-2 XML agent with optional <think> blocks."""

    def __init__(
        self,
        model_name: str,
        max_episodes: int = 80,
        api_base: str | None = None,
        temperature: float = 0.35,
        custom_llm_provider: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._delegate = _TerminusTimeSeriesPlainXmlThink(
            model_name=model_name,
            max_episodes=max_episodes,
            api_base=api_base,
            temperature=temperature,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return AgentName.TERMINUS_TS_XML_THINK.value

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        return self._delegate.perform_task(instruction, session, logging_dir)
