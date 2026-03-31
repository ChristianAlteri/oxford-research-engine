"""Research engine — the core loop that orchestrates agents in structured rounds."""

from __future__ import annotations

import logging
import signal
from typing import Callable

from ore.agents.builder import BuilderAgent
from ore.agents.historian import HistorianAgent
from ore.agents.referee import RefereeAgent
from ore.agents.skeptic import SkepticAgent
from ore.config import EngineConfig
from ore.llm.provider import LLMProvider
from ore.memory.store import JsonMemoryStore
from ore.research_objects import ResearchObject, RoundScore

logger = logging.getLogger(__name__)


class ResearchEngine:
    """Orchestrates the research loop: Builder -> Skeptic -> Builder -> Historian -> Referee."""

    def __init__(
        self,
        config: EngineConfig,
        memory: JsonMemoryStore,
        llm: LLMProvider | None = None,
        on_object: Callable[[ResearchObject, str], None] | None = None,
        on_round_start: Callable[[int], None] | None = None,
        on_round_end: Callable[[int, RoundScore], None] | None = None,
        on_budget_warning: Callable[[float, float], None] | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.llm = llm or LLMProvider()
        self._interrupted = False

        self.on_object = on_object or (lambda obj, phase: None)
        self.on_round_start = on_round_start or (lambda r: None)
        self.on_round_end = on_round_end or (lambda r, s: None)
        self.on_budget_warning = on_budget_warning or (lambda spent, budget: None)

        agent_cfg = config.agents
        self.builder = BuilderAgent(agent_cfg.get("builder", config.agents["builder"]), self.llm)
        self.skeptic = SkepticAgent(agent_cfg.get("skeptic", config.agents["skeptic"]), self.llm)
        self.historian = HistorianAgent(
            agent_cfg.get("historian", config.agents["historian"]), self.llm
        )
        self.referee = RefereeAgent(agent_cfg.get("referee", config.agents["referee"]), self.llm)

    def _handle_interrupt(self, signum: int, frame: object) -> None:
        logger.info("Interrupt received — finishing current round gracefully.")
        self._interrupted = True

    def run(self) -> list[ResearchObject]:
        """Execute the full research loop. Returns all produced research objects."""
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        try:
            return self._run_loop()
        finally:
            signal.signal(signal.SIGINT, original_handler)

    def _run_loop(self) -> list[ResearchObject]:
        question = self.config.question
        start_round = self._detect_start_round()

        for round_num in range(start_round, self.config.max_rounds + 1):
            if self._interrupted:
                logger.info("Stopped by user after round %d.", round_num - 1)
                break

            if self._over_budget():
                logger.info(
                    "Budget exhausted ($%.4f / $%.2f). Stopping.",
                    self.llm.budget_spent,
                    self.config.budget_usd,
                )
                break

            self.on_round_start(round_num)
            score = self._run_round(round_num, question)
            self.on_round_end(round_num, score)

            self.memory.save_round(round_num)
            self.memory.save()

            if score.verdict == "stop":
                logger.info("Referee called stop at round %d: %s", round_num, score.rationale)
                break

            if score.verdict == "pivot":
                logger.info("Referee called pivot at round %d: %s", round_num, score.rationale)

        return self.memory.get_all()

    def _run_round(self, round_num: int, question: str) -> RoundScore:
        """Execute a single round of the research loop."""

        # Phase 1: Builder proposes
        builder_output = self.builder.think(self.memory, round_num, question)
        self.memory.add(builder_output)
        self.on_object(builder_output, "builder_propose")

        # Phase 2: Skeptic challenges
        skeptic_context = (
            f"The Builder just produced:\n{self.builder._format_object(builder_output)}\n\n"
            f"Challenge this output. Reference target_id='{builder_output.id}'."
        )
        skeptic_output = self.skeptic.think(
            self.memory, round_num, question, extra_context=skeptic_context
        )
        self.memory.add(skeptic_output)
        self.on_object(skeptic_output, "skeptic_challenge")

        # Phase 3: Builder responds to Skeptic
        response_context = (
            f"The Skeptic challenged your work:\n{self.skeptic._format_object(skeptic_output)}\n\n"
            f"Respond by refining your idea, pivoting, or reframing."
        )
        builder_response = self.builder.think(
            self.memory, round_num, question, extra_context=response_context
        )
        self.memory.add(builder_response)
        self.on_object(builder_response, "builder_respond")

        # Phase 4: Historian synthesizes
        historian_output = self.historian.think(self.memory, round_num, question)
        self.memory.add(historian_output)
        self.on_object(historian_output, "historian_synthesize")

        # Phase 5: Referee scores
        referee_output = self.referee.think(self.memory, round_num, question)
        self.memory.add(referee_output)
        self.on_object(referee_output, "referee_score")

        if isinstance(referee_output, RoundScore):
            return referee_output

        return RoundScore(
            round_number=round_num,
            novelty=5.0,
            rigor=5.0,
            convergence=5.0,
            verdict="continue",
            rationale="Referee output was not a valid RoundScore; defaulting to continue.",
            created_by="referee",
        )

    def _detect_start_round(self) -> int:
        """If resuming a session, figure out which round to start from."""
        existing = self.memory.get_all()
        if not existing:
            return 1
        return max(o.round_number for o in existing) + 1

    def _over_budget(self) -> bool:
        spent = self.llm.budget_spent
        budget = self.config.budget_usd
        if spent >= budget:
            return True
        if spent >= budget * 0.8:
            self.on_budget_warning(spent, budget)
        return False
