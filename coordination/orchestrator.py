import asyncio
import logging
import random
from typing import List, Tuple

from models import AgentGeneration, ConsensusResult
from pipeline.aggregation import aggregate
from pipeline.filter import filter_agents

logger = logging.getLogger(__name__)


class Orchestrator:
    """Async manager handling agent timeouts, routing, and liveness fallback.

    Args:
        f: Number of tolerated faults. Liveness requires >= 2f+1 agents to pass
           Module 1; otherwise the round is flagged low-confidence.
        agent_timeout: Seconds to wait for each agent before dropping it.
    """

    def __init__(self, f: int, agent_timeout: float = 2.0) -> None:
        self.f = f
        self.agent_timeout = agent_timeout

    async def run(self, generations: List[AgentGeneration]) -> ConsensusResult:
        """Execute one consensus round over a cached list of AgentGenerations.

        Pipeline:
          1. Simulate broadcast + collect responses (with timeouts).
          2. Apply Module 1 reliability filter.
          3. Liveness check: fall back to all responders if < 2f+1 admitted.
          4. Apply Module 2 semantic aggregation.
        """
        responded: List[AgentGeneration] = await self._simulate_broadcast(generations)
        logger.info(
            "Broadcast complete: %d/%d agents responded.", len(responded), len(generations)
        )

        admitted: List[AgentGeneration] = await filter_agents(responded)
        logger.info("Module 1 filter admitted %d agents.", len(admitted))

        admitted, is_low_confidence = self._liveness_check(admitted, responded)
        if is_low_confidence:
            logger.warning(
                "Liveness fallback triggered: only %d admitted (threshold 2f+1=%d). "
                "Admitting all %d responding agents.",
                len(admitted),
                2 * self.f + 1,
                len(responded),
            )

        if not admitted:
            logger.error(
                "All agents failed to respond (f=%d). Cannot produce a consensus answer.",
                self.f,
            )
            return ConsensusResult(
                final_answer="",
                admitted_agents=[],
                is_low_confidence=True,
            )

        final_answer: str = await aggregate(admitted)

        return ConsensusResult(
            final_answer=final_answer,
            admitted_agents=[g.agent_id for g in admitted],
            is_low_confidence=is_low_confidence,
        )

    async def _simulate_broadcast(
        self, generations: List[AgentGeneration]
    ) -> List[AgentGeneration]:
        """Concurrently 'send' a task to every agent and collect responses.

        F1_crash agents are assigned a sleep duration that exceeds agent_timeout,
        so asyncio.wait_for will cancel them — faithfully simulating a crash fault.
        All other exceptions are logged and the offending agent is dropped.
        """
        tasks = [
            asyncio.wait_for(
                self._agent_task(gen),
                timeout=self.agent_timeout,
            )
            for gen in generations
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responded: List[AgentGeneration] = []
        for gen, result in zip(generations, results):
            if isinstance(result, BaseException):
                logger.debug(
                    "Agent %s dropped: %s(%s)",
                    gen.agent_id,
                    type(result).__name__,
                    result,
                )
            else:
                responded.append(result)
        return responded

    async def _agent_task(self, gen: AgentGeneration) -> AgentGeneration:
        """Simulate one agent's response latency.

        F1_crash agents sleep past the timeout so they are always cancelled.
        All other agents (including Byzantine/Drifter) respond within the window;
        fault-content filtering is Module 1's responsibility.
        """
        if gen.fault_type == "F1_crash":
            delay = random.uniform(self.agent_timeout + 0.1, self.agent_timeout + 1.0)
        else:
            delay = random.uniform(0.05, min(0.5, self.agent_timeout * 0.9))
        await asyncio.sleep(delay)
        return gen

    def _liveness_check(
        self,
        admitted: List[AgentGeneration],
        available: List[AgentGeneration],
    ) -> Tuple[List[AgentGeneration], bool]:
        """Return (admitted_pool, is_low_confidence).

        If fewer than 2f+1 agents passed Module 1, fall back to the full pool of
        responding agents and mark the round as low-confidence.
        """
        if len(admitted) >= 2 * self.f + 1:
            return admitted, False
        return available, True
