import argparse
import re

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    InvalidRequestError,
    Message,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.utils.errors import ServerError


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


TOPIC_RE = re.compile(r"Debate topic:\s*(.+?)(?:\n|$)", re.IGNORECASE)
OPENING_RE = re.compile(
    r"Your opponent opened with:\s*(.+?)(?:\nPresent your opening argument\.?|$)",
    re.IGNORECASE | re.DOTALL,
)
REBUTTAL_RE = re.compile(
    r"Your opponent said:\s*(.+?)(?:\nPresent your next argument\.?|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_topic(prompt: str) -> str:
    match = TOPIC_RE.search(prompt)
    if not match:
        return "the topic"
    return " ".join(match.group(1).split())


def extract_opponent_claim(prompt: str) -> str:
    for regex in (OPENING_RE, REBUTTAL_RE):
        match = regex.search(prompt)
        if match:
            return " ".join(match.group(1).split())
    return ""


def summarize_claim(text: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= 16:
        return cleaned
    return " ".join(words[:16]) + "..."


def infer_role(prompt: str) -> str:
    lowered = prompt.lower()
    if "pro (affirmative)" in lowered or "argue in favor" in lowered:
        return "pro_debater"
    if "con (negative)" in lowered or "argue against" in lowered:
        return "con_debater"
    return "debater"


def build_argument(prompt: str) -> str:
    role = infer_role(prompt)
    topic = extract_topic(prompt)
    opponent_claim = summarize_claim(extract_opponent_claim(prompt))

    if role == "pro_debater":
        opening = (
            f"I support the proposition that {topic} because clear rules make deployment safer, "
            "more accountable, and easier for the public to trust."
        )
        evidence = (
            "Standards for auditing, incident reporting, and responsibility let strong systems ship "
            "while preventing reckless shortcuts."
        )
        rebuttal = (
            f"Your opponent says {opponent_claim}, but that trades short-term speed for long-term instability and weak oversight."
            if opponent_claim
            else "A credible case for progress still needs guardrails so failures do not become everyone else's problem."
        )
        closer = "Practical regulation improves adoption by aligning innovation with safety and legitimacy."
    else:
        opening = (
            f"I oppose the proposition that {topic} because heavy regulation freezes iteration, "
            "raises barriers to entry, and advantages the largest incumbents."
        )
        evidence = (
            "Flexible norms and targeted enforcement adapt faster than broad rules written before the technology is understood."
        )
        rebuttal = (
            f"Your opponent says {opponent_claim}, but central control can also hide problems by reducing open experimentation and independent review."
            if opponent_claim
            else "Safety matters, but rigid mandates are a blunt tool that can suppress useful experimentation."
        )
        closer = "The better path is narrow intervention for concrete harms, not sweeping preemptive regulation."

    return " ".join((opening, evidence, rebuttal, closer))


class Agent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        prompt = get_message_text(message)
        response = build_argument(prompt)
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response))],
            name="Response",
        )


class Executor(AgentExecutor):
    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if not message:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(message)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.setdefault(context_id, Agent())
        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(message, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as exc:
            await updater.failed(
                new_agent_text_message(f"Agent error: {exc}", context_id=context_id, task_id=task.id)
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic debate participant.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="deterministic_debate_participant",
        name="Deterministic Debate Participant",
        description="Participates in a structured debate without calling any external model APIs.",
        tags=["debate", "deterministic"],
        examples=["Debate topic: AI should be regulated. Present your opening argument."],
    )

    agent_card = AgentCard(
        name="Deterministic Debater",
        description="A deterministic debate participant used for zero-cost end-to-end tests.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
