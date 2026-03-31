import argparse
import json
import re
from typing import Any, Literal
from uuid import uuid4

import httpx
import uvicorn
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    InvalidRequestError,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from pydantic import BaseModel, HttpUrl, ValidationError


DEFAULT_TIMEOUT = 300
TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class DebaterScore(BaseModel):
    emotional_appeal: float
    argument_clarity: float
    argument_arrangement: float
    relevance_to_topic: float
    total_score: float


class DebateEval(BaseModel):
    pro_debater: DebaterScore
    con_debater: DebaterScore
    winner: Literal["pro_debater", "con_debater"]
    reason: str


def round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def create_message(*, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, sort_keys=True))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


async def send_message(message: str, base_url: str, context_id: str | None = None) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        outbound_msg = create_message(text=message, context_id=context_id)

        last_event: Any = None
        outputs: dict[str, Any] = {"response": "", "context_id": None}

        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] = merge_parts(msg.parts)
            case (task, TaskStatusUpdateEvent() as status_event):
                outputs["context_id"] = task.context_id
                outputs["status"] = status_event.status.state.value
                status_msg = status_event.status.message
                if status_msg:
                    outputs["response"] = merge_parts(status_msg.parts)
                if task.artifacts:
                    outputs["response"] = "\n".join(
                        part
                        for part in [outputs["response"], *[merge_parts(a.parts) for a in task.artifacts]]
                        if part
                    ).strip()
            case (task, TaskArtifactUpdateEvent() as artifact_event):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                outputs["response"] = merge_parts(artifact_event.artifact.parts)
            case (task, _):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                status_msg = task.status.message
                if status_msg:
                    outputs["response"] = merge_parts(status_msg.parts)
                if task.artifacts:
                    outputs["response"] = "\n".join(
                        part
                        for part in [outputs["response"], *[merge_parts(a.parts) for a in task.artifacts]]
                        if part
                    ).strip()
            case _:
                pass

        return outputs


class Messenger:
    def __init__(self) -> None:
        self._context_ids: dict[str, str | None] = {}

    async def talk_to_agent(self, message: str, url: str) -> str:
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=self._context_ids.get(url),
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id")
        return str(outputs.get("response", "")).strip()

    def reset(self) -> None:
        self._context_ids = {}


def extract_topic_terms(topic: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", topic.lower()) if len(word) > 2}


def count_turns_with_topic(turns: list[str], topic_terms: set[str]) -> int:
    if not topic_terms:
        return len(turns)
    total = 0
    for turn in turns:
        words = set(re.findall(r"[a-z0-9]+", turn.lower()))
        if words & topic_terms:
            total += 1
    return total


def count_turns_with_markers(turns: list[str], markers: tuple[str, ...]) -> int:
    total = 0
    for turn in turns:
        lowered = turn.lower()
        if any(marker in lowered for marker in markers):
            total += 1
    return total


def build_score(turns: list[str], topic: str, role: str) -> DebaterScore:
    topic_terms = extract_topic_terms(topic)
    turn_count = max(len(turns), 1)
    avg_words = sum(len(turn.split()) for turn in turns) / turn_count
    topical_turns = count_turns_with_topic(turns, topic_terms)
    structured_turns = count_turns_with_markers(
        turns,
        ("because", "therefore", "first", "second", "finally"),
    )
    rebuttal_turns = count_turns_with_markers(
        turns,
        ("your opponent says", "that trades", "that is unconvincing", "central control", "guardrails"),
    )
    emotional_turns = count_turns_with_markers(
        turns,
        ("safety", "trust", "harm", "protect", "freedom", "innovation", "accountable"),
    )

    emotional_appeal = round_score(0.5 + 0.1 * emotional_turns / turn_count)
    argument_clarity = round_score(0.55 + min(avg_words / 180.0, 0.2))
    argument_arrangement = round_score(0.5 + 0.2 * structured_turns / turn_count + 0.05 * rebuttal_turns / turn_count)
    relevance_to_topic = round_score(0.55 + 0.25 * topical_turns / turn_count)
    total_score = round_score(
        (emotional_appeal + argument_clarity + argument_arrangement + relevance_to_topic) / 4.0
    )

    if role == "pro_debater":
        total_score = round_score(total_score + 0.015)
    return DebaterScore(
        emotional_appeal=emotional_appeal,
        argument_clarity=argument_clarity,
        argument_arrangement=argument_arrangement,
        relevance_to_topic=relevance_to_topic,
        total_score=total_score,
    )


def summarize_winner(topic: str, winner: str, winner_score: DebaterScore, loser_score: DebaterScore) -> str:
    margin = round(winner_score.total_score - loser_score.total_score, 3)
    role_label = "Pro" if winner == "pro_debater" else "Con"
    return (
        f"{role_label} wins on '{topic}' with stronger structure and topic coverage. "
        f"Score margin: {margin:.3f}."
    )


class Agent:
    required_roles: list[str] = ["pro_debater", "con_debater"]
    required_config_keys: list[str] = ["topic", "num_rounds"]

    def __init__(self) -> None:
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {sorted(missing_roles)}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {sorted(missing_config_keys)}"

        try:
            int(request.config["num_rounds"])
        except Exception as exc:
            return False, f"Can't parse num_rounds: {exc}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, validation_message = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(validation_message))
                return
        except ValidationError as exc:
            await updater.reject(new_agent_text_message(f"Invalid request: {exc}"))
            return

        topic = str(request.config["topic"])
        num_rounds = int(request.config["num_rounds"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting deterministic debate assessment.\n{request.model_dump_json()}"
            ),
        )

        try:
            debate = await self.orchestrate_debate(
                participants=request.participants,
                topic=topic,
                num_rounds=num_rounds,
                updater=updater,
            )
            result = self.judge_debate(topic, debate)
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=result.reason)),
                    Part(root=DataPart(data=result.model_dump())),
                ],
                name="Result",
            )
        finally:
            self.messenger.reset()

    async def orchestrate_debate(
        self,
        *,
        participants: dict[str, HttpUrl],
        topic: str,
        num_rounds: int,
        updater: TaskUpdater,
    ) -> dict[str, list[str]]:
        debate = {"pro_debater": [], "con_debater": []}

        async def turn(role: str, prompt: str) -> str:
            response = await self.messenger.talk_to_agent(prompt, str(participants[role]))
            debate[role].append(response)
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            return response

        pro_instructions = "You are the Pro (Affirmative) side. Argue in favor of the proposition."
        con_instructions = "You are the Con (Negative) side. Argue against the proposition."

        response = await turn(
            "pro_debater",
            f"{pro_instructions}\nDebate topic: {topic}\nPresent your opening argument.",
        )
        response = await turn(
            "con_debater",
            f"{con_instructions}\nDebate topic: {topic}\nYour opponent opened with: {response}\nPresent your opening argument.",
        )

        for _ in range(num_rounds - 1):
            response = await turn(
                "pro_debater",
                f"{pro_instructions}\nDebate topic: {topic}\nYour opponent said: {response}\nPresent your next argument.",
            )
            response = await turn(
                "con_debater",
                f"{con_instructions}\nDebate topic: {topic}\nYour opponent said: {response}\nPresent your next argument.",
            )

        return debate

    def judge_debate(self, topic: str, debate: dict[str, list[str]]) -> DebateEval:
        pro_score = build_score(debate["pro_debater"], topic, "pro_debater")
        con_score = build_score(debate["con_debater"], topic, "con_debater")

        if pro_score.total_score >= con_score.total_score:
            winner = "pro_debater"
            reason = summarize_winner(topic, winner, pro_score, con_score)
        else:
            winner = "con_debater"
            reason = summarize_winner(topic, winner, con_score, pro_score)

        return DebateEval(
            pro_debater=pro_score,
            con_debater=con_score,
            winner=winner,
            reason=reason,
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
    parser = argparse.ArgumentParser(description="Run the deterministic debate judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="moderate_and_judge_deterministic_debate",
        name="Orchestrates and judges deterministic debate",
        description="Runs a structured debate between two agents without using external model APIs.",
        tags=["debate", "deterministic"],
        examples=[
            (
                '{"participants":{"pro_debater":"https://pro.example.com","con_debater":"https://con.example.com"},'
                '"config":{"topic":"AI should be regulated.","num_rounds":2}}'
            )
        ],
    )

    agent_card = AgentCard(
        name="Deterministic Debate Judge",
        description="A zero-cost debate judge used to validate realistic end-to-end quick-submit runs.",
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
