import argparse
import json
import os
from typing import Any
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
    InvalidParamsError,
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError


class Messenger:
    async def talk_to_agent(self, text: str, base_url: str) -> str:
        async with httpx.AsyncClient(timeout=30) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            agent_card.url = base_url
            client = ClientFactory(
                ClientConfig(httpx_client=httpx_client, streaming=False)
            ).create(agent_card)
            outbound = Message(
                kind="message",
                role=Role.user,
                parts=[Part(root=TextPart(kind="text", text=text))],
                message_id=uuid4().hex,
                context_id=None,
            )

            last_event = None
            async for event in client.send_message(outbound):
                last_event = event

            if isinstance(last_event, Message):
                return "".join(
                    part.root.text
                    for part in last_event.parts
                    if isinstance(part.root, TextPart)
                )

            if last_event is None:
                raise RuntimeError("Participant returned no response")

            task, _update = last_event
            status_message = task.status.message
            if status_message is None:
                raise RuntimeError("Participant completed without a message")
            return "".join(
                part.root.text
                for part in status_message.parts
                if isinstance(part.root, TextPart)
            )


class Executor(AgentExecutor):
    def __init__(self) -> None:
        self._messenger = Messenger()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = context.get_user_input()
        try:
            request = json.loads(request_text)
        except json.JSONDecodeError as err:
            raise ServerError(
                error=InvalidParamsError(message=f"Invalid assessment request: {err}")
            ) from err

        competitor_url = (request.get("participants") or {}).get("competitor")
        if not competitor_url:
            raise ServerError(
                error=InvalidParamsError(message="Assessment requires a competitor participant")
            )

        message = context.message
        if message is None:
            raise ServerError(error=InvalidParamsError(message="Missing message"))

        task = context.current_task or new_task(message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Starting deterministic quick-submit assessment.",
                context_id=task.context_id,
                task_id=task.id,
            ),
        )

        green_token = os.environ["GREEN_TOKEN"]
        purple_reply = await self._messenger.talk_to_agent(
            f"green-token:{green_token}",
            competitor_url,
        )
        result: dict[str, Any] = {
            "winner": "competitor",
            "detail": {
                "green_token": green_token,
                "purple_reply": purple_reply,
                "config": request.get("config") or {},
            },
        }
        await updater.add_artifact(
            parts=[Part(root=TextPart(kind="text", text=json.dumps(result)))],
            name="Result",
        )
        await updater.complete(
            new_agent_text_message(
                "Assessment completed.",
                context_id=task.context_id,
                task_id=task.id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=InvalidParamsError(message="Cancel is unsupported"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock green agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    agent_card = AgentCard(
        name="mock-green",
        description="Deterministic green agent for AgentBeats quick-submit e2e.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
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
