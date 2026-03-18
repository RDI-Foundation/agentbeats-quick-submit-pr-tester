import argparse
import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, InvalidRequestError, Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.utils.errors import ServerError


class Executor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None:
            raise ServerError(error=InvalidRequestError(message="Missing message"))

        task = context.current_task or new_task(message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        prompt = get_message_text(message)
        reply = f"mock-purple:{os.environ['API_KEY']}:{prompt}"
        await updater.complete(
            new_agent_text_message(reply, context_id=task.context_id, task_id=task.id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=InvalidRequestError(message="Cancel is unsupported"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock purple agent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    agent_card = AgentCard(
        name="mock-purple",
        description="Deterministic participant for AgentBeats quick-submit e2e.",
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
