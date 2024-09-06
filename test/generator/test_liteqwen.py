from typing import Any

from llama_index.legacy.core.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.legacy.llms.custom import CustomLLM

from llm.liteqwen import LiteQwen


def test_complete() -> None:
    llm = LiteQwen()
    prompt = "你是谁？"
    message = ChatMessage(role="user", content=prompt)
    llm.complete(prompt)
    llm.chat([message])


# def test_streaming() -> None:
#     llm = TestLLM()
#
#     prompt = "test prompt"
#     message = ChatMessage(role="user", content="test message")
#
#     llm.stream_complete(prompt)
#     llm.stream_chat([message])
# if __name__ == '__main__':
#     test_complete()