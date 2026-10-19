from typing import List

import pytest

from llm.base import LLMError, Message
from llm.fake import FakeLLM


def test_replies_in_order() -> None:
    llm: FakeLLM = FakeLLM(["a", "b"])
    assert llm.send([]) == "a" and llm.send([]) == "b"


def test_records_each_conversation_as_sent() -> None:
    llm: FakeLLM = FakeLLM(["a", "b"])
    messages: List[Message] = [{"role": "user", "content": "1"}]
    llm.send(messages)
    messages.append({"role": "user", "content": "2"})  # the caller keeps appending
    llm.send(messages)
    assert [len(conv) for conv in llm.received] == [1, 2]


def test_no_reply_left_is_an_llm_error() -> None:
    with pytest.raises(LLMError):
        FakeLLM([]).send([])


def test_caller_list_is_not_consumed() -> None:
    replies: List[str] = ["a"]
    FakeLLM(replies).send([])
    assert replies == ["a"]
