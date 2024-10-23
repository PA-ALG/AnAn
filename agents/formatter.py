# -*- encoding: utf-8 -*-
"""
@File    : formatter.py
@Time    : 27/9/2024 11:27
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import logging
from abc import abstractmethod
from typing import List, Optional, Sequence

from llama_index.core.agent.react.formatter import BaseAgentChatFormatter, get_react_tool_descriptions
from llama_index.core.agent.react.prompts import (
    CONTEXT_REACT_CHAT_SYSTEM_HEADER,
    REACT_CHAT_SYSTEM_HEADER,
)
from llama_index.core.agent.react.types import (
    BaseReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict
from llama_index.core.tools import BaseTool

from agents.react_prompt import TEAM_LEADER_REACT_CHAT_SYSTEM_HEADER

logger = logging.getLogger(__name__)

class BaseTeamLeaderReActChatFormatter(BaseModel):
    """ReAct chat formatter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    system_header: str = TEAM_LEADER_REACT_CHAT_SYSTEM_HEADER  # default
    context: str = ""  # not needed w/ default

    def format(
        self,
        tools: Sequence[BaseTool],
        perspective: Optional[str],
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        """Format chat history into list of ChatMessage."""
        current_reasoning = current_reasoning or []

        format_args = {
            "tool_desc": "\n".join(get_react_tool_descriptions(tools)),
            "tool_names": ", ".join([tool.metadata.get_name() for tool in tools]),
            "perspective": perspective
        }
        if self.context:
            format_args["context"] = self.context

        fmt_sys_header = self.system_header.format(**format_args)

        # format reasoning history as alternating user and assistant messages
        # where the assistant messages are thoughts and actions and the user
        # messages are observations
        reasoning_history = []
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ChatMessage(
                    role=MessageRole.USER,
                    content=reasoning_step.get_content(),
                )
            else:
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=reasoning_step.get_content(),
                )
            reasoning_history.append(message)

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *chat_history,
            *reasoning_history,
        ]

    @classmethod
    def from_defaults(
        cls,
        system_header: Optional[str] = None,
        context: Optional[str] = None,
    ) -> "BaseTeamLeaderReActChatFormatter":
        """Create ReActChatFormatter from defaults."""
        if not system_header:
            system_header = (
                TEAM_LEADER_REACT_CHAT_SYSTEM_HEADER
                if not context
                else CONTEXT_REACT_CHAT_SYSTEM_HEADER
            )

        return BaseTeamLeaderReActChatFormatter(
            system_header=system_header,
            context=context or "",
        )

    @classmethod
    def from_context(cls, context: str) -> "BaseTeamLeaderReActChatFormatter":
        """Create ReActChatFormatter from context.

        NOTE: deprecated

        """
        logger.warning(
            "ReActChatFormatter.from_context is deprecated, please use `from_defaults` instead."
        )
        return BaseTeamLeaderReActChatFormatter.from_defaults(
            system_header=CONTEXT_REACT_CHAT_SYSTEM_HEADER, context=context
        )
