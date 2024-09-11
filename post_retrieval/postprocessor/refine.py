# -*- encoding: utf-8 -*-
"""
@File    : refine.py
@Time    : 5/9/2024 14:38
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

from typing import List, Optional, Callable

from llama_index.core import BasePromptTemplate, Settings
from llama_index.core.bridge.pydantic import Field
from llama_index.core.indices.utils import default_format_node_batch_fn
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import SerializeAsAny, PrivateAttr

from prompts.chat_prompt_zh import CHAT_LLM_REFINE_PROMPT


class LLMRefineContentPostProcessor(BaseNodePostprocessor):

    top_n: int = Field(description="Top N nodes to return.")
    choice_select_prompt: SerializeAsAny[BasePromptTemplate] = Field(
        description="Choice select prompt."
    )
    choice_batch_size: int = Field(description="Batch size for choice select.")
    llm: LLM = Field(description="The LLM to rerank with.")

    _format_node_batch_fn: Callable = PrivateAttr()

    def __init__(
            self,
            llm: Optional[LLM] = None,
            choice_select_prompt: Optional[BasePromptTemplate] = None,
            choice_batch_size: int = 10,
            format_node_batch_fn: Optional[Callable] = None,
            top_n: int = 10,
    ) -> None:
        choice_select_prompt = choice_select_prompt or CHAT_LLM_REFINE_PROMPT

        llm = llm or Settings.llm

        super().__init__(
            llm=llm,
            choice_select_prompt=choice_select_prompt,
            choice_batch_size=choice_batch_size,
            top_n=top_n,
        )
        self._format_node_batch_fn = (
                format_node_batch_fn or default_format_node_batch_fn
        )


    @classmethod
    def class_name(cls) -> str:
        return "LLMRefineContentPostProcessor"

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"choice_select_prompt": self.choice_select_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "choice_select_prompt" in prompts:
            self.choice_select_prompt = prompts["choice_select_prompt"]

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        new_nodes = []
        for n in nodes:
            query_str = query_bundle.query_str
            context_str = self._format_node_batch_fn([n])
            raw_response = self.llm.predict(
                self.choice_select_prompt,
                context_str=context_str,
                query_str=query_str,
            )
            if "无内容" not in raw_response:
                n.node.set_content(raw_response)
                new_nodes.append(n)

        return new_nodes



