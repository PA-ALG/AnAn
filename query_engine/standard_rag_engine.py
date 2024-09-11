# -*- encoding: utf-8 -*-
"""
@File    : standard_rag_engine.py
@Time    : 10/9/2024 10:58
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

from typing import List, Optional, Sequence, Any

import instrument
from llama_index.core import Settings
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
)
import llama_index.core.instrumentation as instrument

from response_synthesizers.factory import get_response_synthesizer

dispatcher = instrument.get_dispatcher(__name__)

class StandardModularRAGQueryEngine(BaseQueryEngine):
    """Transform query engine.

    Applies a query transform to a query bundle before passing
        it to a query engine.

    Args:
        query_engine (BaseQueryEngine): A query engine object.
        query_transform (BaseQueryTransform): A query transform object.
        transform_metadata (Optional[dict]): metadata to pass to the
            query transform.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        pre_retrival: BaseQueryTransform,
        retriever: BaseRetriever,
        post_retrival: List[BaseNodePostprocessor],
        response_synthesizer: Optional[BaseSynthesizer] = None,
        transform_metadata: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._pre_retrival = pre_retrival
        self._retriever = retriever
        self._post_retrival = post_retrival
        self.response_synthesizer = response_synthesizer
        self.transform_metadata = transform_metadata

        super().__init__(callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "pre_retrieval": self._pre_retrival,
            "post_retrieval": self._post_retrival
        }

    @classmethod
    def from_args(
            cls,
            retriever: BaseRetriever,
            llm: Optional[LLM] = None,
            query_transform: Optional[BaseQueryTransform] = None,
            response_synthesizer: Optional[BaseSynthesizer] = None,
            node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
            # response synthesizer args
            response_mode: ResponseMode = ResponseMode.COMPACT,
            text_qa_template: Optional[BasePromptTemplate] = None,
            refine_template: Optional[BasePromptTemplate] = None,
            summary_template: Optional[BasePromptTemplate] = None,
            simple_template: Optional[BasePromptTemplate] = None,
            output_cls: Optional[BaseModel] = None,
            use_async: bool = False,
            streaming: bool = False,
            **kwargs: Any,
    ) -> "StandardModularRAGQueryEngine":
        """Initialize a RetrieverQueryEngine object.".

        Args:
            retriever (BaseRetriever): A retriever object.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            verbose (bool): Whether to print out debug info.
            response_mode (ResponseMode): A ResponseMode object.
            text_qa_template (Optional[BasePromptTemplate]): A BasePromptTemplate
                object.
            refine_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.
            simple_template (Optional[BasePromptTemplate]): A BasePromptTemplate object.

            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.
            optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
                object.

        """
        llm = llm or Settings.llm

        response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            summary_template=summary_template,
            simple_template=simple_template,
            response_mode=response_mode,
            output_cls=output_cls,
            use_async=use_async,
            streaming=streaming,
        )

        callback_manager = Settings.callback_manager

        return cls(
            pre_retrival=query_transform,
            retriever=retriever,
            post_retrival=node_postprocessors,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager
        )

    def _apply_node_postprocessors(
            self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._post_retrival:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def with_retriever(self, retriever: BaseRetriever) -> "StandardModularRAGQueryEngine":
        return StandardModularRAGQueryEngine(
            retriever=retriever,
            response_synthesizer=self.response_synthesizer,
            callback_manager=self.callback_manager,
            post_retrival=self._post_retrival,
        )

    def synthesize(
            self,
            query_bundle: QueryBundle,
            nodes: List[NodeWithScore],
            additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return self.response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    async def asynthesize(
            self,
            query_bundle: QueryBundle,
            nodes: List[NodeWithScore],
            additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return await self.response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = self.retrieve(query_bundle)
            response = self.response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @dispatcher.span
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = await self.aretrieve(query_bundle)

            response = await self.response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever
