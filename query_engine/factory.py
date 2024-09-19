# -*- encoding: utf-8 -*-
"""
@File    : factory.py
@Time    : 9/9/2024 14:35
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from llama_index.core.indices.base import BaseIndex


def build_rag_query_engine(index: BaseIndex,
                           pre_retrival: BaseQueryTransform,
                           retriever: BaseRetriever,
                           post_retrieval: List[BaseNodePostprocessor],
                           response_synthesizer: BaseSynthesizer):
    # Build a tree index over the set of candidate nodes, with a summary prompt seeded with the query. with LLM reranker
    rag_query_engine = index.as_query_engine(retriever=retriever,
                                             response_synthesizer=response_synthesizer,
                                             node_postprocessors=post_retrieval,
                                             response_synthesizer=get_response_synthesizer(
                                                 response_mode=response_mode),
                                             )
    return rag_query_engine