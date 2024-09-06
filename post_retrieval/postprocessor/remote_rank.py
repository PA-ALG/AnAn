# -*- encoding: utf-8 -*-
"""
@File    : remote_rank.py
@Time    : 5/9/2024 14:38
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import logging
from typing import List, Optional

import requests
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field


class RemoteRankPostprocessor(BaseNodePostprocessor):
    """Remote-Rank-Serivce based Node processor."""
    top_n: float = Field(default=10)
    rank_type: Optional[str] = Field(description="content的来源字段，不给定则默认使用node默认excluded_llm_metadata_keys")
    url: str = Field(description="rank service url")

    @classmethod
    def class_name(cls) -> str:
        return "RemoteRankPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes.
        注意这里按照最小功能的原则，我们需要默认nodes已经无需额外的处理，例如去重。
        """

        pay_load = {"query": query_bundle.query_str, "items": []}
        item_list = []
        for node in nodes:
            item_id = node.node_id
            if self.rank_type and self.rank_type in node.metadata:
                item_text = node.metadata.get(self.rank_type, "")
            else:
                item_text = node.get_content()
            item_list.append({"id": item_id, "content": item_text})

        pay_load["items"] = item_list
        sorted_scores = {}
        try:
            response = requests.post(self.url, json=pay_load, timeout=(2, 10))
            response.raise_for_status()
            data = response.json()
            sorted_scores = {_id: score for _id, score in zip(data["data"]["ids"], data["data"]["scores"])}
            sorted_nodes = sorted(nodes,
                                  key=lambda x: sorted_scores.get(x.node_id, 0.0),
                                  reverse=True)
        except Exception as e:
            logging.warning(e)
            sorted_nodes = nodes

        return sorted_nodes


