# -*- encoding: utf-8 -*-
"""
@File    : utils.py
@Time    : 5/9/2024 15:02
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from typing import List

from llama_index.core.schema import NodeWithScore


def nodes_to_text_list(nodes: List[NodeWithScore], text_from: str = "content") -> List[str]:
    """Convert nodes to text list."""
    new_nodes = []
    for node in nodes:
        extracted_text = NodeWithScore