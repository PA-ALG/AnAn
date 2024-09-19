"""You Retriever."""

import logging
import os
import uuid
import warnings
from typing import Any, Dict, List, Literal, Optional

import requests

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)

class AnAnRetriever(BaseRetriever):
    """
    Retriever for AnAn Retrieval API.

    Args:
        api_key: you.com API key, if `YDC_API_KEY` is not set in the environment
        endpoint: you.com endpoints
        num_web_results: The max number of web results to return, must be under 20
        safesearch: Safesearch settings, one of "off", "moderate", "strict", defaults to moderate
        country: Country code, ex: 'US' for United States, see API reference for more info
        search_lang: (News API) Language codes, ex: 'en' for English, see API reference for more info
        ui_lang: (News API) User interface language for the response, ex: 'en' for English, see API reference for more info
        spellcheck: (News API) Whether to spell check query or not, defaults to True
    """

    def __init__(
        self,
        api_url: str,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self.api_url = api_url
        super().__init__(callback_manager)

    def _generate_params(self, query: str, record_id: Optional[str] = None) -> Dict[str, Any]:
        params = {
            "query": query,
            "record_id": record_id if record_id else str(uuid.uuid4())
        }
        return params

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        params = self._generate_params(query_bundle.query_str)

        try:
            response = requests.post(self.api_url, json=params, timeout=(1, 10))
            response.raise_for_status()
            results = response.json()

            if not isinstance(results, list):
                logger.warning(f"Unexpected retrieval response format: {results}")
                results = []

            new_nodes: List[TextNode] = []

            for i, doc in enumerate(results):
                results[i]["index"] = i
                new_node = TextNode(text=doc["keypoint"], metadata=doc.metadata)
                new_nodes.append(new_node)

        except Exception as e:
            new_nodes = [TextNode(text="")]
            logger.error(f"Exception retrieval: {e}")

        return [NodeWithScore(node=node, score=1.0) for node in new_nodes]
