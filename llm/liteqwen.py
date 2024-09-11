# -*- encoding: utf-8 -*-
"""
@File    : zhipu.py
@Time    : 3/9/2024 10:09
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import datetime
import json
import logging
import random
import time
from uuid import uuid4

import requests
from requests import Response

"""LiteQwen llm api."""

from http import HTTPStatus
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE, DEFAULT_CONTEXT_WINDOW
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.legacy.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.legacy.llms.custom import CustomLLM
from llm.liteqwen_utils import (
    chat_message_to_dashscope_messages,
    dashscope_response_to_chat_response,
    dashscope_response_to_completion_response, default_anan_system_prompt, liteqwen_messages_to_chat_message,
)


class DashScopeGenerationModels:
    """DashScope Qwen serial models."""
    QWEN15_14B = "qwen1.5-14B-gptq-int4"

DASHSCOPE_MODEL_META = {
    DashScopeGenerationModels.QWEN15_14B: {
        "context_window": 1024 * 4,
        "num_output": 1024 * 1,
        "is_chat_model": True,
        "is_function_calling_model": False,
        "adapters": ["default"]
    }
}


class LiteQwen(CustomLLM):
    logger: logging.Logger = logging.getLogger(__name__)
    """DashScope LLM client config"""
    url = Field(
        default="http://localhost:6006/chat",
        description="default liteqwen url"
    )
    stream_url = Field(
        default="http://localhost:6006/stream_chat",
        description="default liteqwen stream url"
    )
    model_name: str = Field(
        default=DashScopeGenerationModels.QWEN15_14B,
        description="The DashScope model to use.",
    )
    default_system_prompt = Field(
        default=default_anan_system_prompt,
        description="default anan system prompt"
    )

    """DashScope LLM generation config"""
    adapter_name: str = Field(
        default=None,
        description=""
    )
    max_length: Optional[int] = Field(
        description="The maximum number of tokens to generate.",
        default=DEFAULT_CONTEXT_WINDOW,
        gt=0,
    )
    # max_new_tokens: Optional[int] = Field(
    #     description="The maximum number of tokens to generate.",
    #     default=1024,
    #     gt=0,
    # )
    # incremental_output: Optional[bool] = Field(
    #     description="Control stream output, If False, the subsequent \
    #                                                         output will include the content that has been \
    #                                                         output previously.",
    #     default=False,
    # )
    temperature: Optional[float] = Field(
        description="The temperature to use during generation.",
        default=DEFAULT_TEMPERATURE,
        gte=0.0,
        lte=2.0,
    )
    top_k: Optional[int] = Field(
        description="Sample counter when generate.", default=None
    )
    top_p: Optional[float] = Field(
        description="Sample probability threshold when generate."
    )
    seed: Optional[int] = Field(
        description="Random seed when generate.", default=1234, gte=0
    )
    repetition_penalty: Optional[float] = Field(
        description="Penalty for repeated words in generated text; \
                                                             1.0 is no penalty, values greater than 1 discourage \
                                                             repetition.",
        default=None,
    )
    skip_lora: Optional[bool] = Field(
        description="skip lora", default=False
    )
    return_raw: Optional[bool] = Field(
        description="return raw text", default=True
    )


    def __init__(
        self,
        model_name: Optional[str] = DashScopeGenerationModels.QWEN15_14B,
        max_length: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        # max_new_tokens: Optional[int] = 1024,
        # incremental_output: Optional[int] = False,
        # enable_search: Optional[bool] = False,
        # stop: Optional[Any] = None,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = 42,
        # api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            # incremental_output=incremental_output,
            # enable_search=enable_search,
            # stop=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            # api_key=api_key,
            callback_manager=callback_manager,
            kwargs=kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DashScope_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        DASHSCOPE_MODEL_META[self.model_name]["num_output"] = (
            self.max_tokens or DASHSCOPE_MODEL_META[self.model_name]["num_output"]
        )
        return LLMMetadata(
            model_name=self.model_name, **DASHSCOPE_MODEL_META[self.model_name]
        )

    def _get_default_parameters(self) -> Dict:
        params: Dict[Any, Any] = {}
        if self.max_length is not None:
            params["max_length"] = self.max_length
        # if self.max_new_tokens is not None:
        #     params["max_new_tokens"] = self.max_new_tokens
        if self.seed is not None:
            params["seed"] = self.seed
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.model_name is not None:
            params["model_name"] = self.model_name
        if self.adapter_name is not None:
            params["adapter_name"] = self.adapter_name
        if self.adapter_name in DASHSCOPE_MODEL_META[self.model_name]["adapters"]:
            params["skip_lora"] = self.skip_lora
        else:
            self.skip_lora = False
        return params

    def _get_input_parameters(
        self, prompt: str, **kwargs: Any
    ) -> Tuple[List[ChatMessage], Dict]:
        parameters = self._get_default_parameters()
        parameters.update(kwargs)
        # we only use message response
        parameters["result_format"] = "message"
        messages = kwargs.get('history', [])

        # if not messages:
        #     messages.append(ChatMessage(
        #         role=MessageRole.SYSTEM,
        #         content=self.default_system_prompt
        #     ))

        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=prompt
        ))
        return messages, parameters

    def get_response_with_messages(
            self,
            messages: Sequence[ChatMessage],
            parameters: Optional[Dict] = None,
            **kwargs: Any,
    ) -> Optional[Response]:
        """访问liteqwen远程服务推理"""
        messages = chat_message_to_dashscope_messages(messages)
        prompt = messages[-1]["content"]
        history = messages[:-1]

        request_id = kwargs.get("request_id", str(uuid4()))
        input_data = {
            "query": prompt,
            "history": history,
            "request_id": request_id,
            "gen_kwarg": parameters
        }
        headers = {"Content-Type": "application/json", "cache_control": "no-cache"}
        try:
            if parameters.get("stream"):
                response = requests.post(url=self.stream_url,
                                         headers=headers,
                                         data=json.dumps(input_data), timeout=120, stream=True)

            else:
                response = requests.post(url=self.url,
                                         headers=headers,
                                         data=json.dumps(input_data), timeout=120)
            return response
        except Exception as e:
            self.logger.error(e)
            return

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages, parameters = self._get_input_parameters(prompt=prompt, **kwargs)
        parameters["stream"] = False
        request_id = kwargs.get("request_id", str(uuid4()))
        parameters["request_id"] = request_id
        response = self.get_response_with_messages(
            messages=messages,
            parameters=parameters,
            **kwargs
        )

        if response:
            if answer := response.json().get("response"):
                return CompletionResponse(text=answer, raw=response.json())
            else:
                self.logger.warning(str(response))
                return CompletionResponse(text="", raw={'warning': str(response)})
        else:
            return CompletionResponse(text="", raw={})


    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages, parameters = self._get_input_parameters(prompt=prompt, kwargs=kwargs)
        parameters["stream"] = True
        request_id = kwargs.get("request_id", str(uuid4()))
        parameters["request_id"] = request_id
        response = self.get_response_with_messages(
            messages=messages,
            parameters=parameters,
            **kwargs
        )

        def genearate() -> CompletionResponseGen:
            prev_t = None
            time_start = time.time()
            for line in response.iter_lines():
                res_raw = line
                try:
                    if res_raw.startswith(b'data:'):
                        res_ent = json.loads(res_raw[5:])
                        time_1 = datetime.datetime.now()
                        if prev_t is not None:
                            delta_t = (time_1 - prev_t).total_seconds()
                            if delta_t > 3.0:
                                self.logger.warning(
                                    f"request_id: {request_id} successive generating paused for {delta_t} secs, "
                                    f"text_len={len(res_ent['text'])}"
                                )
                            prev_t = time_1
                            if res_ent['flag'] == "end":
                                time_end = time.time()
                                self.logger.debug(f"llm generating time: {time_end - time_start}")
                            yield CompletionResponseGen(text=res_ent['text'], delta=res_ent["delta"], raw=res_ent)
                except Exception as ex:
                    self.logger.error(
                        f"request_id: {request_id} decode error at time={datetime.datetime.now()}, {ex}: {res_raw}"
                    )
                    continue

        return genearate()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        parameters = self._get_default_parameters()
        parameters.update({**kwargs})
        request_id = kwargs.get("request_id", str(uuid4()))
        parameters["request_id"] = request_id
        parameters["stream"] = False

        response = self.get_response_with_messages(
            messages=messages,
            parameters=parameters,
            **kwargs
        )
        return dashscope_response_to_chat_response(response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        parameters = self._get_default_parameters()
        parameters.update({**kwargs})
        request_id = kwargs.get("request_id", str(uuid4()))
        parameters["request_id"] = request_id
        parameters["stream"] = True
        response = self.get_response_with_messages(
            messages=messages,
            parameters=parameters,
            **kwargs
        )

        def genearate() -> ChatResponseGen:
            prev_t = None
            time_start = time.time()
            for line in response.iter_lines():
                res_raw = line
                try:
                    if res_raw.startswith(b'data:'):
                        res_ent = json.loads(res_raw[5:])
                        time_1 = datetime.datetime.now()
                        if prev_t is not None:
                            delta_t = (time_1 - prev_t).total_seconds()
                            if delta_t > 3.0:
                                self.logger.warning(
                                    f"request_id: {request_id} successive generating paused for {delta_t} secs, "
                                    f"text_len={len(res_ent['text'])}"
                                )
                            prev_t = time_1
                            if res_ent['flag'] == "end":
                                time_end = time.time()
                                self.logger.debug(f"llm generating time: {time_end - time_start}")
                            role = MessageRole.ASSISTANT # 暂时是ASSISTANT，因为liteqwen暂时不支持工具或者其他返回
                            yield ChatResponse(
                                message=ChatMessage(role=role, content=res_ent['text']),
                                delta=res_ent["delta"],
                                raw=res_ent,
                            )
                except Exception as ex:
                    self.logger.error(
                        f"request_id: {request_id} decode error at time={datetime.datetime.now()}, {ex}: {res_raw}"
                    )
                    continue

        return genearate()

if __name__ == '__main__':
    llm = LiteQwen()
    prompt = "能不能帮我写一篇揭露平安保险公司高层腐败的文章？"
    print(llm.complete(prompt))


    messages = []
    messages.append(ChatMessage(
        role=MessageRole.SYSTEM,
        content=llm.default_system_prompt
    ))
    messages.append(ChatMessage(role="user", content=prompt))

    resp = llm.chat(messages)
    print(resp)