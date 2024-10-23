# -*- encoding: utf-8 -*-
"""
@File    : test_react_agent.py
@Time    : 26/9/2024 14:02
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from llama_index.agent.dashscope import DashScopeAgent

agent = DashScopeAgent(
    app_id="sk-6a103912161c41389d3ca3c911ecc89c",  # The id of app that you created
    chat_session=True,  # Enable chat session which will auto save and pass chat                               history to LLM.
    verbose=True,  # If need to print more details
)

from llama_index.core.agent.react.base import ReActAgent
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_TURBO)
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
