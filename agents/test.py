# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 26/9/2024 10:23
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
from typing import List, Callable, Any

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.react.base import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool, QueryPlanTool

from agents.react_prompt import CONTEXT_TEAM_LEADER_REACT_CHAT_SYSTEM_HEADER, daily_inspection_perspectives
from agents.team_leader.base import TeamLeaderAgent
from response_synthesizers.factory import get_response_synthesizer


# llm = OpenAI(temperature=0.1,
#              model="qwen-plus",
#              api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")

def init_chat_llm():
    dashscope_llm = DashScope(model_name="qwen-max", api_key="sk-6a103912161c41389d3ca3c911ecc89c")
    return dashscope_llm


def running_data_tool(month="202409"):
    """查询经营指标表现"""
    result = (
        "管日常：13月保费继续率=87%，达成一星；品质扣分5分人力占比=0%，达成五星；参会率=60%，达成三星。整体管日常一星（目标二星）。"
    )
    return result


def team_member_tool(month="202409"):
    """查询代理人日常表现"""
    result = "①13月保费继续率 主任1：76%；主任2：82%；主任3：65%；主任4：93%；代理人1：65%；代理人2：74%；代理人3：76%；代理人4：88%；代理人5：49%；代理人6：69%；代理人7：72%；代理人8：92%；代理人9：92%；代理人10：92%；代理人11：92%；代理人12：92%；代理人13：92%；代理人14：92%；代理人15：92%；代理人16：92%；代理人17：92%；代理人18：92%；代理人19：92%；代理人20：92%；代理人21：92%；代理人22：92%；代理人23：93%；代理人24：93%；代理人25：93%；代理人26：93%；代理人27：93%；代理人28：93%；代理人29：93%；代理人30：93%；代理人31：93%；代理人32：93% ②02考核通过率 1年内新人有5人，其中代理人1的季度FYC=0、代理人2的季度FYC=0、代理人3的季度FYC=2000、代理人的4季度FYC=4000，代理人5的季度FYC=5000  ③主管辅导率 目标审核率（审目标）：77.3%；追计划活动率（追计划）：85.3%；辅导活动率（做辅导）：57.1% "
    return result


def test_react_agent():
    from test import running_data_tool, team_member_tool
    running_data_tool = FunctionTool.from_defaults(fn=running_data_tool)
    team_member_tool = FunctionTool.from_defaults(fn=team_member_tool)

    llm = init_chat_llm()
    # agent with fixed workflow

    daily_team_leader_agent = TeamLeaderAgent.from_tools(tools=[running_data_tool, team_member_tool],
                                                         context=daily_inspection_perspectives,
                                                         llm=llm,
                                                         verbose=True)

    while True:
        query = input("提问：")
        response = daily_team_leader_agent.chat(query, tool_choice="auto")


def test_react_agent_with_query_plan_ahead():
    # Agent提前的规划适合，适合复杂一些但是没有标准SOP的任务

    from test import running_data_tool, team_member_tool
    running_data_tool = FunctionTool.from_defaults(fn=running_data_tool)
    team_member_tool = FunctionTool.from_defaults(fn=team_member_tool)

    # 生成query plan的工具
    response_synthesizer = get_response_synthesizer(llm=init_chat_llm())
    query_plan_tool = QueryPlanTool.from_defaults(
        query_engine_tools=[running_data_tool, team_member_tool],
        response_synthesizer=response_synthesizer,
    )
    # print(query_plan_tool.metadata.to_openai_tool())
    # print(query_plan_tool.metadata)

    agent = ReActAgent.from_tools(
        [query_plan_tool],
        max_function_calls=3,
        llm=init_chat_llm(),
        verbose=True,
    )




if __name__ == '__main__':
    test_react_agent()
    # test_react_agent_with_query_plan_ahead()
