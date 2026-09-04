import sys
sys.path.append('/home/data/lcxnb1314/big/hello-agents-main/trip_agent')
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
import asyncio
import json
import time
from typing_extensions import TypedDict
from uuid import uuid4

from config import get_settings
from services.llm_service import get_llm_service
from models.schemas import TripRequest, TripPlan
from utils.jsonl_tracer import JsonlTracer

ATTRACTION_AGENT_PROMPT = """
你是景点搜索专家。

根据用户提供的城市和旅行偏好搜索合适的景点。

要求：
1. 必须调用 maps_text_search 获取真实景点。
2. 不允许自己编造景点。
3. 优先选择符合用户旅行偏好的景点。
4. 返回景点名称、地址、经纬度等有用信息。
"""

# ATTRACTION_AGENT_PROMPT = """
# 你是专业的旅行景点检索专家。

# 你的任务是根据用户的目的地和旅行偏好，
# 利用高德地图工具搜索真实、适合游客参观的景点。

# 工作流程：

# 1. 首先分析用户偏好，区分哪些偏好与“景点选择”有关。
#    例如：
#    - 历史文化 → 博物馆、历史遗迹、古建筑、名胜古迹、文化场馆
#    - 自然风景 → 公园、山岳、湖泊、自然景区
#    - 亲子 → 动物园、科技馆、主题乐园
#    - 美食 → 属于餐饮偏好，不要直接作为景点搜索关键词

# 2. 不要把多个抽象偏好直接拼成一个 maps_text_search 查询词。

# 3. 根据与景点相关的偏好，生成多个具体景点类别或关键词，
#    分别调用 maps_text_search 进行搜索。

# 4. 优先推荐：
#    - 具有明确旅游属性的景点；
#    - 具有代表性的博物馆、名胜古迹、历史遗迹、文化场馆等；
#    - 与用户偏好高度相关的地点。

# 5. 排除：
#    - 普通餐厅；
#    - 普通商店；
#    - 连锁餐饮；
#    - 便利店；
#    - 与旅游无明显关系的普通生活服务 POI。

# 6. 所有推荐地点必须来自 maps_text_search 的真实返回结果。
#    不得编造工具未返回的景点。

# 7. 如果搜索结果不足，可以继续换一个更具体的关键词搜索，
#    不要自行补充不存在的地点。

# 8. 最终对搜索结果去重后，推荐最合适的景点。
# """

WEATHER_AGENT_PROMPT = """
你是天气查询专家。

必须使用 maps_weather 工具查询用户指定城市的天气。
禁止自行编造天气信息。
"""

HOTEL_AGENT_PROMPT = """
你是酒店推荐专家。

必须使用 maps_text_search 搜索真实酒店。
根据城市、住宿偏好推荐合适酒店。
禁止自行编造酒店。
"""

PLANNER_AGENT_PROMPT = """
你是专业的旅行行程规划专家。

你的任务是根据用户的旅行需求，以及上游智能体提供的景点、天气和酒店信息，
生成完整、合理、可执行的旅行计划。

你会获得以下信息：
1. 用户旅行需求，包括城市、日期、预算、旅行偏好、住宿偏好等；
2. 景点搜索专家提供的真实景点信息；
3. 天气查询专家提供的天气信息；
4. 酒店推荐专家提供的真实酒店信息。

规划要求：

1. 只能基于用户输入和上游智能体提供的信息进行规划，不得自行编造不存在的景点、酒店、天气等事实。

2. 根据旅行日期合理安排每日行程，确保每天的景点数量和游览强度合理，避免安排过于紧凑。

3. 根据景点的位置和行程顺序合理组织每日路线，尽量减少不必要的往返。

4. 必须考虑天气因素：
   - 如果天气适合户外活动，可以优先安排户外景点；
   - 如果存在降雨、高温或其他不利天气，应优先考虑室内景点或调整活动安排。

5. 根据用户的旅行偏好选择和排序景点，例如历史文化、自然风景、美食、亲子、休闲等。

6. 根据用户的住宿偏好和行程安排选择合适酒店。
   优先考虑交通便利、靠近主要活动区域的酒店。

7. 结合用户预算合理安排住宿、餐饮、交通和景点支出，不要明显超过用户预算。

8. 每日行程应具有明确的时间顺序，例如上午、下午、晚上，并保证整体安排符合现实旅行逻辑。

9. 如果上游提供的信息不足，不要编造缺失信息。
   可以在计划中采用保守安排，或者明确说明相关信息不足。

10. 最终结果必须严格符合系统要求的 TripPlan 数据结构。
不要额外输出 TripPlan 结构之外的解释性文字、Markdown、代码块或其他内容。
"""

class TripPlannerState(TypedDict, total=False):
    run_id: str

    request: TripRequest

    attraction_response: str
    weather_response: str
    hotel_response: str

    trip_plan: TripPlan

class MultiAgentTripPlanner():
    """多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print('🔄 开始初始化多智能体旅行规划系统...')

        self.tracer = JsonlTracer(log_dir='/home/data/lcxnb1314/big/hello-agents-main/my_agent/logs')
        settings = get_settings()
        self.llm = get_llm_service()

        print("  - 创建MCP工具...")
        self.amap_tool = MultiServerMCPClient(
            {
                "amap":{
                    "transport":"stdio",
                    "command":"npx",
                    "args":[
                        "-y",
                        "@amap/amap-maps-mcp-server"
                    ],
                    "env":{
                        "AMAP_MAPS_API_KEY":settings.amap_api_key
                    }
                }
            }
        )

    async def initialize(self):
        """异步初始化MCP工具和Agent"""

        self.tools = await self.amap_tool.get_tools()

        tool_map = {
            tool.name: tool
            for tool in self.tools
        }

        # 1.创建景点搜索Agent
        print("  - 创建景点搜索Agent...",end=' ')
        self.attraction_agent = create_agent(
            name='景点搜索专家',
            model=self.llm,
            tools=[tool_map['maps_text_search']],
            system_prompt=ATTRACTION_AGENT_PROMPT
        )
        print('✅成功创建')

        # 2.创建天气查询Agent
        print("  - 创建天气查询Agent...", end=' ')
        self.weather_agent = create_agent(
            name="天气查询专家",
            model=self.llm,
            tools=[tool_map['maps_weather']],
            system_prompt=WEATHER_AGENT_PROMPT
        )
        print('✅成功创建')

        # 3.创建酒店推荐Agent
        print("  - 创建酒店推荐Agent...", end=' ')
        self.hotel_agent = create_agent(
            name="酒店推荐专家",
            model=self.llm,
            tools=[tool_map['maps_text_search']],
            system_prompt=HOTEL_AGENT_PROMPT
        )
        print('✅成功创建')

        # 4.创建行程规划Agent
        print("  - 创建行程规划Agent...", end=' ')
        self.planner_agent = create_agent(
            name="行程规划专家",
            model=self.llm,
            tools=[],
            system_prompt=PLANNER_AGENT_PROMPT,
            response_format=TripPlan
        )
        print('✅成功创建')

        # 5.agent逻辑规划
        print("  - langgraph创建...", end=' ')
        self.graph = self._build_graph()
        print('✅成功创建')

    async def _attraction_node(
            self,
            state:TripPlannerState
    ):
        start_time = time.perf_counter()

        run_id = state["run_id"]
        request = state['request']

        preferences = (
            "、".join(request.preferences)
            if request.preferences
            else "热门景点"
        )

        query = (
            f"搜索{request.city}适合旅行的景点。"
            f"用户偏好：热门景点"
        )

        print(
            f"🏛️ [{run_id}] attraction START "
            f"| city={request.city}"
        )

        # 节点开始
        self.tracer.write(
            run_id,
            "attraction",
            "start",
            {
                "city": request.city,
                "preferences": request.preferences
            }
        )

        # 保存 System Prompt
        self.tracer.write(
            run_id,
            "attraction",
            "system_prompt",
            ATTRACTION_AGENT_PROMPT
        )

        # 保存 User Message
        self.tracer.write(
            run_id,
            "attraction",
            "query",
            query
        )

        result = await self.attraction_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )
         # 保存完整 Agent 对话
        self._trace_messages(
            run_id,
            "attraction",
            result["messages"]
        )

        response = result["messages"][-1].content

        elapsed = time.perf_counter() - start_time

        self.tracer.write(
            run_id,
            "attraction",
            "end",
            {
                "elapsed": elapsed,
                "response": response
            }
        )

        print(
            f"✅ [{run_id}] attraction DONE "
            f"| {elapsed:.2f}s"
        )

        return {
            "attraction_response":response
        }

    async def _weather_node(
        self,
        state: TripPlannerState
    ):
        start_time = time.perf_counter()

        run_id = state["run_id"]
        request = state["request"]

        query = (
            f"查询{request.city}的天气情况。"
            f"旅行时间：{request.start_date}到{request.end_date}"
        )

        print(
            f"🌤️ [{run_id}] weather START "
            f"| city={request.city}"
        )

        self.tracer.write(
            run_id,
            "weather",
            "start",
            {
                "city": request.city,
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        )

        # 保存 System Prompt
        self.tracer.write(
            run_id,
            "weather",
            "system_prompt",
            WEATHER_AGENT_PROMPT
        )

        # 保存 User Message
        self.tracer.write(
            run_id,
            "weather",
            "query",
            query
        )

        result = await self.weather_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        self._trace_messages(
            run_id,
            "weather",
            result["messages"]
        )

        response = result["messages"][-1].content
        
        elapsed = time.perf_counter() - start_time
        
        self.tracer.write(
            run_id,
            "weather",
            "end",
            {
                "elapsed": elapsed,
                "response": response
            }
        )

        # 控制台：只输出摘要
        print(
            f"✅ [{run_id}] weather DONE "
            f"| {elapsed:.2f}s"
        )

        return {
            "weather_response":response
        }

    async def _hotel_node(
        self,
        state: TripPlannerState
    ):
        
        start_time = time.perf_counter()

        run_id = state["run_id"]
        request = state["request"]

        query = (
            f"搜索{request.city}适合用户的酒店。"
            f"住宿偏好：{request.accommodation}"
        )

        print(
            f"🏨 [{run_id}] hotel START "
            f"| city={request.city}"
        )

        # JSONL：节点开始
        self.tracer.write(
            run_id,
            "hotel",
            "start",
            {
                "city": request.city,
                "accommodation": request.accommodation
            }
        )

        # 保存 System Prompt
        self.tracer.write(
            run_id,
            "hotel",
            "system_prompt",
            HOTEL_AGENT_PROMPT
        )

        # 保存 User Message
        self.tracer.write(
            run_id,
            "hotel",
            "query",
            query
        )
        
        result = await self.hotel_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        self._trace_messages(
            run_id,
            "hotel",
            result["messages"]
        )
        
        response = result["messages"][-1].content
                
        elapsed = time.perf_counter() - start_time
                
        self.tracer.write(
            run_id,
            "hotel",
            "end",
            {
                "elapsed": elapsed,
                "response": response
            }
        )

        print(
            f"✅ [{run_id}] hotel DONE "
            f"| {elapsed:.2f}s"
        )

        return {
            "hotel_response":response
        }

    async def _planner_node(
        self,
        state: TripPlannerState
    ):
        request = state["request"]
        run_id = state["run_id"]

        start_time = time.perf_counter()


        query = f"""
请根据以下信息生成完整旅行计划。

【用户需求】
{request}

【景点搜索结果】
{state["attraction_response"]}

【天气查询结果】
{state["weather_response"]}

【酒店推荐结果】
{state["hotel_response"]}
"""

        print(
            f"🧠 [{run_id}] planner START "
            f"| city={request.city}"
        )

        self.tracer.write(
            run_id,
            "planner",
            "start"
        )

        self.tracer.write(
            run_id,
            "planner",
            "system_prompt",
            PLANNER_AGENT_PROMPT
        )

        self.tracer.write(
            run_id,
            "planner",
            "query",
            query
        )
        
        result = await self.planner_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        self._trace_messages(
            run_id,
            "planner",
            result["messages"]
        )

        trip_plan = result[
            "structured_response"
        ]
                   
        elapsed = time.perf_counter() - start_time

        self.tracer.write(
            run_id,
            "planner",
            "end",
            {
                "elapsed": elapsed,
                "trip_plan":
                    trip_plan.model_dump()
            }
        )

        print(
            f"✅ [{run_id}] planner DONE "
            f"| {elapsed:.2f}s"
        )          
        
        return {
            "trip_plan": trip_plan
        }

    def _build_graph(self):

        graph = StateGraph(TripPlannerState)

        # 添加节点
        graph.add_node(
            "attraction",
            self._attraction_node
        )

        graph.add_node(
            "weather",
            self._weather_node
        )

        graph.add_node(
            "hotel",
            self._hotel_node
        )

        graph.add_node(
            "planner",
            self._planner_node
        )

        # 入口
        graph.add_edge(
            START,
            "attraction"
        )
        graph.add_edge(
            START,
            "weather"
        )
        graph.add_edge(
            START,
            "hotel"
        )

        graph.add_edge(
            ["attraction", "weather", "hotel"],
            "planner"
        )

        # 结束
        graph.add_edge(
            "planner",
            END
        )

        return graph.compile()

    async def plan_trip(
            self,
            request:TripRequest
    ):
        run_id = uuid4().hex[:12]

        self.tracer.write(
            run_id=run_id,
            node="graph",
            event="request",
            data=request.model_dump()
        )

        start_time = time.perf_counter()
        
        result = await self.graph.ainvoke(
            {
                "run_id": run_id,
                "request":request
            }
        )
        elapsed = time.perf_counter() - start_time

        # 保存最终输出
        self.tracer.write(
            run_id=run_id,
            node="graph",
            event="completed",
            data={
                "elapsed": elapsed,
                "trip_plan":
                    result["trip_plan"].model_dump()
            }
        )

        print(
            f"✅ 旅行规划完成 "
            f"[run_id={run_id}] "
            f"[{elapsed:.2f}s]"
        )

        return result['trip_plan']
    
    async def print_mcp_tools(self):
        tools = await self.amap_tool.get_tools()

        for tool in tools:
            print("=" * 80)
            print(f"工具名称：{tool.name}")
            print(f"工具描述：{tool.description}")

            print("参数定义：")
            print(
                json.dumps(
                    tool.args_schema,
                    ensure_ascii=False,
                    indent=2
                )
            )

    def _trace_messages(
        self,
        run_id: str,
        node: str,
        messages
    ):
        for index, message in enumerate(messages):

            data = {
                "index": index,

                "message_type":
                    type(message).__name__,

                "content":
                    getattr(
                        message,
                        "content",
                        None
                    ),

                "name":
                    getattr(
                        message,
                        "name",
                        None
                    ),

                "tool_calls":
                    getattr(
                        message,
                        "tool_calls",
                        None
                    ),

                "tool_call_id":
                    getattr(
                        message,
                        "tool_call_id",
                        None
                    ),

                "additional_kwargs":
                    getattr(
                        message,
                        "additional_kwargs",
                        None
                    ),

                "response_metadata":
                    getattr(
                        message,
                        "response_metadata",
                        None
                    )
            }

            self.tracer.write(
                run_id=run_id,
                node=node,
                event="message",
                data=data
            )

async def main():
    # 1. 创建系统
    agent = MultiAgentTripPlanner()

    # 2. 初始化 MCP + Agent + LangGraph
    await agent.initialize()

    print("\n" + "=" * 80)
    print("开始测试完整旅行规划流程")
    print("=" * 80)

    # 3. 构造测试请求
    request = TripRequest(
        city="北京",
        start_date="2026-08-12",
        end_date="2026-08-15",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=[
            "历史文化",
            "美食"
        ],
        free_text_input="希望多安排一些博物馆"
    )

    print("\n测试请求：")
    print(
        json.dumps(
            request.model_dump(),
            ensure_ascii=False,
            indent=2
        )
    )
    # 4. 执行整个 LangGraph
    print("\n开始执行 LangGraph...\n")
    trip_plan = await agent.plan_trip(request)

    # 5. 查看结果
    print("\n" + "=" * 80)
    print("旅行规划完成")
    print("=" * 80)

    print("返回类型：")
    print(type(trip_plan))

    print("\n最终 TripPlan：")
    print(
        json.dumps(
            trip_plan.model_dump(),
            ensure_ascii=False,
            indent=2
        )
    )

if __name__ == '__main__':
    asyncio.run(main())
