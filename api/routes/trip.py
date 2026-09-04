from fastapi import APIRouter, Request
from models.schemas import TripRequest, TripPlanResponse

router = APIRouter(prefix="/trip")

@router.post("/plan", response_model=TripPlanResponse)
async def plan_trip(trip_request:TripRequest, request:Request):
    print(f"\n{'='*60}")
    print(f"📥 收到旅行规划请求:")
    print(f"   城市: {trip_request.city}")
    print(f"   日期: {trip_request.start_date} - {trip_request.end_date}")
    print(f"   天数: {trip_request.travel_days}")
    print(f"{'='*60}\n")

    agent = request.app.state.planner

    print("🚀 开始生成旅行计划...")
    trip_plan = await agent.plan_trip(trip_request)

    return TripPlanResponse(
        success=True,
        message="旅行计划生成成功",
        data=trip_plan
    )     