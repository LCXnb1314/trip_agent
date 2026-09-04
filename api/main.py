import sys
sys.path.append('/home/data/lcxnb1314/big/hello-agents-main/trip_agent')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import get_settings
from agents.trip_agent import MultiAgentTripPlanner
from api.routes import trip

settings = get_settings()

# app = FastAPI(lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(trip.router, prefix="/api")
# app.include_router(poi.router, prefix="/api")
# app.include_router(map_routes.router, prefix="/api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print("=" * 60)

    print("\n📚 API文档: http://localhost:8000/docs")
    print("📖 ReDoc文档: http://localhost:8000/redoc")

    planner = MultiAgentTripPlanner()
    await planner.initialize()
    app.state.planner = planner
    # 到这里，启动阶段完成
    yield

    # 以后如果需要，可以在这里写关闭逻辑
    print("应用正在关闭")

app = FastAPI(lifespan=lifespan)

app.include_router(trip.router, prefix="/api")

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )