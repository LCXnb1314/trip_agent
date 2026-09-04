import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

load_dotenv('/home/data/lcxnb1314/big/hello-agents-main/trip_agent/.env')


class Settings(BaseSettings):

    app_name:str = '智能体搭建测试'
    app_version:str = '1.0.0'

    host: str = "0.0.0.0"
    port: int = 8000

    # LLM配置
    llm_model_id:str = ''
    llm_api_key:str = ''
    llm_base_url:str = ''
    llm_timeout:int

    amap_api_key:str = ''

    model_config = SettingsConfigDict(
        env_file='/home/data/lcxnb1314/big/hello-agents-main/trip_agent/.env',
        case_sensitive=False,
        extra='ignore',
        frozen=True
    )

@lru_cache
def get_settings():
    return Settings()

if __name__ == '__main__':
    settings = get_settings()
    # settings.app_version = '1.0.1'
    print(settings)