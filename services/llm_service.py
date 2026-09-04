import sys
sys.path.append('/home/data/lcxnb1314/big/hello-agents-main/trip_agent')
from functools import lru_cache
from langchain_openai import ChatOpenAI
from config import get_settings

class LLMService:
    """LLM服务"""

    def __init__(self):
        settings = get_settings()

        self.llm = ChatOpenAI(
            model=settings.llm_model_id,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
        )

    def get_model(self) -> ChatOpenAI:
        return self.llm
    
    def chat(self, message: str) -> str:
        """非流式调用"""
        response = self.llm.invoke(message)
        return response.content
    
    def stream_chat(self, message: str):
        """流式调用"""
        for chunk in self.llm.stream(message):
            if chunk.content:
                yield chunk.content
        print()

@lru_cache
def get_llm_service() -> ChatOpenAI:
    return LLMService().get_model()

if __name__ == '__main__':
    llm = get_llm_service()
    print(type(llm))