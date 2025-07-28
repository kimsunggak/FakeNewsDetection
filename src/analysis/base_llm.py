# LiteLLM 라이브러리, 모델 호출 함수 
from litellm import completion
import logging

logger = logging.getLogger(__name__)

# LiteLLM의 Ollama LLM 인스턴스 생성
class BaseLLM:
    def __init__(self,model,api_base):
        self.model = model
        self.api_base = api_base
        logger.debug(f"BaseLLM initialized with model: {self.model}")
    # LiteLLM의 completion 함수(여러 모델을 일관성 있게 가져옴) 호출  
    def llm_call(self,messages):



        try:
            response = completion(
                model=self.model,
                messages=messages,
                api_base=self.api_base
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed for model: {self.model}", exc_info=True)
    
