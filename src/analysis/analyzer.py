import logging
# LiteLLM의 LLM 호출 구조 상속 받음
from .base_llm import BaseLLM
import json
# keyword가 llm의 답변으로써 문자열로 반환되기 때문에 리스트 변환과정이 필요함
import ast

# 로거 인스턴스 생성
logger = logging.getLogger(__name__)

# BaseLLM 을 상속받아 주장,근거,키워드 추출을 위한 LLM 인스턴스 생성
class TextAnalyzer:
    def __init__(self,llm:BaseLLM,prompt_manager):
        self.llm = llm # 상속을 여기서 받는게 아니라 BaseLLM 객체를 전달받기
        self.prompt_manager = prompt_manager
    # 주장과 근거 추출 함수
    def extract_claim_evidence(self,transcript:str) -> dict: 
        prompt = self.prompt_manager.get_prompt("claim_evidence")
        
        messages = [
            {"role": "system","content": prompt["prompts"]["system"]},
            {"role": "user","content": prompt["prompts"]["user"].format(transcript=transcript)}
        ]

        response = self.llm.llm_call(messages)
        logger.debug(f"LLM response for claim/evidence: {response}")
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response. Response was: {response}", exc_info=True)

    # 키워드 추출 함수
    def extract_keywords(self, claim_evidence: dict) -> list[str]:
        prompt = self.prompt_manager.get_prompt("keywords")
        claim = claim_evidence.get("claim","")
        evidence = claim_evidence.get("evidence",[]) 
        # 주장과 근거를 하나의 문자열로 결합
        combined_text = claim + "\n" + " ".join(evidence) 
        messages = [
            {"role": "system","content": prompt["prompts"]["system"]},
            {"role": "user","content": prompt["prompts"]["user"].format(combined_text=combined_text)}
        ]
        try:
            response = self.llm.llm_call(messages)
            logger.debug(f"LLM response for keywords: {response}")
            keyword_list = ast.literal_eval(response)
            if isinstance(keyword_list,list):
                return keyword_list
            else:
                return []
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse list from LLM response. Response was: {response}", exc_info=True)
            return []