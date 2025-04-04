import os
import openai
import dotenv
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# --- ClaimEvidence Pydantic 모델 ---
class ClaimEvidence(BaseModel):
    claim: str = Field(description="The main claim or assertion made in the text")
    evidence: List[str] = Field(description="A list of sentences or phrases supporting the claim")

# --- 핵심 주장과 근거 추출 함수 ---
def extract_claims_and_evidence_lc(transcript: str, model_name: str = "gpt-4o") -> dict:
    dotenv.load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        print("[ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        return {"claim": "API KEY 없음", "evidence": []}
    
    try:
        parser = JsonOutputParser(pydantic_object=ClaimEvidence)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that extracts the main claim and supporting evidence from a transcript. Respond using the following JSON format:\n{format_instructions}. If there are any typos or sentences that do not fit the context, please correct them accordingly. Answer in Korean."),
            ("user", "Analyze the following transcript:\n\n{transcript}")
        ])
        llm = ChatOpenAI(temperature=0.7, model_name=model_name)
        chain = prompt | llm | parser
        print(f"'{model_name}' 모델을 사용하여 주장/근거 추출 시작 (LCEL, JsonOutputParser)...")
        result_dict = chain.invoke({
            "transcript": transcript,
            "format_instructions": parser.get_format_instructions()
        })
        print("주장/근거 추출 완료.")
        return result_dict
    except openai.AuthenticationError as e:
        print(f"[ERROR] OpenAI API 키 인증 실패: {e}")
        return {"claim": "API Key 인증 오류", "evidence": []}
    except Exception as e:
        print(f"주장/근거 추출 중 오류 발생: {e}")
        return {"claim": "처리 중 오류 발생", "evidence": [str(e)]}

# --- Keywords Pydantic 모델 ---
class Keywords(BaseModel):
    keywords: List[str] = Field(description="A list of core keywords extracted from the input text")

# --- 핵심 키워드 추출 함수 ---
def extract_keywords_from_claim_evidence(claim_evidence: dict, model_name: str = "gpt-4o") -> dict:
    claim_text = claim_evidence.get("claim", "")
    evidence_list = claim_evidence.get("evidence", [])
    combined_text = claim_text + "\n" + " ".join(evidence_list)
    
    parser = JsonOutputParser(pydantic_object=Keywords)
    #프롬프트 템플릿 수정: 근거의 진위 여부를 확인할 수 있는 구체적 문구를 추출하도록 요청
    keyword_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts verification phrases from a given text. Instead of just listing individual keywords, provide concise phrases that capture the core verifiable facts. Respond using the following JSON format:\n{format_instructions}"),
        ("user", "Analyze the following text and generate key verification phrases that could be used to check the veracity of the evidence. For example, if the text indicates a unique medical case, include a phrase like '국내 최초 폐암 4기 환자가 완치된 사례'.\n\nText:\n\n{combined_text}")
    ])
    
    llm = ChatOpenAI(temperature=0.7, model_name=model_name)
    chain = keyword_prompt | llm | parser
    print(f"'{model_name}' 모델을 사용하여 핵심 키워드 추출 시작...")
    result_keywords = chain.invoke({
        "combined_text": combined_text,
        "format_instructions": parser.get_format_instructions()
    })
    print("핵심 키워드 추출 완료.")
    return result_keywords

# --- 테스트 실행 ---
if __name__ == "__main__":
    test_transcript = """
현재 난리난 국내 최초 폐암 4기 완치 사례. 한 63세 남성이 일주일간 언어장애를 호소하며 영남대 응급실을 찾았는데, 비소세포 폐암으로 판명됨.
이 환자는 6개월간 유한양행의 폐암치료제 렉라자를 투여받았으며, 기존 암세포가 모두 사라져 완전 관해 판정을 받았고, 이후 11개월간 재발 증거는 발견되지 않음.
일부에서는 담배를 안 끊어도 되는 것 아니냐는 반응도 나옴.
    """
    # 핵심 주장 및 근거 추출
    claim_evidence_result = extract_claims_and_evidence_lc(test_transcript)
    print("\n=== 주장/근거 추출 결과 ===")
    print(json.dumps(claim_evidence_result, ensure_ascii=False, indent=2))
    
    # 핵심 키워드 추출
    keywords_result = extract_keywords_from_claim_evidence(claim_evidence_result)
    print("\n=== 핵심 키워드 추출 결과 ===")
    print(json.dumps(keywords_result, ensure_ascii=False, indent=2))
