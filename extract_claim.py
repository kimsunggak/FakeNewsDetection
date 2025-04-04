import os
import openai # Keep for potential error types, though direct key setting is removed
import dotenv
import json
from langchain_community.chat_models import ChatOpenAI
# Corrected core imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage # HumanMessage might not be needed if using .from_messages with tuples
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field # For defining JSON structure
from typing import List # For type hinting in Pydantic model

# --- Define the desired output structure using Pydantic ---
class ClaimEvidence(BaseModel):
    claim: str = Field(description="The main claim or assertion made in the text")
    evidence: List[str] = Field(description="A list of sentences or phrases supporting the claim")

def extract_claims_and_evidence_lc(transcript: str, model_name: str = "gpt-4o") -> dict:
    """
    주어진 transcript(자막 텍스트)에서 핵심 주장과 근거를 추출하여
    사전 정의된 Pydantic 모델(ClaimEvidence) 형태의 dict로 반환합니다.
    최신 LangChain (LCEL, JsonOutputParser) 방식을 사용합니다.

    Args:
        transcript (str): 분석할 텍스트.
        model_name (str): 사용할 OpenAI 챗 모델 이름.

    Returns:
        dict: 추출된 주장과 근거. 오류 시 기본 오류 dict 반환.
    """
    # .env 파일 로드 (OPENAI_API_KEY 필요)
    dotenv.load_dotenv()
    # API 키는 ChatOpenAI가 환경 변수에서 자동으로 읽음

    if not transcript:
        print("[ERROR] 분석할 텍스트가 비어 있습니다.")
        return {"claim": "입력 텍스트 없음", "evidence": []}

    try:
        # 1. Output Parser 초기화 (Pydantic 모델 사용)
        parser = JsonOutputParser(pydantic_object=ClaimEvidence)

        # 2. Prompt Template 생성
        #    JsonOutputParser의 get_format_instructions()를 포함하여 LLM이 JSON 형식 지침을 받도록 함
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that extracts the main claim and supporting evidence from a transcript. Respond using the following JSON format:\n{format_instructions},If there are any typos or sentences that do not fit the context, please correct them accordingly ,you must talk in Korean,"),
            ("user", "Analyze the following transcript:\n\n{transcript}")
        ])

        # 3. Chat Model 초기화
        llm = ChatOpenAI(temperature=0.7, model=model_name) # model_name -> model

        # 4. LCEL을 사용하여 Chain 구성
        #    prompt의 format_instructions 변수는 parser에서 자동으로 채워짐
        chain = prompt | llm | parser

        print(f"'{model_name}' 모델을 사용하여 주장/근거 추출 시작 (LCEL, JsonOutputParser)...")
        # 5. Chain 실행 (.invoke 사용)
        #    parser가 포함되었으므로 결과는 이미 파싱된 dict 형태임
        result_dict = chain.invoke({
            "transcript": transcript,
            "format_instructions": parser.get_format_instructions() # Inject format instructions
        })
        print("주장/근거 추출 완료.")

        # Pydantic 모델 검증을 거친 dict 반환
        return result_dict

    except openai.AuthenticationError as e:
            print(f"[ERROR] OpenAI API 키 인증 실패: {e}")
            return {"claim": "API Key 인증 오류", "evidence": []}
    except ImportError:
        print("[ERROR] 필요한 LangChain 라이브러리가 설치되지 않았습니다.")
        print("pip install langchain langchain-openai langchain-community pydantic")
        return {"claim": "라이브러리 누락", "evidence": []}
    except Exception as e:
        print(f"주장/근거 추출 중 오류 발생: {e}")
        # LLM 출력이 JSON 형식을 따르지 않아 파싱 실패하는 경우 포함될 수 있음
        return {"claim": "처리 중 오류 발생", "evidence": [str(e)]}

# --- 테스트 실행 ---
if __name__ == "__main__":
    test_transcript = """
현재 난리난 국내 최초 폐암 4기 완치 사례. 한 63세 남성이 일주일간 언어장애를 호소하며 영남대 응급실을 찾았는데, 비소세포 폐암으로 판명됨.
이 환자는 6개월간 유한양행의 폐암치료제 렉라자를 투여받았으며, 기존 암세포가 모두 사라져 완전 관해 판정을 받았고, 이후 11개월간 재발 증거는 발견되지 않음.
일부에서는 담배를 안 끊어도 되는 것 아니냐는 반응도 나옴.
    """ # '자기 비누세포' -> '비소세포' 수정, 마지막 문장 약간 수정
    result = extract_claims_and_evidence_lc(test_transcript)
    print("\n=== 테스트 결과 (LCEL, JsonOutputParser) ===")
    # 결과를 보기 좋게 JSON 문자열로 출력
    print(json.dumps(result, ensure_ascii=False, indent=2))

