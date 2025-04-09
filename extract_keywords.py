import os
import openai
# .env 파일에서 환경변수 로드하기 위한 모듈
import dotenv
import json
# Langchain 커뮤니티 버전에서 사용하는 Chat 모듈들
from langchain_openai import ChatOpenAI
# 메시지 목록을 기반으로 프롬프트 템플릿을 구성하는 클래스
from langchain_core.prompts import ChatPromptTemplate
# 시스템 메시지를 정의하는 클래스
from langchain_core.messages import SystemMessage
# 출력 결과를 JSON 형식으로 파싱
from langchain_core.output_parsers import JsonOutputParser
# 출력을 텍스트로만 뽑기 위해서서
from langchain_core.output_parsers import StrOutputParser
# Pydantic을 사용하여 데이터 스키마를 정의하기 위한 모듈
# pydantic : 데이터 유효성 검사와 설정 관리에 사용되는 파이썬 라이브러리
# pydantic의 핵심은 타입 힌트를 기반으로 입력 데이터를 자동으로 검증하고 변환하는 것
from pydantic import BaseModel, Field
# ??
from typing import List

dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 텍스트 정제 함수 선언
def clean_text(text:str):
    system_message = (
        """
        너는 텍스트 내에서 문맥의 흐름에 맞지 않는 오탈자를 탐지하여, 문맥에 부합하는 적절한 표현으로 수정하는 텍스트 분석 전문가야.
        주어진 텍스트에서 문맥에 어색하거나 부자연스러운 표현을 식별한 후, 이를 자연스럽게 수정하여 전체 문장이 원활하게 연결되도록 해야한다.
        문장을 이해하는 방식은 다음과 같다.
        - 단순히 단어의 사전적 의미뿐만 아니라 문맥 속에서 단어가 가지는 함축적 의미나 뉘앙스를 고려해야한다.
        - 특정 전문 분야의 텍스트일 경우, 해당 분야의 용어 사용 관례나 특수성을 이해하고 적용해야 한다.
        - 특히 전문 용어는 함부로 다른 용어로 바꾸지 말고 발음 유사성을 반드시 고려해서 수정해야 합니다.
        예를 들면 비누세포 폐암은 비소세포 폐암을 잘못 인식한 단어이다. "소세포 폐암"과 "비소세포 폐암"은 전혀 다른 질환이므로, 단순히 소리만 유사하다고 해서 "소세포 폐암"으로 바꾸지 말라.
        - 음성 데이터를 텍스트로 변환한 후의 텍스트라 완전 다른 단어가 아닌 발음이 비슷한 단어로 잘못 인식된 경우일 확률이 높다.
        문장에서 단어의 오류를 탐지하는 원칙은 다음과 같다.
        - 분석할 텍스트는 음성 데이터를 텍스트로 변환한 결과물이다.
        - 오류로 탐지된 단어들은 발음이 비슷한 단어로 잘못 인식된 경우가 많다.
            예를 들면 4기 -> 사기,자기 , 12개월 -> 11개월 , 
        - 단순히 철자가 틀린 것 뿐만 아니라, 문맥상 의미가 통하지 않거나 어색한 단어를 찾아내는데 집중해야 한다.
        - 오류를 감지한 후 해당 문장 전반적인 문법적 정확성과 표현의 자연스러움을 반드시 재검토하여 최종 확인해본다.
        문장 교정 원칙은 다음과 같다.
        - 문맥에 어울리지 않는 단어를 찾아내고, 그 단어를 문맥에 맞는 적절한 표현으로 수정한다.
        - 텍스트는 음성 데이터를 텍스트로 변환한 결과물이라는것을 고려해야한다.
        - 오류로 탐지된 단어는 발음이 유사한 단어일 가능성이 높으므로, 완전히 다른 단어로 수정하기보다는 발음이 유사한 단어로 수정하는 것이 우선이다.
            예를 들면 정변 -> 병변, 사기 -> 자기가 아니라 4기로 수정해야 하빈다.
        - 수정할 단어는 해당 문맥에서 가장 자연스럽고 의미가 명확하게 전달되는 표현이어야 한다.
        - 명확한 판단이 어려울 경우 임의로 수정하기보다는 해당 부분을 *로 표시한다.
        - 단어를 수정한 후 해당 문장 전체의 문법적 정확성과 자연스러운 표현 여부를 반드시 재확인한다.
        - 전체 문장을 그대로 출력하되, 문장 내 오류가 있는 부분만 정확하게 수정하여 반영하세요.
        """
    )
    user_message= (
        """
        다음 텍스트 {text}내에서 문맥의 흐름에 맞지 않는 오탈자를 탐지하여, 문맥에 부합하는 적절한 표현으로 수정해주세요 
        단순히 단어의 사전적 의미에 국한하지 않고, 문맥 속에서 나타나는 함축적 의미와 뉘앙스를 고려하여 문장을 해석한 후, 오류를 탐지하세요.
        오류를 식별한 후에는 해당 문장의 전반적인 문법적 정확성과 표현의 자연스러움을 재검토하여, 전체 문장이 매끄럽고 문법적으로 올바르게 수정되도록 하세요.
        오류로 탐지된 단어는 발음이 유사한 단어일 가능성이 높으므로, 완전히 다른 단어로 수정하기보다는 발음이 유사한 단어로 수정하는 것이 우선이다.
        문맥이 자연스럽고 문법적으로 정확한지 재검토한 후, 애매하거나 잘못된 부분이 발견되면 오류 탐지 단계를 다시 수행하여 한 번 더 검증하고 수정하세요.
        재검토 한 후에도 애매한 부분이 없다면 그대로 출력하세요. 애매한 부분이 있다면 *로 표시하세요.
        """
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_message),
        ("user",user_message)
    ])
    # LLM 객체 생성
    llm = ChatOpenAI(model_name='gpt-4o')
    # 여러 단계를 연결하는 'Chain(체인)'을 구성 - LangChain Expression Language (LCEL)의 파이프(|) 연산자
    chain = prompt | llm | StrOutputParser()
    cleaned_text = chain.invoke({"text": text})
    print("텍스트 정제 완료")
    return cleaned_text
"""
# BaseModel : pydantic에서 제공하는 기본 모델 클래스 상속 받아 사용
# 이 클래스를 인스턴스화(클래스를 바탕으로 실제 객체를 생성)할 때, BaseModel은 자동으로 입력 데이터의 타입과 구조가 정확한지 검증
# 원하는 데이터 구조(예: { "claim": "주장 내용", "evidence": ["근거1", "근거2"] })
class ClaimEvidence(BaseModel):
    # 주장 필드는 문자열이어야 함 , description은 필드에 대한 설명
    # Field함수는 타입 힌트 이상의 상세한 설정(설명, 기본값, 별칭, 유효성 검증 규칙 등)을 부여하여 모델을 더욱 명확하게 정의
    claim: str = Field(description="The main claim or assertion made in the text")
    # "evidence"는 반드시 리스트 형태여야 함
    evidence: List[str] = Field(description="A list of sentences or phrases supporting the claim")

# --- 핵심 주장과 근거 추출 함수 ---
# transcript : 영상 텍스트 , model_name : 사용할 모델 기본값 "gpt-4o"
def extract_claims_and_evidence(transcript: str, model_name: str = "gpt-4o"):
    # OpenAI API 키를 환경변수에서 로드
    dotenv.load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        print("[ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        return {"claim": "API KEY 없음", "evidence": []}
    # try-except 블록 : 예외가 발생하면 프로그램이 중단되는 대신 except블록으로 이동
    try:
        # JsonOutputParser : JSON 형식으로 출력 결과를 파싱하기 위한 클래스
        # 이 파서가 기대하는 JSON 구조가 ClaimEvidence 모델과 일치해야 함
        # LangChain에게 "LLM의 최종 응답이 ClaimEvidence 모델 구조에 맞는 JSON 형식이 되도록 유도하고, 그 결과를 파싱(해석)해줘"라는 의미
        parser = JsonOutputParser(pydantic_object=ClaimEvidence)
        # Langchain의 ChatPromptTemplate 객체 생성 , LLM에게 전달될 메시지의 역할과 내용을 정의
        # 프롬프트 템플릿 : 시스템 메시지와 사용자 메시지를 결합하여 LLM에게 전달할 최종 프롬프트를 생성
        prompt = ChatPromptTemplate.from_messages([
            # 시스템 메시지 : LLM에게 전체 대화의 맥락이나 따라야 할 지침을 제공
            ("system", "너는 특정 문장에서 핵심 주장과 이를 뒷받침하는 근거를 정확하게 추출하는 전문가야.여기서 '주장'은 문장이 전달하고자 하는 주요 메시지를 의미하며, '근거'는 그 주장을 뒷받침하는 세부정보들을 나타냅니다."),
            # 사용자 메시지 : LLM에게 특정 작업을 요청하는 역할
            ("user", "다음 텍스트를 분석하여, 핵심 주장과 근거를 JSON 형식으로 추출해 주세요. 텍스트에 오타가 있거나 문맥상 어색한 부분이 있으면 수정하고 분석을 진행해주세요.:\n\n{transcript}")
        ])
        # LangChain의 ChatOpenAI 객체를 생성 - gpt-4o 모델 사용
        llm = ChatOpenAI(model_name='gpt-4o')
        # LangChain Expression Language (LCEL)의 파이프(|) 연산자를 사용
        # 여러 단계를 연결하는 'Chain(체인)'을 구성 - 최신 방식
        # 데이터는 프롬프트를 통과하여 형식이 변환됨 -> 그 결과가 LLM에 전달됨 -> LLM의 응답이 다시 parser로 전달되어 최종 결과가 생성됨
        chain = prompt | llm | parser
        print(f"'{model_name}' 모델을 사용하여 주장/근거 추출 시작 (LCEL, JsonOutputParser)...")
        # 구성된 chain을 실행 - invoke() 메서드 사용
        result_dict = chain.invoke({
            "transcript": transcript,
            # parser.get_format_instructions() 메서드는 JsonOutputParser 객체(parser)가 가지고 있는 ClaimEvidence정보를 바탕으로
            #LLM에게 어떤 JSON 형식으로 응답해야 하는지에 대한 구체적인 지침 문자열을 자동으로 생성해주는 함수
            "format_instructions": parser.get_format_instructions()
        })
        print("주장/근거 추출 완료.")
        return result_dict
    # 예외 처리 : OpenAI API와의 통신 중 발생할 수 있는 다양한 오류를 처리
    except openai.AuthenticationError as e:
        print(f"[ERROR] OpenAI API 키 인증 실패: {e}")
        return {"claim": "API Key 인증 오류", "evidence": []}
    except Exception as e:
        print(f"주장/근거 추출 중 오류 발생: {e}")
        return {"claim": "처리 중 오류 발생", "evidence": [str(e)]}

# LLM이 생성할 키워드 관련 JSON 출력의 **구조(스키마)**를 정의하는 역할
class Keywords(BaseModel):
    # 핵심 키워드 필드 : 문자열 리스트 형태로 정의

    keywords: List[str] = Field(description="입력 텍스트로부터 추출된 핵심 키워드들의 목록")

# --- 핵심 키워드 추출 함수 ---
def extract_keywords_from_claim_evidence(claim_evidence: dict, model_name: str = "gpt-4o"):
    # 주장과 근거를 분리하여 하나의 문장으로 결합
    claim_text = claim_evidence.get("claim", "")
    evidence_list = claim_evidence.get("evidence", [])
    combined_text = claim_text + "\n" + " ".join(evidence_list)
    # Keywords Pydantic 모델 구조를 따라야 함을 명시
    parser = JsonOutputParser(pydantic_object=Keywords)
    # 프롬프트 템플릿 수정: 근거의 진위 여부를 확인할 수 있는 구체적 문구를 추출하도록 요청
    keyword_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
    ),
    (
        "user",
    )
])

    llm = ChatOpenAI(model_name=model_name)
    chain = keyword_prompt | llm | parser
    print(f"'{model_name}' 모델을 사용하여 핵심 키워드 추출 시작...")
    result_keywords = chain.invoke({
        "combined_text": combined_text,
        "format_instructions": parser.get_format_instructions()
    })
    print("핵심 키워드 추출 완료.")
    return result_keywords
"""
# --- 테스트 실행 ---
if __name__ == "__main__":
    test_transcript = """
    현재 난리난 국내 최초 폐암 사기 완치 사례. 한 63세 남성이 일주일간 언어장애를 호소하다 영남대 응급실을 찾았는데 자기 비누세포 폐암으로 판명됨. 이 환자는 6개월간 유한양행의 폐암치료제 렉라자를 투여받았는데 정변이 모두 사라지고 새로운 암세포가 보이지 않는 완전관의 판정을 받음. 환자는 이후에도 렉라자를 투여받고 있는데 11개월간 재발 증거는 발견되지 않음. 담배 안 끌어도 되겠는데?
    """
    cleaned_text = clean_text(test_transcript)
    print("\n=== 정제된 텍스트 ===")
    print(cleaned_text)
    
    """
    # 핵심 주장 및 근거 추출
    claim_evidence_result = extract_claims_and_evidence(test_transcript)
    print("\n=== 주장/근거 추출 결과 ===")
    print(json.dumps(claim_evidence_result, ensure_ascii=False, indent=2))
    
    # 핵심 키워드 추출
    keywords_result = extract_keywords_from_claim_evidence(claim_evidence_result)
    print("\n=== 핵심 키워드 추출 결과 ===")
    print(json.dumps(keywords_result, ensure_ascii=False, indent=2))
    """