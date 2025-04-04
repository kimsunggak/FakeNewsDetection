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
# Pydantic을 사용하여 데이터 스키마를 정의하기 위한 모듈
# pydantic : 데이터 유효성 검사와 설정 관리에 사용되는 파이썬 라이브러리
# pydantic의 핵심은 타입 힌트를 기반으로 입력 데이터를 자동으로 검증하고 변환하는 것
from pydantic import BaseModel, Field
# ??
from typing import List

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
    ("system", """
[의료 과학 검색어 생성 규칙]
당신은 **사실 검증용 검색어 최적화 전문가**입니다. 다음 원칙을 엄격히 적용하세요:

1. **핵심 개체 식별 범위** (모든 검색어에 2개 이상 포함 필수)
   • 기관: 병원/연구소/제약사명 (예: 서울대병원, 화이자)
   • 질병: 정확한 병명 + 병기 (예: 비소세포폐암 4기)
   • 치료법: 약물 상품명(성분명) + 투여 기간 (예: 렉라자(레크토닙) 6개월)
   • 결과: 수치 포함 증거 (예: 11개월 무재발, 종양 70% 감소)

2. **검색어 생성 전략**
   • (필수) 기관 + 질병 + 치료법 + 결과 조합
   • (선택) 다국어 병기: "EGFR 돌연변이" → "EGFR mutation"
   • 자연스러운 구어체 문장 (예: "유한양행 렉라자 폐암 4기 완전관해 사례")

3. **금지 사항**
   ※ 모호한 일반 용어 (예: "최신 치료법")
   ※ 특수문자(#/) 사용

4. **출력 형식**
   → {format_instructions}
   → 5~7개 검색어 생성

[예시 출력]
**Good**: 
- "영남대병원 비소세포폐암 4기 렉라자 6개월 투여 사례" 
- "레크토닙 EGFR 돌연변이(mutation) 생존율 연구"
- "폐암 4기 완전관해 11개월 무재발 통계"

**Bad**: 
- "암 치료" (too generic)
- "렉라자" (single word)
"""),
    ("user", """
▶︎ 검증 대상 텍스트:
{combined_text}

▶︎ 검색어 생성 조건:
- 의학적 정확성 100% 보장
- 검색 결과 신뢰도 최적화
""")
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

# --- 테스트 실행 ---
if __name__ == "__main__":
    test_transcript = """
현재 난리난 국내 최초 폐암 4기 완치 사례. 한 63세 남성이 일주일간 언어장애를 호소하며 영남대 응급실을 찾았는데, 비소세포 폐암으로 판명됨.
이 환자는 6개월간 유한양행의 폐암치료제 렉라자를 투여받았으며, 기존 암세포가 모두 사라져 완전 관해 판정을 받았고, 이후 11개월간 재발 증거는 발견되지 않음.
일부에서는 담배를 안 끊어도 되는 것 아니냐는 반응도 나옴.
    """
    # 핵심 주장 및 근거 추출
    claim_evidence_result = extract_claims_and_evidence(test_transcript)
    print("\n=== 주장/근거 추출 결과 ===")
    print(json.dumps(claim_evidence_result, ensure_ascii=False, indent=2))
    
    # 핵심 키워드 추출
    keywords_result = extract_keywords_from_claim_evidence(claim_evidence_result)
    print("\n=== 핵심 키워드 추출 결과 ===")
    print(json.dumps(keywords_result, ensure_ascii=False, indent=2))
