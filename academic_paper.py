# arXiv API와 연동하기 위한 파이썬 라이브러리
# arxiv에 있는 논문을 검색하고 메타데이터와 PDF링크 등의 정보를 가져옴
import arxiv
def search_arxiv(keyword,max_papers=10):
    try:
        # arxiv에 api요청을 보낼 클라이언트 생성
        # 반복 호출 시 일관된 설정 적용하기 위함 , 연결 관리 및 요청 최적화
        client = arxiv.Client(
            page_size=100,  # 한 페이지당 최대 100개의 논문 요청
            delay_seconds=3.0,  # 요청 간 대기 시간 - 원문 지침 (서버 과부화 방지)
            num_retries=3,  # 재시도 횟수 (요청 실패 시)
        )
        # 검색 쿼리 설정
        # ti: 제목, abs: 초록 - 제목 또는 초록에 키워드가 포함된 논문 검색
        query = f'ti:"{keyword}" OR abs:"{keyword}"'
        # arxiv.Search() : arxiv API에 검색 요청을 보내기 위한 객체
        Search = arxiv.Search(
            query= query,  # 검색할 카테고리
            max_results= max_papers,  # 최대 검색 결과 수
            sort_by=arxiv.SortCriterion.Relevance,  # 정렬 기준 - 관련성 기준으로 정렬
            sort_order = arxiv.SortOrder.Descending  #내림차순 = 최신 논문 우선
            )
        print(f"arxiv에서 '{keyword}' 키워드로 검색된 논문을 가져오는 중...")
        results = client.results(Search) # 검색 실행 및 결과
        founded_papers = 0
        top_paper = None # 가장 관련성 높은 논문 pdf는 저장
        for paper in results:
            founded_papers += 1
            if founded_papers == 1:
                top_paper = paper
            print(f"논문 제목: {paper.title}")
            print(f"저자: {', '.join([author.name for author in paper.authors])}")
            print(f"초록: {paper.summary}")
            print(f"링크: {paper.entry_id}")
            print(f"제출일: {paper.published}")
            print(f"최종 수정일: {paper.updated}")
            print(f"논문 카테고리: {paper.primary_category}")
            print(f"PDF 링크: {paper.pdf_url}")
        if founded_papers == 0:
            print(f"'{keyword}' 키워드로 검색된 논문이 없습니다.")
        else:
            print(f"'{keyword}' 키워드로 총 {founded_papers}개의 논문이 검색되었습니다.")
            if top_paper:
                print(f"가장 관련성 높은 논문 제목: {top_paper.title}")
                top_paper.download_pdf(dirpath=".", filename="top_paper.pdf")  # 논문 PDF 다운로드
    except Exception as e:
        print(f"arxiv에서 논문 검색 중 오류 발생: {e}")
        return None
# 원본 XML 문자에서 body 부분의 텍스트만 추출하는 함수
# 우리가 필요한건 본문 텍스트뿐임    
import xml.etree.ElementTree as ET
def get_pmc_body_text(xml_text: str) -> str:
    """
    주어진 PMC 원본 XML 문자열(xml_text)에서 모든 <article> 요소의 <body> 부분의 텍스트를 
    재귀적으로 추출하여 하나의 문자열로 합쳐 반환한다.
    각 논문은 구분자를 통해 분리된다.
    """
    body_texts = []
    root = ET.fromstring(xml_text)
    articles = root.findall(".//article")
    
    for idx, article in enumerate(articles, start=1):
        body_elem = article.find(".//body")
        if body_elem is None:
            print(f"[WARN] article #{idx} 에는 <body>가 없음 (본문 공개 안됨)")
            continue
        # itertext()를 사용하여 <body> 내부의 모든 텍스트를 재귀적으로 추출.
        # strip()을 사용하여 불필요한 공백 제거.
        body_text = "\n".join(text.strip() for text in body_elem.itertext() if text.strip())
        body_texts.append(body_text)
    
    return "\n\n--- ARTICLE SPLIT ---\n\n".join(body_texts)
# BioPython 라이브러리 : PubMed API와 연동하기 위한 파이썬 라이브러리
# Entrez은 미국 국립생물공학정보센터(NCBI)에서 제공하는 생물의학 데이터베이스에 대해 접근 가능
from Bio import Entrez
# 환경변수에서 API 키와 이메일을 가져오기 위한 라이브러리
import os
import dotenv
# search_term : PubMed DB에 검색을 요청할 때 사용할 검색어 또는 검색 쿼리 문자열
# max_papers : 검색 결과의 최대 개수
def search_PubMed(search_term,max_papers=10):
    # 환경변수에서 이메일과 API 키 읽어오기
    dotenv.load_dotenv()
    Entrez.email = os.getenv("NCBI_Email") # 문제 발생 시 사용자에게 연락하기 위함 - 필수로 입력해야 함
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    # NCBI 서버 입장에서 어떤 프로그램이 요청을 보냈는지 식별하기 위한 이름표임
    Entrez.tool = "PMC_Paper_Search"
    try:
        # 데이터베이스(pmc)에서 지정한 검색어로 검색 요청 - 정렬기준은 관련성
        search_handle = Entrez.esearch(db="pmc", term=search_term, retmax=max_papers,sort="relevance")
        # API의 XML응답을 파싱하여 파이썬 객체(딕셔너리 형태)로 변환
        search_record = Entrez.read(search_handle)
        search_handle.close() # 네트워크 연결 닫기

        pmids = search_record["IdList"] # 검색 결과에서 PMID 추출 - 논문 고유 식별자
        total_count = int(search_record["Count"]) # 검색 결과의 총 개수
        print(f"'{search_term}' 키워드로 총 {total_count}개의 논문 중 {len(pmids)}개 검색됨.")
        if not pmids:
            print(f"'{search_term}' 키워드로 검색된 논문이 없습니다.")
            # 검색 결과가 없을 경우 함수 종료
            exit()
        # 논문 원본 가져오기 
        fetch_handle = Entrez.efetch(db="pmc", id=pmids,rettype="xml",retmode="xml")
        
        xml_data =fetch_handle.read() # XML 데이터 읽기
        fetch_handle.close() # 네트워크 연결 닫기
        xml_text = xml_data.decode("utf-8") # XML 데이터를 UTF-8로 디코딩
        all_body_text = get_pmc_body_text(xml_text)
        print("\n\n=== <BODY> TEXT EXTRACTED ===\n")
        if all_body_text.strip():
            snippet_length = 3000
            print(all_body_text[:snippet_length] + ("... [TRUNCATED] ..." if len(all_body_text) > snippet_length else ""))
        else:
            print("[WARN] <body> 태그가 없거나 본문 텍스트가 비어있습니다.")
        return all_body_text
    except Exception as e:
        print(f"PubMed에서 논문 검색 중 오류 발생: {e}")
        return None
"""
pdf링크로 다운로드한 논문을 읽어와서 본문 텍스트를 추출하는 함수로 해보기
"""
if __name__ == "__main__":
    keyword = "The relationship between Kimchi and COVID-19"
    max_papers = 10  # 최대 검색 결과 수
    #search_arxiv(keyword, max_papers)
    search_PubMed(keyword, max_papers)