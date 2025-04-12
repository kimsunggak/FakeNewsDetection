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
if __name__ == "__main__":
    keyword = "Theory of Relativity"
    max_papers = 10  # 최대 검색 결과 수
    search_arxiv(keyword, max_papers)