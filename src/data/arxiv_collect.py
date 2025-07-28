import arxiv
import tempfile
# pdf에서 텍스트만 추출
import fitz
# 대소문자 구분없이 Reference 내용을 제거하기 위함
import re
from .base_data import BaseData
import logging

logger = logging.getLogger(__name__)

class ArxivCollector(BaseData):
    def __init__(self, arxiv_client :arxiv.Client ,max_result=10):
        self.client = arxiv_client
        self.max_result = max_result

    def collect(self, queries: list[str]) -> list[dict]:
        logger.info(f"Starting paper collection for queries: {queries}")
        all_papers =[]
        for query in queries:
            try:
                logger.info(f"Searching arxiv for query: '{query}'")
                search = arxiv.Search(
                    query=f'all:"{query}"',
                    max_results = self.max_result,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                for paper in self.client.results(search):
                    body_text = self.extract_text_from_pdf(paper)
                    all_papers.append({
                        "id": paper.entry_id,
                        "title": paper.title,
                        "date": paper.published,
                        "body": body_text,
                        "source" : "arxiv"
                    })
            except Exception as e:
                logger.error(f"Failed to collect papers for query '{query}'", exc_info=True)
        logger.info(f"Total {len(all_papers)} papers collected. Normalizing data...")
        return self.normalize(all_papers)
    # self없이 정적 메소드로 정의
    @staticmethod
    def normalize(papers_raw: list[dict]) -> list[dict]:
        normalized = []
        for p in papers_raw:
            pid = p.get("id") or p.get("Id") or p.get("entry_id") 
            title = p.get("Title") or p.get("title")
            date = p.get("date") or p.get("Date") or p.get("published")
            date_str = str(date.date()) if hasattr(date, 'date') else str(date)

            normalized.append({
                "id": pid,
                "title": title,
                "date": date_str,
                "body": p.get("Body") or p.get("body", ""),
                "source": p.get("source", "unknown").lower(),
            })
        return normalized

    # arxiv 객체를 받아 원본 텍스트만 추출
    def extract_text_from_pdf(self,paper: arxiv.Result) -> str:
        if not paper.pdf_url:
            logger.warning(f"No PDF URL found for paper '{paper.title}'. Skipping.")
            return ""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = paper.download_pdf(dirpath=tmp_dir)
                # pdf 파일 열기
                doc = fitz.open(pdf_path)
                full_text = ""
                # 페이지 순회하며 텍스트 추출 및 결합
                for page in doc:
                    full_text += page.get_text()
                doc.close()
                match = re.search(r'\n\s*references\s*\n', full_text, re.IGNORECASE)
                return full_text[:match.start()].strip() if match else full_text
                
        except Exception as e:
            logger.error(f"Failed to process PDF for paper '{paper.title}' (ID: {paper.entry_id})", exc_info=True)
            return ""