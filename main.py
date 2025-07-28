import gradio as gr
from gradio.components import Textbox,Markdown
import arxiv

from configs.settings import Settings
from src.utils import PromptManager,YouTubeTranscriber,TextProcessor
from src.data import ArxivCollector
from src.analysis import BaseLLM,TextAnalyzer,FactCheck
from src.database.qdrant import VectorDB
from src.utils.logger import setup_logger

class App():
    def __init__(self):
        self.settings = Settings()
        self.prompt_manager = PromptManager(
            prompt_dir='Prompts'
        )
        self.transcriber = YouTubeTranscriber(
            openai_client=self.settings.openai_client
        )
        self.base_llm = BaseLLM(
            model=self.settings.LLM_MODEL,
            api_base=self.settings.API_BASE,
        )
        self.analyzer  = TextAnalyzer(
            llm=self.base_llm,
            prompt_manager=self.prompt_manager
        )
        self.fact_check = FactCheck(
            llm=self.base_llm,
            prompt_manager=self.prompt_manager
        )
        self.data_collector = ArxivCollector(
            arxiv_client = arxiv.Client()
        )
        self.db_manager = VectorDB(
            client=self.settings.qdrant_client,
            text_processer=TextProcessor(
                embedding_model=self.settings.embedding_model
            ),
            collection_name=self.settings.COLLECTION_NAME,
            vector_size=self.settings.VECTOR_SIZE,
        )

    def run_pipeline(self,url:str):
        if not url:
            yield "URL을 입력해주세요",""
            return
        try:
            yield "(1/6)🔍 유튜브 -> 자막 추출", ""
            transcript = self.transcriber.extract_transcripts(url)

            yield "(2/6) LLM으로 영상의 주장/근거와 키워드를 분석", ""
            claim_evidence = self.analyzer.extract_claim_evidence(transcript)
            keywords = self.analyzer.extract_keywords(claim_evidence)
            
            yield "(3/6) 키워드로 관련 논문 검색", ""
            data = self.data_collector.collect(keywords)
            
            yield "(4/6) 논문 데이터 청크 단위로 분할 및 임베딩", ""
            self.db_manager.setup_qdrant()
            self.db_manager.upload_data(data)
            # 주장과 근거 벡터값과 가장 유사한 데이터 검색

            yield "(5/6) 진위여부 판단을 위한 데이터 탐색", ""
            search_results = self.db_manager.search_data(claim_evidence)

            yield "(6/6) 사실여부 확인중", ""
            answer = self.fact_check.factcheck_llm(claim_evidence, search_results)

            yield "팩트체크 결과", answer

        except Exception as e:
            yield "❌ 오류 발생: " + str(e), ""

    # Gradio UI 구성
    def launch_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🕵️ LiteLLM(OLlama)와 Qdrant로 구현한 유튜브 영상 팩트체크 파이프라인")
            gr.Markdown("유튜브 URL을 입력하면, OLlama가 영상 내용을 분석하고 관련 논문을 찾아 사실 여부를 검증합니다.")
            gr.Markdown("RAG는 Qdrant를 통해 이루어집니다.")

            with gr.Row():
                url_input = Textbox(label="분석할 유튜브 URL", placeholder="http://googleusercontent.com/youtube.com/...")

            submit_button = gr.Button("검증 시작", variant="primary")
            
            gr.Markdown("---")
            
            # 실시간 진행 상황을 표시할 텍스트박스
            status_output = Textbox(label="진행 상황", interactive=False)
            
            # 최종 팩트체크 결과를 표시할 마크다운
            result_output = Markdown(label="최종 검증 결과")

            # 버튼 클릭 이벤트 연결
            submit_button.click(
                fn=self.run_pipeline,
                inputs=url_input,
                outputs=[status_output, result_output]
            )
        demo.launch(share=True)

# Gradio 앱 실행
if __name__ == "__main__":
    setup_logger()  # 로거 설정
    # 앱 인스턴스 생성 및 실행
    app = App()
    app.launch_ui()
