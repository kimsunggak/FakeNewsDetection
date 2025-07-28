import dotenv, os, openai
from sentence_transformers import SentenceTransformer
# 메모이제이션 적용을 위한 데코레이터
from functools import lru_cache
from qdrant_client import QdrantClient

# 환경 변수 로드
dotenv.load_dotenv()

class Settings:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.API_BASE = os.getenv("OLlama_API_BASE")
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

        self.LLM_MODEL = os.getenv("LLM_MODEL")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        self.VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))
        self.COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

    @property
    @lru_cache
    def openai_client(self) -> openai.Client:
        return openai.Client(api_key=self.OPENAI_API_KEY)
    
    # Qdrant 클라이언트 인스턴스 생성
    @property
    @lru_cache
    def qdrant_client(self):
        return QdrantClient(
            url=self.QDRANT_URL,
            api_key=self.QDRANT_API_KEY
        )
    @property
    @lru_cache
    def embedding_model(self) -> SentenceTransformer:
        return SentenceTransformer(self.EMBEDDING_MODEL)