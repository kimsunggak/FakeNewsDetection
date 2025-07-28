# Qdrant 연결 - > 외부 데이터 저장, 임베딩 벡터 저장, 유사도 측정 목적
from qdrant_client import models,QdrantClient
from src.utils.processing import TextProcessor
import logging

logger = logging.getLogger(__name__)

class VectorDB:
    # Qdrant 컬렉션 생성을 위한 클라이언트와, 컬렉션 이름, 벡터 크기를 초기화
    def __init__(self,client:QdrantClient,text_processer:TextProcessor,collection_name: str,vector_size: int):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.text_processer = text_processer

    # 컬렉션 있는지 확인하고 없으면 생성
    def setup_qdrant(self):
        try:
            # 컬렉션 존재 여부 확인
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config= models.VectorParams(size=self.vector_size,distance=models.Distance.COSINE)
            )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            else:
                logger.info(f"Using existing collection: '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to setup collection '{self.collection_name}'", exc_info=True)
    
    # 수집한 외부 데이터 qdrant에 업로드
    # 데이터가 많을 경우 생길 수 있는 메모리 문제를 방지하기 위해 배치 단위로 업로드
    def upload_data(self,data,batch_size: int = 500):
        # qdrant 컬렉션 구조에 맞게 전처리
        collection = self.text_processer.process_for_qdrant(data)
        if not collection:
            logger.warning("No data to upload after processing.")
            return
        try:
            for i in range(0, len(collection), batch_size):
                batch = collection[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=batch
                )
            logger.info(f"Successfully uploaded {len(collection)} points.")
        except Exception as e:
            logger.error(f"Failed to upload data to collection '{self.collection_name}'", exc_info=True)
            raise RuntimeError(f"데이터 업로드 실패: {e}")

    # 벡터 유사도가 가장 높은 데이터 찾기 (RAG)
    def search_data(self, claim_evidence, limit=5) -> list:
        logger.info(f"Searching for similar data in '{self.collection_name}'")

        query_vector = self.text_processer.claim_evidence_embedding(claim_evidence)
        # qdrant의 search api를 사용하여 유사도 검색 수행
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector = query_vector,
            limit=limit,
            with_payload=True
        )
        logger.info(f"Found {len(search_results)} search results.")
        id_and_chunks = [
            {
                "id": result.payload.get('paper_id'),
                "chunk_text": result.payload.get('chunk_text')
            }
            for result in search_results
        ]
        return id_and_chunks