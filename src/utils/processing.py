# 원본 텍스트 청크 단위로 분할하기 위한 라이브러리
# langchain으로 rag를 구현할 때 사용하면 호환성이 좋으나 공백,특수문자를 이해한 청크 분할이 가능하므로 랭체인 텍스트 분할 라이브러리 사용
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models
import uuid
import logging

logger = logging.getLogger(__name__)

# 텍스트 청킹 및 임베딩과 관련된 모든 작업
class TextProcessor:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    # 논문 데이터 원본 청크 단위로 분할
    @staticmethod
    def chunk_text(text: str,size: int = 1500,overlap: int=100) -> list[str]:
        if not text:
            return []
        # 텍스트 분할기 선언
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = size,
            chunk_overlap = overlap,
            length_function = len
        )
        chunks = text_splitter.split_text(text)
        logger.debug(f"Created {len(chunks)} chunks.")
        return chunks # [청크1,청크2...] 형식

    def embed_documents(self,chunks) -> list[float]:
        if not chunks:
            logger.warning("embed_documents called with no chunks.")
            return []
        logger.info(f"Embedding {len(chunks)} text chunks...")
        # encode메서드에 청크 텍스트 리스트를 통째로 전달함 -> 라이브러리가 내부적으로 GPU를 활용하여 모든 청크를 처리함
        embedding_vector = self.embedding_model.encode(chunks)
        logger.info("Embedding complete.")
        return embedding_vector

    def claim_evidence_embedding(self,ce: dict)-> list[float]:
        # 주장과 근거 dict -> 하나의 텍스트로 합침
        claim_text = ce.get("claim","")
        evidence_list = ce.get("evidence","")
        evidence_text = " ".join(evidence_list)
        combined_text = f"{claim_text}\n---\n{evidence_text}".strip()
        if not combined_text:
            logger.warning("No claim/evidence text to embed.")
            return []
        logger.debug(f"Text for embedding: '{combined_text[:100]}...'")
        embedding_vector = self.embedding_model.encode(combined_text)
        logger.info("Claim/evidence embedding complete.")
        return embedding_vector
    
    def process_for_qdrant(self,data)-> list[models.PointStruct]:
        points = []
        for paper in data:
            chunks = self.chunk_text(paper.get('body',''))
            if not chunks:
                continue
            embeddings = self.embed_documents(chunks)
            for i,chunk in enumerate(chunks):
                # Qdrant가 요구하는 DB 구조
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=embeddings[i],
                        payload={
                            "paper_id": paper['id'],
                            "title": paper['title'],
                            "date": paper['date'],
                            "chunk_text" : chunk
                        }
                    )
                )
        logger.info(f"Successfully created {len(points)} Qdrant points from {len(data)} papers.")
        return points
