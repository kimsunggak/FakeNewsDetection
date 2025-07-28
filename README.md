# YouTube 영상 분석 및 사실 확인 RAG 시스템

## 1. 프로젝트 개요

본 프로젝트는 사용자가 제공한 YouTube 영상의 자막을 분석하여 **핵심 주장(Claim)과 근거(Evidence)를 추출**하고, 이를 **학술 논문 데이터와 비교하여 사실 여부를 검증**하는 Python 기반의 자동화된 팩트체크 파이프라인입니다.

`Ollama`를 통해 로컬 LLM을 유연하게 활용하고, `Qdrant` 벡터 데이터베이스와 `Arxiv` 논문 검색 API를 결합하여 신뢰도 높은 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 것을 목표로 합니다.

## 2. 주요 기능

- **음성-텍스트 변환**: `yt-dlp`와 OpenAI `Whisper` 모델을 사용한 자동 자막 추출
- **핵심 정보 추출**: LLM을 통한 영상 스크립트의 주장/근거 및 검색 키워드 분석
- **데이터 수집 및 처리**: `Arxiv` API를 이용한 관련 논문 검색 및 PDF 텍스트 추출
- **임베딩 및 벡터 저장**: 추출된 논문 데이터를 임베딩하여 `Qdrant` 벡터 데이터베이스에 저장
- **유사도 기반 검색 (RAG)**: 영상의 주장과 관련된 근거 논문을 Qdrant에서 신속하게 검색
- **사실 검증 보고서 생성**: 검색된 논문 데이터를 근거로, LLM이 최종 사실 검증 보고서를 생성
- **사용자 인터페이스**: `Gradio`를 활용한 간단하고 직관적인 웹 UI 제공

## 3. 기술 스택 및 폴더 구조

- **패키지 및 환경 관리**: `uv`, `pyenv`
- **핵심 라이브러리**: `litellm`, `openai`, `qdrant-client`, `sentence-transformers`, `arxiv`, `gradio`
- **데이터베이스**: `Qdrant` (Vector DB)
- **외부 서비스**: `Ollama`, `OpenAI API`

  
## 4. 설치 방법

### 4.1. 사전 준비

- **`pyenv`** 및 **`uv`**가 시스템에 설치되어 있어야 합니다.
- **필수 외부 서비스**: `Ollama`가 로컬에 설치 및 실행 중이어야 하며, `Qdrant` 인스턴스가 실행 중이어야 합니다.
- **`.env` 파일 생성**: 프로젝트 루트에 `.env` 파일을 생성하고 아래 예시를 참고하여 필요한 환경 변수를 입력합니다.

  ```env
  # .env 파일 예시
  OPENAI_API_KEY="sk-..."
  QDRANT_URL="http://localhost:6333"
  QDRANT_API_KEY="your-qdrant-key"
  
  # Ollama 및 모델 설정
  OLlama_API_BASE="http://localhost:11434"
  LLM_MODEL="ollama/llama3.1:8b"
  EMBEDDING_MODEL="allenai/scibert_scivocab_uncased"
  
  # Qdrant 컬렉션 설정
  VECTOR_SIZE=768
  QDRANT_COLLECTION_NAME="arxiv_papers"
  
프로젝트를 처음 설정할 때, 아래의 명령어를 터미널에서 순서대로 실행합니다.

  # 1. Git 리포지토리 클론
git clone [your-repository-url]
cd [project-folder]

# 2. (선택사항) pyenv가 .python-version 파일을 읽어 자동으로 해당 버전으로 전환합니다.
#    만약 버전이 설치되어 있지 않다면 'pyenv install'로 먼저 설치해야 합니다.

# 3. uv를 사용하여 가상환경을 생성합니다. (기본적으로 .venv 폴더 생성)
uv venv

# 4. uv.lock 파일을 기반으로 모든 의존성을 정확하고 빠르게 설치합니다.
uv sync

# 5. 가상환경을 활성화합니다.
source .venv/bin/activate

# 가상환경이 활성화된 상태에서 메인 스크립트 실행
python main.py

6. 향후 확장 계획
본 프로젝트는 다양한 외부 데이터 소스와의 연동을 통해 기능을 강화할 수 있는 유연한 구조로 설계되었습니다.
뉴스 기사/웹사이트 크롤링: BeautifulSoup, Scrapy 등을 활용하여 특정 주제에 대한 최신 뉴스 기사나 공신력 있는 웹사이트의 정보를 수집하고, 이를 사실 검증의 근거로 추가할 수 있습니다.
다른 학술 데이터베이스 연동: PubMed(의학), IEEE Xplore(공학) 등 다른 전문 분야의 데이터베이스 API를 연동하여 검증의 범위를 확장할 수 있습니다.
실시간 소셜 미디어 분석: Twitter API 등을 연동하여 특정 주장에 대한 대중의 반응이나 관련 정보를 실시간으로 수집하고 분석하는 기능을 추가할 수 있습니다.
이러한 데이터 소스들은 src/data/ 디렉토리 내에 새로운 데이터 수집 모듈을 추가하는 방식으로 쉽게 통합할 수 있습니다.
