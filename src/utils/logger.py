import logging
import sys
from pathlib import Path
# 파일 크기가 커지면 자동으로 파일을 분리해주는 로깅 라이브러리
from logging.handlers import RotatingFileHandler

def setup_logger():
    """
    로그 메시지를 어떤 형식(template)으로 보여줄지 정의하는 Formatter 객체를 생성
    - %(asctime)s: 로그가 기록된 시간
    - %(levelname)s: 로그 레벨(심각도) (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - %(name)s: 로거의 이름(주로 모듈 이름)
    - %(lineno)d: 로그가 기록된 파일의 줄 번호
    - %(message)s: 실제 로그 메시지
    """
    logformatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s'
    )
    # 로거 객체 생성
    logger = logging.getLogger()
    # 로거의 기본 레벨을 DEBUG로 설정 (이보다 낮은 레벨의 로그는 출력되지 않음)
    logger.setLevel(logging.DEBUG)

    # 로그를 콘솔(터미널) 화면으로 보내는 역할을 하는 streamhandler 객체를 만듬
    console_handler = logging.StreamHandler(sys.stdout)
    # 콘솔 핸들러가 사용할 로그 형식을 설정
    console_handler.setFormatter(logformatter)
    # 로거 객체에 콘솔 핸들러를 추가
    logger.addHandler(console_handler)
    
    # 현재 파일의 절대 경로  
    path = Path(__file__).resolve().parents[2]
    log_dir = path /'log'

    # 로그를 파일로 보내는 역할을 하는 RotatingFileHandler 객체를 만듬
    file_handler = RotatingFileHandler(
        # 파일 경로와 이름 지정
        filename= log_dir / 'app.log', 
        # 한개의 로그 파일의 최대 크기 설정
        maxBytes=5*1024*1024, # 5 MB
        # 최대 5개의 백업 파일을 유지
        backupCount=5,
        # 로그 파일이 UTF-8로 인코딩되도록 설정
        encoding='utf-8'
    )
    # 파일 핸들러가 사용할 로그 형식 지정
    file_handler.setFormatter(logformatter)
    # 로거 객체에 파일 핸들러를 추가
    logger.addHandler(file_handler)

    # 외부 라이브러리의 상세 로그 출력을 줄여 전체 흐름을 더 명확하게 볼 수 있도록 설정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.INFO)
    logging.getLogger("arxiv").setLevel(logging.INFO)