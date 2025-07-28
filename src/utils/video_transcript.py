import subprocess, glob, tempfile
import os
import logging

logger = logging.getLogger(__name__)

# URL을 받아 텍스트로 변환하는 객체
class YouTubeTranscriber:
    def __init__(self,openai_client):
        # openai client 저장
        self.client = openai_client
    # yt-dlp 라이브러리로 유튜브 영상 음성 데이터 다운로드
    def download_audio(self,url: str, output: str) :
        logger.info(f"Starting audio download for URL: {url}")
        cmd = [
            "yt-dlp", 
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",  # mp3로 강제 변환
            "--audio-quality", "0",   # 최고 품질
            "-o", f"{output}.%(ext)s", 
            url
        ]
        try:
            #터미널 코드 내에서 명령어 실행
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("yt-dlp audio download completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp execution failed.", exc_info=True)
            raise RuntimeError(f"오디오 다운로드 실패: {e.stderr}")
        
        paths = glob.glob(f"{output}.*")
        return paths[0]

    # 다운받은 음성 데이터를 텍스트로 변환
    def stt(self,audio_path : str) -> str:
        with open(audio_path,"rb") as f:
            stt_model = self.client.audio.transcriptions.create(
                model = "whisper-1",
                file = f
            )
        logger.info("Speech-to-Text completed successfully.")
        return stt_model.text.strip()

    # url을 받아 전체 과정을 실행하고 음성 데이터 삭제
    def extract_transcripts(self,url: str) -> str:
        # 임시 폴더 생성 후 해당 경로가 tmp변수에 저장
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self.download_audio(url,output=f"{tmp}/yt_audio")
            transcript = self.stt(audio_path)
        logger.debug(f"Temporary directory and its contents have been removed.")
        logger.info(f"Successfully extracted transcript for URL: {url}")
        return transcript
