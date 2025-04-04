# 파이썬 코드 내에서 터미널 명령어를 실행할 수 있게 하기 위한 모듈
import subprocess
# STT 모듈
import whisper
from glob import glob
import os
import time  # 추가
import json

def download_audio_from_youtube(youtube_url, output_base="downloaded_audio"):
    try:
        # 고유 파일명 생성
        unique_id = int(time.time())
        output_base = f"{output_base}_{unique_id}"

        # 기존 동일 파일 삭제
        for old_file in glob(f"{output_base}.*"):
            os.remove(old_file)

        command = [
            'yt-dlp',
            '-f', 'bestaudio',
            '--extract-audio',
            '--audio-format', 'wav',  # 확장자 강제 지정
            "-o", f"{output_base}.%(ext)s",
            youtube_url
        ]

        print(f"원본 음성 데이터 다운로드 시작: {youtube_url}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("원본 음성 데이터 다운로드 완료.")

        # 새로 생성된 파일만 선택
        downloaded_file = glob(f"{output_base}.*")
        if not downloaded_file:
            raise FileNotFoundError("오류: 다운로드된 파일을 찾을 수 없습니다.")
        audio_file_path = downloaded_file[0]
        return audio_file_path

    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("오류: yt-dlp 명령어를 찾을 수 없습니다. 먼저 yt-dlp를 설치해야 합니다.")
        return None

def STT(audio_file_path,model_name):
    if not os.path.isfile(audio_file_path):
        print("[ERROR] 오디오 파일이 존재하지 않습니다.")
        return ""
    # whisper 모델을 로드
    model = whisper.load_model(model_name)
    print("음성 -> 텍스트 변환 시작...")
    result = model.transcribe(audio_file_path)
    text = result["text"]
    print("음성 -> 텍스트 변환 완료.")
    return text
# 웹 스크래핑 방식 - API키 가져오는거 대신 사용 (위반되면 API키 사용)
def get_video_metadata(youtube_url):
    try:
        command =[
            'yt-dlp',
            '-j',
            youtube_url
        ]
        print(f"메타데이터 가져오기 : {youtube_url}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        metadata = json.loads(result.stdout)
        return metadata
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print(f"Stderr: {e.stderr}")
        return None

if __name__ == "__main__":
    video_url = input("음성을 다운로드할 유튜브 영상 주소를 입력하세요: ")
    audio = download_audio_from_youtube(video_url)
    if audio:
        transcript = STT(audio, model_name="medium")
        print("변환된 텍스트 :")
        print(transcript)
    # 텍스트 변환 완료 후, 다운로드된 오디오 파일 삭제
        try:
            os.remove(audio)
            print("다운로드된 오디오 파일을 삭제했습니다.")
        except Exception as e:
            print(f"파일 삭제 중 오류 발생: {e}")
    # 메타데이터 가져오기
    metadata = get_video_metadata(video_url)
    if metadata:
        print("\n=== 영상 메타데이터 ===")
        for key, label in [
            ("upload_date", "업로드 날짜"),
            ("view_count", "조회수"),
            ("like_count", "좋아요 수"),
            ("comment_count", "댓글 수")
        ]:
            print(f"{label}: {metadata.get(key, '정보 없음')}")
    else:
        print("메타데이터를 가져오는 데 실패했습니다.")