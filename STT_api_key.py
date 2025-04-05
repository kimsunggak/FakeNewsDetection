import subprocess
import os
from glob import glob
import dotenv
from openai import OpenAI
import openai
import json
from extract_keywords import extract_claims_and_evidence, extract_keywords_from_claim_evidence
import csv

def download_audio_from_youtube(youtube_url, output_base="downloaded_audio"):
    """
    주어진 유튜브 영상 URL의 음성 데이터를 원본 형식으로 다운로드하여 저장합니다.

    Args:
        youtube_url: 음성을 다운로드할 유튜브 영상의 URL (문자열).
        output_base: 저장될 파일명 기본값 (확장자는 자동 결정)
    Returns:
        str: 다운로드된 오디오 파일의 경로
    """
    try:
        command = [
            # yt-dlp 호출
            'yt-dlp',
            # 다운로드할 파일 형식 지정 (-f bestaudio: 가장 좋은 오디오 스트림 선택)
            '-f',
            'bestaudio',
            # 출력 파일 이름 지정, 예: downloaded_audio.m4a, downloaded_audio.webm 등
            "-o", f"{output_base}.%(ext)s",
            youtube_url
        ]

        print(f"원본 음성 데이터 다운로드 시작: {youtube_url}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("원본 음성 데이터 다운로드 완료.")

        downloaded_files = glob(f"{output_base}.*")
        if not downloaded_files:
            raise FileNotFoundError("오류: 다운로드된 파일을 찾을 수 없습니다.")
        audio_file_path = downloaded_files[0]
        return audio_file_path

    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("오류: yt-dlp 명령어를 찾을 수 없습니다. 먼저 yt-dlp를 설치해야 합니다.")
        return None

def transcribe_audio_openai(audio_file_path):
    """
    OpenAI의 Whisper API를 사용해 주어진 오디오 파일을 텍스트로 변환(STT)합니다.
    OpenAI 서버에서 모델 처리를 하므로, 로컬 컴퓨팅 자원을 사용하지 않습니다.

    Args:
        audio_file_path (str): 음성 파일 경로
    Returns:
        str: 변환된 텍스트(자막)
    """
    dotenv.load_dotenv()
    client = OpenAI()
    if not os.path.isfile(audio_file_path):
        print("[ERROR] 오디오 파일이 존재하지 않습니다.")
        return ""

    # OpenAI API 키 설정: 환경변수 'OPENAI_API_KEY'가 설정되어 있어야 합니다.
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        print("[ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        return ""

    print("OpenAI Whisper API를 사용하여 음성 -> 텍스트 변환 시작...")
    # 파일을 바이너리 읽기 모드로 열어서 OpenAI API 호출
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model = "whisper-1",
            file = audio_file)
    print("음성 -> 텍스트 변환 완료.")
    return transcript.text

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

# 추출된 데이터 CSV파일로 저장
def save_data_to_csv(video_url,transcript,claim_evidence,keywords,metadata):
    filename = "youtube_data.csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "video_url", "transcript", "claim", "evidence",
            "keywords", "upload_date", "view_count", "like_count", "comment_count"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "video_url": video_url,
            "transcript": transcript,
            "claim": claim_evidence.get("주장", ""),
            # evidence와 keywords는 리스트이므로 JSON 문자열로 저장 (가독성 및 재사용성 때문에)
            "evidence": json.dumps(claim_evidence.get("근거", []), ensure_ascii=False),
            "keywords": json.dumps(keywords.get("keywords", []), ensure_ascii=False),
            "upload_date": metadata.get("upload_date", ""),
            "view_count": metadata.get("view_count", ""),
            "like_count": metadata.get("like_count", ""),
            "comment_count": metadata.get("comment_count", "")
        })
    print(f"데이터가 {filename} 파일에 저장되었습니다.")


if __name__ == "__main__":
    video_url = input("음성을 다운로드할 유튜브 영상 주소를 입력하세요: ")
    audio = download_audio_from_youtube(video_url)
    transcript = None
    claim_evidence = None
    keywords = None
    metadata = None
    if audio:
        transcript = transcribe_audio_openai(audio)
        print("\n=== 변환된 텍스트 ===")
        print(transcript)
    # 텍스트 변환 완료 후, 다운로드된 오디오 파일 삭제
        try:
            os.remove(audio)
            print("다운로드된 오디오 파일을 삭제했습니다.")
        except Exception as e:
            print(f"파일 삭제 중 오류 발생: {e}")
    # 1) 핵심 주장과 근거 추출 (LangChain 활용)
        print("\n=== 핵심 주장 및 근거 추출 ===")
        claim_evidence = extract_claims_and_evidence(transcript)
        print(json.dumps(claim_evidence, ensure_ascii=False, indent=2))
    # 2) 주장/근거에서 키워드 추출 (LangChain 활용)
        print("\n=== 주장/근거에서 키워드 추출 ===")
        keywords = extract_keywords_from_claim_evidence(claim_evidence)
        print(json.dumps(keywords, ensure_ascii=False, indent=2))
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
    if transcript and claim_evidence and keywords and metadata:
        save_data_to_csv(video_url, transcript, claim_evidence, keywords, metadata)
        print("\n=== CSV 파일로 저장 완료 ===")

