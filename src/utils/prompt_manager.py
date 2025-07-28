import os
import yaml
import logging

logger = logging.getLogger(__name__)

# 프롬프트를 관리하는 유틸리티 클래스
class PromptManager:
    def __init__(self,prompt_dir):
        self.prompts = self.load_prompt_dir(prompt_dir)

    def load_prompt_dir(self, prompt_dir) -> dict:
        """
        주어진 디렉토리에서 모든 프롬프트 파일을 읽어와 딕셔너리 형태로 저장.
        """
        loaded_prompts = {}
        for filename in os.listdir(prompt_dir):
            # 폴더 안의 모든 yaml 파일을 읽어 딕셔너리로 저장함
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                # 파일 이름에서 확장자를 제거하여 키로 사용
                prompt_key = os.path.splitext(filename)[0]
                # 운영 체제에 맞는 경로를 사용하여 파일 경로를 생성
                file_path = os.path.join(prompt_dir, filename)
                with open(file_path,'r',encoding='utf-8') as file:
                    loaded_prompts[prompt_key] = yaml.safe_load(file)
        logger.info(f"Successfully loaded {len(loaded_prompts)} prompts: {list(loaded_prompts.keys())}")
        return loaded_prompts

    # 프롬프트 파일 이름을 받아서 프롬프트를 가져오는 메소드
    def get_prompt(self, prompt_key):
        try:
            # 프롬프트만 반환 {system: , user: } 형태로
            return self.prompts[prompt_key]
        except (KeyError, ValueError):
            logger.warning(f"Prompt key '{prompt_key}' not found in loaded prompts.")
            raise ValueError(f"프롬프트 '{prompt_key}'를 찾을 수 없습니다. 올바른 형식은 '파일명.프롬프트명'입니다.")