import json
import os

class Config:
    """JSON 파일로부터 설정값을 로드하여 속성(Attribute) 형태로 변환하는 클래스"""
    @classmethod
    def load_from_json(cls, json_path: str = "config.json") -> None:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ Config 파일을 찾을 수 없습니다: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        # JSON 딕셔너리의 key-value를 Config 클래스 속성으로 자동 할당
        for key, value in config_dict.items():
            setattr(cls, key, value)
            
        print(f"⚙️ [Config 로드 완료] '{json_path}' 파일로부터 설정을 불러왔습니다.")