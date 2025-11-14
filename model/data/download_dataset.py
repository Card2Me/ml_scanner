"""
Roboflow 데이터셋 다운로드 스크립트
Document Segmentation 데이터셋을 다운로드하고 전처리합니다.
"""
import os
from roboflow import Roboflow

def download_dataset(api_key=None, output_dir="./dataset"):
    """
    Roboflow에서 Document Segmentation 데이터셋 다운로드

    Args:
        api_key: Roboflow API key (없으면 환경변수 ROBOFLOW_API_KEY 사용)
        output_dir: 데이터셋 저장 경로
    """
    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if api_key is None:
            print("⚠️  ROBOFLOW_API_KEY 환경변수를 설정하거나 api_key 인자를 제공하세요.")
            print("Roboflow 계정에서 API 키를 얻을 수 있습니다: https://app.roboflow.com/")
            return None

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("maulvi-zm").project("document-segmentation-j6olp")
        dataset = project.version(2).download("coco-segmentation", location=output_dir)

        print(f"✅ 데이터셋이 {output_dir}에 다운로드되었습니다.")
        return dataset
    except Exception as e:
        print(f"❌ 데이터셋 다운로드 실패: {e}")
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Roboflow 데이터셋 다운로드")
    parser.add_argument("--api-key", type=str, help="Roboflow API key")
    parser.add_argument("--output", type=str, default="./dataset", help="출력 디렉토리")

    args = parser.parse_args()
    download_dataset(args.api_key, args.output)
