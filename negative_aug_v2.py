import os
import ast
import re
import time
import pandas as pd
from openai import OpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from prompt_template import NEGATIVE_PROMPT_NEWS, NEGATIVE_PROMPT_REPORT

# =============================================================================
# 1. API 키 및 OpenAI 클라이언트 초기화
# =============================================================================
api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
hf_api_key = os.environ.get("HF_API_KEY", "your_hf_api_key_here")
client = OpenAI(api_key=api_key)

# =============================================================================
# 2. 원본 데이터셋 로드 및 DataFrame 변환
# =============================================================================
dataset_path = "./data/nmixx-cont-classified.csv"  # 필요에 따라 경로 수정
df = pd.read_csv(dataset_path)

# 카테고리별 필터링
news_df = df[df["category"] == "뉴스"]
report_df = df[df["category"] == "리포트"]

# 한글이 포함된 텍스트만 필터링 (뉴스)
korean_news_df = news_df[news_df["text"].str.contains("[가-힣]", regex=True)]

# =============================================================================
# 3. 중간 저장 설정 및 체크포인트 로드
# =============================================================================
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
news_output_path = os.path.join(output_dir, "news_data_gen.csv")
report_output_path = os.path.join(output_dir, "report_data_gen.csv")


# =============================================================================
# 4. 헬퍼 함수들
# =============================================================================
def get_neg_response(template, text):
    """
    주어진 템플릿과 텍스트를 바탕으로 OpenAI API를 호출하여,
    TEXT1과 TEXT2 형식으로 응답을 받아 파싱합니다.

    실패 시 재시도하며, 최대 시도 횟수까지 실패하면 None, None을 반환합니다.
    """
    # LLM 모델 설정 및 재시도 로직
    max_retries = 3
    retry_delay = 2  # 초

    for attempt in range(max_retries):
        try:
            # API 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 금융 텍스트를 변환하는 전문가입니다.",
                    },
                    {"role": "user", "content": template.format(input_text=text)},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            response_text = response.choices[0].message.content.strip()

            # 결과 검증
            if (
                "원문은 금융, 정치, 경제, 사회 STS 벤치마크 데이터셋에 적합하지 않습니다."
                in response_text
            ):
                print(response_text)
                return None, None

            # 결과 파싱
            try:
                # TEXT1 추출
                if (
                    "<<<TEXT1>>>" in response_text
                    and "<<<END TEXT1>>>" in response_text
                ):
                    text1 = (
                        response_text.split("<<<TEXT1>>>")[1]
                        .split("<<<END TEXT1>>>")[0]
                        .strip()
                    )
                else:
                    text1 = ""

                # TEXT2 추출
                if (
                    "<<<TEXT2>>>" in response_text
                    and "<<<END TEXT2>>>" in response_text
                ):
                    text2 = (
                        response_text.split("<<<TEXT2>>>")[1]
                        .split("<<<END TEXT2>>>")[0]
                        .strip()
                    )
                else:
                    text2 = ""

                return text1, text2

            except Exception as parse_error:
                # 파싱 오류 발생 시 재시도 준비
                print(f"응답 파싱 오류: {parse_error}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None, None

        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"API 요청 실패 (시도 {attempt+1}/{max_retries}): {str(e)}. {retry_delay}초 후 재시도..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # 지수 백오프
            else:
                print(f"최대 재시도 횟수에 도달했습니다. 오류: {str(e)}")
                return None, None


# =============================================================================
# 5. 뉴스 데이터 처리
# =============================================================================
def process_news_data(sample_size=100, random_seed=42):
    # 샘플링
    sample_news_df = korean_news_df.sample(sample_size, random_state=random_seed)
    news_texts = sample_news_df["text"].tolist()

    # 결과를 저장할 리스트
    news_data = []

    # 각 텍스트에 대해 처리
    for text in tqdm(news_texts, desc="뉴스 데이터 처리 중"):
        text1, text2 = get_neg_response(NEGATIVE_PROMPT_NEWS, text)
        news_data.append({"positive": text1, "negative": text2, "original": text})

    # 데이터프레임으로 변환하여 저장
    news_df_result = pd.DataFrame(news_data)
    news_df_result.to_csv(news_output_path, index=False)
    print(f"뉴스 데이터 생성 완료: {news_output_path}에 저장됨")

    return news_df_result


# =============================================================================
# 6. 리포트 데이터 처리
# =============================================================================
def process_report_data(sample_size=100, random_seed=42):
    # 샘플링
    sample_report_df = report_df.sample(sample_size, random_state=random_seed)
    report_texts = sample_report_df["text"].tolist()

    # 결과를 저장할 리스트
    report_data = []

    # 각 텍스트에 대해 처리
    for text in tqdm(report_texts, desc="리포트 데이터 처리 중"):
        text1, text2 = get_neg_response(NEGATIVE_PROMPT_REPORT, text)
        report_data.append({"positive": text1, "negative": text2, "original": text})

    # 데이터프레임으로 변환하여 저장
    report_df_result = pd.DataFrame(report_data)
    report_df_result.to_csv(report_output_path, index=False)
    print(f"리포트 데이터 생성 완료: {report_output_path}에 저장됨")

    return report_df_result


# =============================================================================
# 7. 메인 실행 코드
# =============================================================================
if __name__ == "__main__":
    print("뉴스 및 리포트 데이터 증강을 시작합니다.")

    # 샘플 크기 설정
    news_sample_size = 100
    report_sample_size = 100

    # 뉴스 데이터 처리
    news_results = process_news_data(sample_size=news_sample_size)

    # 리포트 데이터 처리
    report_results = process_report_data(sample_size=report_sample_size)

    print("데이터 증강이 완료되었습니다.")
