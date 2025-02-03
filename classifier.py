import os
import ast
import re
from openai import OpenAI
from datasets import load_dataset
from prompt_template import NEWS_CLS, RPT_CLS, LAW_CLS, DIS_CLS

# ---------------------------------------------------------------------------
# API 키 설정
# ---------------------------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
hf_api_key = os.environ.get("HF_API_KEY", "your_hf_api_key_here")

client = OpenAI(api_key=api_key)

# ============================================================================
# 데이터셋 불러오기 및 DataFrame 변환
# ============================================================================
dataset = load_dataset("nmixx-fin/nmixx-const", split="train", token=hf_api_key)
df_full = dataset.to_pandas()

# "classification_result" 컬럼이 없으면 추가 (전체 row 수와 동일하게 생성)
if "classification_result" not in df_full.columns:
    df_full["classification_result"] = None

# ============================================================================
# 중간 저장 파일 설정 및 재개를 위한 로딩
# ============================================================================
output_path = "./output/processed.csv"
os.makedirs("./output", exist_ok=True)

start_index = 0
if os.path.exists(output_path):
    try:
        # 저장된 CSV 파일을 읽어 재개 (단, CSV로 저장된 리스트는 문자열로 저장됨)
        df_saved = pd.read_csv(output_path)
        # 이미 처리된 row 수에 맞춰 시작 인덱스를 결정합니다.
        start_index = len(df_saved)
        # 처리된 row에 대해 저장된 결과를 반영 (단, CSV에서는 classification_result가 문자열일 수 있음)
        for idx in range(len(df_saved)):
            df_full.at[idx, "classification_result"] = df_saved.at[idx, "classification_result"]
        print(f"재개: {start_index}번째 row부터 처리합니다.")
    except Exception as e:
        print("저장된 파일을 읽는 중 오류가 발생했습니다. 처음부터 시작합니다:", e)
        start_index = 0

# ============================================================================
# 프롬프트 템플릿 및 시스템 프롬프트 설정
# ============================================================================
prompt_templates = {
    "뉴스": NEWS_CLS,
    "리포트": RPT_CLS,
    "법률": LAW_CLS,
    "공시": DIS_CLS,
}

system_prompt = """
    당신은 전문 금융인으로서, 다음의 텍스트들을 분류하여야 합니다. 당신의 역할은 주어진 텍스트가 주어진 유형들의 의미 변화를 만들 수 있는지 확인하고,
    그 결과를 파이썬 리스트 형태로 반환하여야 하는 것입니다. 이는 Anchor 문장과 밀접하게 관련된 Hard Negative 문장을 만들기 위한 것으로서, 이후 해당 유형에 맞게 생성을 위해 분류하는 작업입니다.
    답변에는 순수히 파이썬 리스트와 유형들의 인덱스(0~N)만 포함되어야 합니다.
"""

# ============================================================================
# OpenAI 응답 텍스트에서 리스트를 파싱하는 함수
# ============================================================================
def parse_list_from_output(text):
    """
    모델 응답 텍스트에서 첫 번째 '['와 마지막 ']' 사이의 문자열을 추출한 후,
    ast.literal_eval을 이용해 파이썬 리스트로 변환합니다.
    실패 시 원본 문자열을 반환합니다.
    """
    if text is None:
        return None
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        list_str = text[start:end+1]
        try:
            return ast.literal_eval(list_str)
        except Exception as e:
            return list_str
    else:
        return None

# ============================================================================
# 한 row 처리 함수 (오류 발생 시 빈 리스트를 반환)
# ============================================================================
def process_row(text, category):
    template = prompt_templates.get(category, "다음 텍스트를 분류해줘: {source_text}")
    prompt = template.format(source_text=text)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # 실제 모델명으로 변경
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        generated_text = completion.choices[0].message.content.strip()
        # 불필요한 접두어(예: "출력:" 등) 제거
        generated_text = re.sub(r'출력\s*:\s*', '', generated_text)
    except Exception as e:
        print(f"텍스트 처리 중 오류: {e}")
        generated_text = None

    parsed_result = parse_list_from_output(generated_text)
    if not isinstance(parsed_result, list):
        # 오류 발생 또는 파싱 실패 시에도 반드시 리스트 형태로 저장
        parsed_result = [generated_text] if generated_text is not None else []
    return parsed_result

# ============================================================================
# DataFrame의 각 row를 순차 처리 (오류 발생 시 건너뛰고, 중간 저장)
# ============================================================================
total_rows = len(df_full)
for i in tqdm(range(start_index, total_rows)):
    try:
        row = df_full.iloc[i]
        result = process_row(row["text"], row["category"])
    except Exception as e:
        print(f"row {i} 처리 중 예외 발생: {e}")
        result = []  # 오류 발생 시 빈 리스트로 기록
    df_full.at[i, "classification_result"] = result
    # 100개 row마다 진행 상황 저장
    if (i + 1) % 100 == 0 or (i + 1) == total_rows:
        df_full.to_csv(output_path, index=False)
        print(f"{i+1}번째 row까지 저장됨.")

print("전체 처리 완료.")

# ============================================================================
# 최종 DataFrame을 Hugging Face 데이터셋으로 변환 후 Hub에 push
# ============================================================================
final_dataset = Dataset.from_pandas(df_full)
final_dataset.push_to_hub("Albertmade/nmixx-const-classified", token=hf_api_key)
