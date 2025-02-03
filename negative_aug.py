import os
import ast
import re
import pandas as pd
from openai import OpenAI
from datasets import load_dataset, Dataset
from prompt_template import NEWS_AUG, RPT_AUG, LAW_AUG, DIS_AUG, ETC_AUG

# =============================================================================
# 1. API 키 및 OpenAI 클라이언트 초기화
# =============================================================================
api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
hf_api_key = os.environ.get("HF_API_KEY", "your_hf_api_key_here")
client = OpenAI(api_key=api_key)

# =============================================================================
# 2. 원본 데이터셋 로드 (Albertmade/nmixx-const-classified) 및 DataFrame 변환
# =============================================================================
dataset = load_dataset("Albertmade/nmixx-const-classified", split="train", token=hf_api_key)
df_orig = dataset.to_pandas()

def parse_classification_result(cr):
    if isinstance(cr, list):
        return cr
    elif isinstance(cr, str):
        try:
            return ast.literal_eval(cr)
        except Exception:
            return []
    else:
        return []

df_orig["classification_result"] = df_orig["classification_result"].apply(parse_classification_result)

# =============================================================================
# 3. 중간 저장 설정 및 체크포인트 로드
# =============================================================================
output_path = "./output/augmented.csv"
os.makedirs("./output", exist_ok=True)

# augmented_rows: 최종 확장된 row들을 저장할 리스트
augmented_rows = []
start_index = 0
if os.path.exists(output_path):
    try:
        df_checkpoint = pd.read_csv(output_path)
        # 체크포인트 파일에 'orig_index' 컬럼이 있다면, 마지막 원본 row 인덱스를 확인
        if 'orig_index' in df_checkpoint.columns and not df_checkpoint.empty:
            start_index = int(df_checkpoint["orig_index"].max()) + 1
            augmented_rows = df_checkpoint.to_dict("records")
            print(f"[INFO] 체크포인트 존재: 원본 row {start_index}번째부터 재개합니다.")
        else:
            start_index = 0
    except Exception as e:
        print(f"[WARNING] 체크포인트 로드 중 오류 발생 ({e}). 처음부터 시작합니다.")
        start_index = 0

# =============================================================================
# 4. 헬퍼 함수들
# =============================================================================
def parse_list_from_output(text):
    """
    OpenAI 응답 텍스트에서 첫 번째 '['와 마지막 ']' 사이의 문자열을 추출한 후,
    ast.literal_eval()로 파이썬 리스트로 변환합니다.
    실패 시 None을 반환합니다.
    """
    if text is None:
        return None
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        list_str = text[start:end+1]
        try:
            return ast.literal_eval(list_str)
        except Exception:
            return None
    else:
        return None

def get_aug_prompts(category, classification_result, source_text, fallback=False):
    """
    주어진 category와 classification_result(리스트)를 기반으로 augmentation prompt 리스트를 생성합니다.
    - category에 따라 NEWS_AUG, RPT_AUG, LAW_AUG, DIS_AUG 중 하나를 선택합니다.
    - classification_result의 각 요소(숫자)를 int로 변환하여 해당 인덱스의 프롬프트를 사용하고,
      만약 변환이 불가능하거나 해당 키가 없으면 fallback용 ETC_AUG를 사용합니다.
    - fallback 인자가 True이면 무조건 ETC_AUG를 사용합니다.
    
    반환값은 각 augmentation 호출에 대해 (aug_index, prompt_text) 튜플의 리스트입니다.
    """
    # category에 따른 augmentation 프롬프트 딕셔너리 선택
    if category == "뉴스":
        aug_dict = NEWS_AUG
    elif category == "리포트":
        aug_dict = RPT_AUG
    elif category == "법률":
        aug_dict = LAW_AUG
    elif category == "공시":
        aug_dict = DIS_AUG
    else:
        aug_dict = ETC_AUG

    prompts = []
    if isinstance(classification_result, list) and len(classification_result) > 0:
        for elem in classification_result:
            try:
                idx = int(elem)
            except Exception:
                idx = None
            if fallback or idx is None or idx not in aug_dict:
                prompt_template = ETC_AUG
            else:
                prompt_template = aug_dict[idx]
            prompt_text = prompt_template.format(source_text=source_text)
            prompts.append((elem, prompt_text))
    else:
        # classification_result가 없으면 fallback 프롬프트 한 번만 사용
        prompts.append((None, ETC_AUG.format(source_text=source_text)))
    return prompts

def process_single_augmentation(prompt_text):
    """
    주어진 prompt_text에 대해 OpenAI API를 호출하여, 응답에서 파이썬 리스트(세 개의 텍스트)를 파싱합니다.
    올바른 리스트(세 개의 문자열)가 나오면 이를 반환하고, 그렇지 않으면 None을 반환합니다.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 필요시 실제 모델명으로 수정
            messages=[
                {"role": "system", "content": "응답은 반드시 파이썬 리스트 형식으로만 출력하세요."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.0,
        )
        output_text = response.choices[0].message.content.strip()
        parsed = parse_list_from_output(output_text)
        if isinstance(parsed, list) and len(parsed) == 3 and all(isinstance(x, str) for x in parsed):
            return parsed
        else:
            return None
    except Exception as e:
        print(f"[ERROR] augmentation API 호출 중 오류: {e}")
        return None

# =============================================================================
# 5. 각 원본 row에 대해 augmentation 처리 (확장)
# =============================================================================
total_rows = len(df_orig)
for i in range(start_index, total_rows):
    row = df_orig.iloc[i]
    category = row["category"]
    source_text = row["text"]
    classification_result = row["classification_result"]

    # classification_result와 category를 바탕으로 augmentation prompt 목록 생성
    prompt_tuples = get_aug_prompts(category, classification_result, source_text, fallback=False)
    
    # 각 augmentation 호출마다 새로운 row를 생성 (원본 row가 여러 augmented row로 확장됨)
    for aug_idx, prompt_text in prompt_tuples:
        result = process_single_augmentation(prompt_text)
        # fallback 시도: 기본 호출이 실패하면 ETC_AUG 프롬프트로 재시도
        if result is None:
            fallback_prompt = ETC_AUG.format(source_text=source_text)
            result = process_single_augmentation(fallback_prompt)
        if result is None:
            result = [None, None, None]
        new_row = {
            "orig_index": i,
            "category": category,
            "text": source_text,
            "classification_result": str(classification_result),
            "aug_index": aug_idx,  # classification_result에 나온 해당 인덱스 (없으면 None)
            "hard_negative_1": result[0],
            "hard_negative_2": result[1],
            "hard_negative_3": result[2]
        }
        augmented_rows.append(new_row)
    
    # 매 100개의 원본 row마다 중간 저장
    if (i + 1) % 100 == 0 or (i + 1) == total_rows:
        df_out = pd.DataFrame(augmented_rows)
        df_out.to_csv(output_path, index=False)
        print(f"[INFO] 원본 row {i+1}번째까지 처리 및 저장 (총 {total_rows} row).")

print("[INFO] 전체 augmentation 처리 완료.")

# =============================================================================
# 6. 최종 결과를 Hugging Face Hub에 push
# =============================================================================
final_dataset = Dataset.from_pandas(pd.DataFrame(augmented_rows))
final_dataset.push_to_hub("Albertmade/nmixx-const-hard-neg", token=hf_api_key)
print("[INFO] 최종 데이터셋이 Hugging Face Hub에 push되었습니다.")
