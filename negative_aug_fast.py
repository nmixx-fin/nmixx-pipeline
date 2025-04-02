import os
import ast
import re
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset, Dataset
from prompt_template import NEWS_AUG, RPT_AUG, LAW_AUG, DIS_AUG, ETC_AUG
from tqdm import tqdm

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

augmented_rows = []
start_index = 0
if os.path.exists(output_path):
    try:
        df_checkpoint = pd.read_csv(output_path)
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
    반환값은 (aug_index, prompt_text) 튜플들의 리스트.
    """
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
        prompts.append((None, ETC_AUG.format(source_text=source_text)))
    return prompts

def process_single_augmentation(prompt_text):
    """
    주어진 prompt_text에 대해 OpenAI API를 호출하여, 응답에서 파이썬 리스트(세 개의 텍스트)를 파싱
    올바른 리스트(세 개의 문자열)가 나오면 이를 반환하고, 그렇지 않으면 None을 반환.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "응답은 반드시 파이썬 리스트 형식으로만 출력하세요."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=16000,
            temperature=1.0,
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

def process_row(i):
    """
    원본 DataFrame의 i번째 row에 대해 classification_result와 category를 바탕으로
    여러 augmentation prompt를 생성한 후 각각에 대해 API 호출
    """
    row = df_orig.iloc[i]
    category = row["category"]
    source_text = row["text"]
    classification_result = row["classification_result"]

    prompt_tuples = get_aug_prompts(category, classification_result, source_text, fallback=False)
    row_results = []
    for aug_idx, prompt_text in prompt_tuples:
        result = process_single_augmentation(prompt_text)
        # 기본 호출 실패 시 fallback: ETC_AUG 프롬프트로 재시도
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
            "aug_index": aug_idx,
            "hard_negative_1": result[0],
            "hard_negative_2": result[1],
            "hard_negative_3": result[2]
        }
        row_results.append(new_row)
    return row_results

# =============================================================================
# 5. 멀티쓰레딩을 활용하여 각 원본 row에 대해 augmentation 처리 (확장)
# =============================================================================
total_rows = len(df_orig)
results = []
max_workers = 10  # 동시에 처리할 스레드 수 (API rate limit에 따라 조절)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_row, i): i for i in range(start_index, total_rows)}
    for count, future in tqdm(enumerate(as_completed(futures), start=1), total=len(futures)):
        try:
            row_results = future.result()  # 해당 원본 row에서 생성된 augmented row들의 리스트
            results.extend(row_results)
        except Exception as e:
            print(f"[EXCEPTION] row {futures[future]} 처리 중 예외 발생: {e}")
        # 매 100개 원본 row마다 중간 저장
        if count % 100 == 0 or count == len(futures):
            augmented_rows.extend(results)
            df_out = pd.DataFrame(augmented_rows)
            df_out.to_csv(output_path, index=False)
            print(f"[INFO] {count + start_index}번째 원본 row까지 처리 및 저장 (총 {total_rows} row).")
            results = []

# =============================================================================
# 6. 최종 결과를 Hugging Face Hub에 push
# =============================================================================
final_df = pd.DataFrame(augmented_rows)
final_dataset = Dataset.from_pandas(final_df)
final_dataset.push_to_hub("Albertmade/nmixx-const-hard-neg", token=hf_api_key)
print("[INFO] 최종 데이터셋이 Hugging Face Hub에 push되었습니다.")
