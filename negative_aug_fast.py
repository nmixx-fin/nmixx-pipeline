import os
import ast
import re
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset, Dataset
# Import the new parameterized prompts
from prompt_template import (
    PARAM_NEWS_AUG,
    PARAM_REPORT_AUG,
    PARAM_DART_AUG,
    PARAM_LAW_AUG,
    PARAM_POS_AUG
)
from tqdm import tqdm
import time 

NEWS_AUG_RULES = {
    0: "[감정 반전] 입력 텍스트에서 긍정/부정 표현을 찾아 반대 감정으로 전환하여 Hard Negative 문장을 생성하세요.",
    1: "[감정 강도 조절] 입력 텍스트의 감정 표현 강도를 극단적으로 강화하거나 약화시켜 Hard Negative 문장을 생성하세요.",
    2: "[세부사항 추가/삭제] 입력 텍스트에 추가 정보 또는 불필요한 세부사항을 조정하여 Hard Negative 문장을 생성하세요.",
    3: "[시제 전환] 입력 텍스트의 미래 예측을 이미 발생한 사건으로 전환하여 Hard Negative 문장을 생성하세요.",
    4: "[새로운 정보 도입] 입력 텍스트에 새로운 정보나 대체 시나리오를 추가하여 Hard Negative 문장을 생성하세요.",
}

RPT_AUG_RULES = {
    0: "[감정 조정] 입력 텍스트의 감정 표현을 강화하거나 약화시켜 Hard Negative 문장을 생성하세요.",
    1: "[비즈니스 세부 정보 추가] 입력 텍스트에 비즈니스 관련 추가 세부 정보를 삽입하여 Hard Negative 문장을 생성하세요.",
    2: "[미래 예측 전환] 입력 텍스트의 미래 예측을 실제 사건으로 전환하여 Hard Negative 문장을 생성하세요.",
    3: "[새로운 상황 도입] 입력 텍스트에 변화된 비즈니스 상황이나 새로운 환경을 반영하여 Hard Negative 문장을 생성하세요.",
    4: "[관점 전환] 입력 텍스트의 서술 관점을 미시적/거시적으로 전환하여 Hard Negative 문장을 생성하세요.",
    5: "[사실 vs 의견 재구성] 입력 텍스트 내 사실과 의견의 표현 방식을 재구성하여 Hard Negative 문장을 생성하세요.",
    6: "[금융 용어 전환] 입력 텍스트에서 금융 전문 용어와 일반 용어를 상호 전환하여 Hard Negative 문장을 생성하세요.",
}

LAW_AUG_RULES = {
    0: "[법적 해석 전환] 입력 텍스트의 핵심 법적 해석이나 판단을 반전 또는 변형하여 Hard Negative 문장을 생성하세요.",
    1: "[형량/보상 방식 조정] 입력 텍스트의 형량 또는 보상 적용 방식을 강화하거나 완화하여 Hard Negative 문장을 생성하세요.",
    2: "[규정 및 절차 재구성] 입력 텍스트 내 규정 및 절차 관련 용어와 표현을 명확하거나 모호하게 전환하여 Hard Negative 문장을 생성하세요.",
}

DIS_AUG_RULES = { # Assuming DART uses DIS rules
    0: "[감정 강도 조절] 입력 텍스트의 감정 표현을 조절하여 Hard Negative 문장을 생성하세요.",
    1: "[추가 세부 정보 삽입] 입력 텍스트에 공시 관련 추가 정보를 삽입하여 Hard Negative 문장을 생성하세요.",
    2: "[시제 전환] 입력 텍스트의 미래 예측을 실제 발생한 사건으로 전환하여 Hard Negative 문장을 생성하세요.",
    3: "[새로운 상황 도입] 입력 텍스트에 새로운 비즈니스 상황이나 대체 시나리오를 반영하여 Hard Negative 문장을 생성하세요.",
}

# For Etc category, we might need a fallback description or reuse one
ETC_RULE_DESCRIPTION = "[기본 증강] 입력 텍스트와 매우 유사하되, 미묘하게 의미가 다른 Hard Negative 문장을 생성하세요." # Placeholder

POS_AUG_RULE_DESCRIPTION = """
[동일 의미 재전환] 입력 문장을 표현 방식만 다르게 변환하여, 의미는 동일하게 유지하는 텍스트 쌍을 생성하세요.
아래 네 가지 방법 중 텍스트의 특성과 길이에 맞게 2가지를 선택하여 적용합니다:
  - (1) 한국어 텍스트인 경우, 정확한 의미를 가진 영어 텍스트로 전환.
  - (2) 일반 텍스트인 경우, 마크다운 형식으로 재구성.
  - (3) 마크다운 형식 텍스트인 경우, HTML 형식으로 변환.
  - (4) 위 조건이 부적합하면, 원본 문장을 paraphrasing 방식으로 재구성.
""" # From original POS_AUG

# Mapping category names to rule sets and templates
CATEGORY_MAP = {
    "News": {"rules": NEWS_AUG_RULES, "template": PARAM_NEWS_AUG},
    "Rpt": {"rules": RPT_AUG_RULES, "template": PARAM_REPORT_AUG},
    "Law": {"rules": LAW_AUG_RULES, "template": PARAM_LAW_AUG},
    "Dis": {"rules": DIS_AUG_RULES, "template": PARAM_DART_AUG}, # Mapping Dis to DART template
    "Etc": {"rules": {0: ETC_RULE_DESCRIPTION}, "template": PARAM_NEWS_AUG} # Fallback for Etc - using News template & basic rule
}


# =============================================================================
# 1. API 키 및 OpenAI 클라이언트 초기화
# =============================================================================
api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
hf_api_key = os.environ.get("HF_API_KEY", "your_hf_api_key_here")
client = OpenAI(api_key=api_key)
# Ensure API key is set
if api_key == "your_openai_api_key_here":
    print("[WARNING] OpenAI API Key not found in environment variables. Using placeholder.")
if hf_api_key == "your_hf_api_key_here":
    print("[WARNING] Hugging Face API Key not found in environment variables. Using placeholder.")


# =============================================================================
# 2. 원본 데이터셋 로드 및 DataFrame 변환
# =============================================================================
try:
    dataset = load_dataset("Albertmade/nmixx-const-classified", split="train", token=hf_api_key)
    df_orig = dataset.to_pandas()
    print("[INFO] Successfully loaded dataset from Hugging Face Hub.")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    exit()

def parse_classification_result(cr):
    if isinstance(cr, list):
        return cr
    elif isinstance(cr, str):
        try:
            # Basic cleaning before eval
            cr_cleaned = cr.strip()
            if not (cr_cleaned.startswith('[') and cr_cleaned.endswith(']')):
                 # Handle cases like single numbers or malformed strings if necessary
                 return [] # Or try to parse differently if needed
            return ast.literal_eval(cr_cleaned)
        except Exception:
            return []
    else:
        return []

df_orig["classification_result"] = df_orig["classification_result"].apply(parse_classification_result)
print("[INFO] Parsed classification results.")

# =============================================================================
# 3. 중간 저장 설정 및 체크포인트 로드
# =============================================================================
output_path = "./output/augmented_pairs.csv" # Changed filename
os.makedirs("./output", exist_ok=True)

# augmented_rows: 최종 확장된 row들을 저장할 리스트
augmented_rows = []
processed_orig_indices = set() # Track processed original indices to avoid duplicates on resume

if os.path.exists(output_path):
    try:
        df_checkpoint = pd.read_csv(output_path)
        if 'orig_index' in df_checkpoint.columns and not df_checkpoint.empty:
            processed_orig_indices = set(df_checkpoint["orig_index"].unique())
            augmented_rows = df_checkpoint.to_dict("records")
            print(f"[INFO] 체크포인트 로드 완료: {len(processed_orig_indices)}개의 원본 인덱스 처리됨.")
        else:
             print("[INFO] 체크포인트 파일은 비어있거나 'orig_index' 컬럼이 없습니다. 처음부터 시작합니다.")
    except Exception as e:
        print(f"[WARNING] 체크포인트 로드 중 오류 발생 ({e}). 처음부터 시작합니다.")
        augmented_rows = []
        processed_orig_indices = set()
else:
    print("[INFO] 체크포인트 파일 없음. 처음부터 시작합니다.")

# =============================================================================
# 4. 헬퍼 함수들 (NEW - API Call & Parsing)
# =============================================================================

def parse_text1_text2_from_output(text):
    """
    Parses TEXT1 and TEXT2 from the '<<<TEXT1>>>...<<<END TEXT1>>>\n<<<TEXT2>>>...<<<END TEXT2>>>' format.
    Uses regex for robustness. Returns (text1, text2) tuple or (None, None) if parsing fails.
    Also checks for '부적합'.
    """
    if text is None:
        return None, None

    text = text.strip()
    if text == "부적합":
        return "부적합", None # Special marker for unsuitable

    # Regex to capture content within delimiters, allowing for varying whitespace/newlines
    match = re.search(
        r"<<<TEXT1>>>(.*?)<<<END TEXT1>>>.*<<<TEXT2>>>(.*?)<<<END TEXT2>>>",
        text,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        text1 = match.group(1).strip()
        text2 = match.group(2).strip()
        return text1, text2
    else:
        # Fallback: Maybe the format is slightly off? Try simple splitting.
        parts = text.split("<<<END TEXT1>>>")
        if len(parts) == 2:
            text1_part = parts[0].split("<<<TEXT1>>>")[-1].strip()
            text2_part = parts[1].split("<<<TEXT2>>>")[-1].split("<<<END TEXT2>>>")[0].strip()
            if text1_part and text2_part:
                 return text1_part, text2_part
        print(f"[WARNING] Could not parse output: {text[:200]}...") # Log parsing failure
        return None, None


def call_openai_api(prompt_template, source_text, rule_description, model="gpt-4o-mini", max_retries=2, delay=5):
    """
    Calls the OpenAI API with the given prompt template and parameters.
    Handles retries and parsing of the new format.
    Returns (text1, text2) or ("부적합", None) or (None, None) on failure.
    """
    prompt = prompt_template.format(source_text=source_text, rule_description=rule_description)
    retries = 0
    while retries <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    # No system message needed as format is in user prompt now
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000, # Adjusted max_tokens
                temperature=0.8, # Slightly lower temperature
            )
            output_text = response.choices[0].message.content.strip()
            return parse_text1_text2_from_output(output_text)

        except Exception as e:
            retries += 1
            print(f"[ERROR] API call failed (Attempt {retries}/{max_retries+1}): {e}")
            if retries <= max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Failing this call.")
                return None, None # Failed after retries
    return None, None # Should not be reached, but for safety

# =============================================================================
# 5. 코어 로직 수정 (process_row -> generate_pairs_for_row)
# =============================================================================

def generate_pairs_for_row(i):
    """
    Processes the i-th row of the original DataFrame.
    Generates multiple text pairs (positive and negative based on classification)
    and returns a list of dictionaries, each representing an augmented pair.
    """
    if i in processed_orig_indices: # Skip already processed rows on resume
        return []

    row = df_orig.iloc[i]
    category = row["category"]
    source_text = row["text"]
    # Ensure classification_result is a list, even if empty
    classification_result = row["classification_result"] if isinstance(row["classification_result"], list) else []

    generated_pairs = [] # Store results for this single original row

    # --- 1. Generate Positive Pair ---
    try:
        # print(f"[DEBUG] Processing POS pair for index {i}")
        text1_pos, text2_pos = call_openai_api(
            prompt_template=PARAM_POS_AUG,
            source_text=source_text,
            rule_description=POS_AUG_RULE_DESCRIPTION
        )
        if text1_pos == "부적합":
            print(f"[INFO] Row {i}: Positive augmentation marked '부적합'.")
        elif text1_pos is not None and text2_pos is not None:
            generated_pairs.append({
                "orig_index": i,
                "category": category,
                "source_text": source_text, # Optional: Keep original text for reference
                "rule_index": -1, # Use -1 or specific code for positive pair
                "rule_description": POS_AUG_RULE_DESCRIPTION,
                "pair_type": "positive",
                "text1": text1_pos,
                "text2": text2_pos
            })
            # print(f"[DEBUG] Successfully generated POS pair for index {i}")
        else:
            print(f"[WARNING] Row {i}: Failed to generate valid Positive pair.")
    except Exception as e:
         print(f"[ERROR] Unexpected error during Positive pair generation for row {i}: {e}")


    # --- 2. Generate Negative Pairs based on Classification ---
    if category in CATEGORY_MAP and classification_result: # Only if category is known and classification exists
        category_info = CATEGORY_MAP[category]
        prompt_template = category_info["template"]
        rules_dict = category_info["rules"]

        for rule_index in classification_result:
            if rule_index in rules_dict:
                rule_description = rules_dict[rule_index]
                try:
                    # print(f"[DEBUG] Processing NEG pair for index {i}, rule {rule_index}")
                    text1_neg, text2_neg = call_openai_api(
                        prompt_template=prompt_template,
                        source_text=source_text,
                        rule_description=rule_description
                    )

                    if text1_neg == "부적합":
                        print(f"[INFO] Row {i}, Rule {rule_index}: Negative augmentation marked '부적합'.")
                    elif text1_neg is not None and text2_neg is not None:
                        generated_pairs.append({
                            "orig_index": i,
                            "category": category,
                            "source_text": source_text, # Optional
                            "rule_index": rule_index,
                            "rule_description": rule_description,
                            "pair_type": "negative",
                            "text1": text1_neg,
                            "text2": text2_neg
                        })
                        # print(f"[DEBUG] Successfully generated NEG pair for index {i}, rule {rule_index}")
                    else:
                         print(f"[WARNING] Row {i}, Rule {rule_index}: Failed to generate valid Negative pair.")
                except Exception as e:
                     print(f"[ERROR] Unexpected error during Negative pair generation for row {i}, rule {rule_index}: {e}")

            else:
                print(f"[WARNING] Row {i}: Rule index {rule_index} not found in rules for category '{category}'. Skipping.")
    # else:
    #     print(f"[INFO] Row {i}: Category '{category}' not in MAP or no classification results. Skipping negative pairs.")


    return generated_pairs

# =============================================================================
# 6. 멀티쓰레딩 실행 루프 수정
# =============================================================================
total_rows = len(df_orig)
results_buffer = [] # Temporary buffer before extending main list
max_workers = 10  # Adjust based on API limits and system resources
save_interval = 50 # Save every 50 *original* rows processed

indices_to_process = [i for i in range(total_rows) if i not in processed_orig_indices]
print(f"[INFO] Total original rows to process: {len(indices_to_process)}")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit tasks only for indices not already processed
    futures = {executor.submit(generate_pairs_for_row, i): i for i in indices_to_process}

    for count, future in enumerate(as_completed(futures), 1):
        original_index = futures[future]
        try:
            # generate_pairs_for_row returns a LIST of pairs for the original index
            generated_pairs_list = future.result()
            if generated_pairs_list: # Only add if pairs were generated
                 results_buffer.extend(generated_pairs_list)

        except Exception as e:
            print(f"[EXCEPTION] Row {original_index} 처리 중 예외 발생: {e}")

        # Save based on count of *original rows processed*
        if count % save_interval == 0 or count == len(futures):
            if results_buffer: # Check if buffer has new data
                 augmented_rows.extend(results_buffer) # Add buffer to main list
                 df_out = pd.DataFrame(augmented_rows)
                 try:
                     df_out.to_csv(output_path, index=False)
                     print(f"[INFO] Checkpoint saved. {count}/{len(futures)} original rows processed in this run. Total augmented rows: {len(augmented_rows)}.")
                     results_buffer = [] # Clear buffer after saving
                 except Exception as e:
                     print(f"[ERROR] Failed to save checkpoint: {e}")
            else:
                 print(f"[INFO] Reached save interval, but no new results in buffer to save ({count}/{len(futures)} original rows processed).")


# =============================================================================
# 7. 최종 결과 저장 및 Hugging Face Hub에 push
# =============================================================================
print("[INFO] Augmentation process finished.")
if not augmented_rows:
     print("[WARNING] No augmented rows were generated. Check logs for errors.")
     exit()

final_df = pd.DataFrame(augmented_rows)

# Final save to CSV
try:
    final_df.to_csv(output_path, index=False)
    print(f"[INFO] Final augmented data saved to {output_path}")
except Exception as e:
    print(f"[ERROR] Failed to save final CSV: {e}")


# Push to Hub
try:
    # Sort by original index and rule index for consistency
    final_df = final_df.sort_values(by=['orig_index', 'pair_type', 'rule_index']).reset_index(drop=True)
    final_dataset = Dataset.from_pandas(final_df)
    # Consider creating a new dataset name or version due to schema change
    hub_dataset_name = "Albertmade/nmixx-sts-pairs-generated"
    print(f"[INFO] Pushing final dataset ({len(final_dataset)} rows) to Hugging Face Hub: {hub_dataset_name}")
    final_dataset.push_to_hub(hub_dataset_name, token=hf_api_key)
    print(f"[INFO] 최종 데이터셋이 Hugging Face Hub ({hub_dataset_name})에 push되었습니다.")
except Exception as e:
    print(f"[ERROR] Failed to push dataset to Hugging Face Hub: {e}")