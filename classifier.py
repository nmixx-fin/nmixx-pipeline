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

# ---------------------------------------------------------------------------
# Hugging Face 데이터셋 불러오기 (예: "your_dataset_name", train split)
# ---------------------------------------------------------------------------
dataset = load_dataset("nmixx-fin/nmixx-const", split="train", token=hf_api_key)

# ---------------------------------------------------------------------------

prompt_templates = {
    "뉴스": NEWS_CLS,
    "리포트": RPT_CLS,
    "법률": LAW_CLS,
    "공시": DIS_CLS,
}

# ---------------------------------------------------------------------------
# 시스템 프롬프트: 모델에게 응답은 오직 파이썬 리스트([ ... ]) 형태로만 출력하도록 지시
# ---------------------------------------------------------------------------
system_prompt = """
    당신은 전문 금융인으로서, 다음의 텍스트들을 분류하여야 합니다. 당신의 역할은 주어진 텍스트가 주어진 유형들의 의미 변화를 만들 수 있는지 확인하고,
    그 결과를 파이썬 리스트 형태로 반환하여야 하는 것입니다. 이는 Anchor 문장과 밀접하게 관련된 Hard Negative 문장을 만들기 위한 것으로서, 이후 해당 유형에 맞게 생성을 위해 분류하는 작업입니다.
    답변에는 순수히 파이썬 리스트와 유형들의 인덱스(0~N)만 포함되어야 합니다.

"""

# ---------------------------------------------------------------------------
# OpenAI 응답 텍스트 내에서 리스트를 파싱하는 함수
# ---------------------------------------------------------------------------
def parse_list_from_output(text):
    """
    모델 응답 텍스트에서 첫 번째 '['와 마지막 ']' 사이의 문자열을 추출하고,
    ast.literal_eval을 통해 파이썬 리스트로 변환합니다.
    파싱에 실패하면 추출한 문자열을 그대로 반환합니다.
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

# ---------------------------------------------------------------------------
# 각 row(예제)를 처리하는 함수
# ---------------------------------------------------------------------------
def process_example(example):
    category = example["category"]
    text = example["text"]
    
    template = prompt_templates.get(category, "['error'] 라고 남겨겨")
    prompt = template.format(source_text=text)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,      
            temperature=0.0,     
        )
        generated_text = completion.choices[0].message.content.strip()
        generated_text = re.sub(r'출력력\s*:\s*', '', generated_text)
    except Exception as e:
        print(f"Error processing example with text: {text}\nError: {e}")
        generated_text = None

    parsed_result = parse_list_from_output(generated_text)
    
    if not isinstance(parsed_result, list):
        parsed_result = [generated_text] if generated_text is not None else []
    
    example["classification_result"] = parsed_result
    return example

# ---------------------------------------------------------------------------
# 데이터셋의 각 row에 대해 처리 (row 단위로 OpenAI API 호출)
# ---------------------------------------------------------------------------
new_dataset = dataset.map(process_example, batched=False, desc="Processing examples")


# ---------------------------------------------------------------------------
# 처리된 결과 데이터셋을 Hugging Face Hub에 push (API 키 포함)
# ---------------------------------------------------------------------------
new_dataset.push_to_hub("Albertmade/nmixx-const-classified", token=hf_api_key)
