NEWS_CLS = """
    분류 유형:
    0: 감정 변화 (Sentiment Shift). 문장의 긍정적/부정적 감정을 반대로 표현하여 의미를 변화시킬 수 있음.
    1: 강조 수준 변화 (Intensified/Neutralized Sentiment). 문장의 강조 수준을 높이거나 낮추어 의미를 변화시킬 수 있음.
    2: 세부사항 추가/삭제 (Elaborated Details). 문장에 대한 더 많은 정보를 포함시키거나, 빼서 의미를 변화시킬 수 있음.
    3: 미래 예측과 현재 결과 전환 (Plan Realization). 예측된 사건이 실제로 발생했음을 명시하여 의미를 변화시킬 수 있음.
    4: 새로운 정보 도입 (Emerging Situations). 새로운 정보를 추가하여, 의미를 변화시킬 수 있음.
    전부 해당하지 않는 경우엔 빈 리스트를 반환해주세요.

    예시: 
    입력: "Stock prices are rising."
    출력: [0, 1, 2, 3, 4]

    위 유형과 예시를 바탕으로, 다음의 텍스트를 분류해주세요.
    입력: {source_text}
    출력:
    """

RPT_CLS = """
    분류 유형:
    0: Sentiment Shifts (긍/부정 감정 변화). 문장의 긍정적/부정적 감정을 더 강하게 혹은 약하게게 표현하여 의미를 변화시킬 수 있음.
    1: 세부사항 추가 (Elaborated Details). 비즈니스 상황에 대해 추가적인 세부 정보 제공하여 의미를 변화시킬 수 있음.
    2: 계획 실현 (Plan Realization): 예측된 사건이 실제로 발생했음을 명시하여 의미를 변화시킬 수 있음.
    3: 새로운 상황 (Emerging Situations): 새롭게 등장한 비즈니스 상황을 기술하여 의미를 변화시킬 수 있음.
    4: 미시적 관점 vs 거시적 관점 (Micro vs Macro Perspective). 문장의 관점을 미시적 혹은 거시적으로 변경하여 의미를 변화시킬 수 있음.
    5: 사실 vs 의견 (Fact vs Opinion). 사실과 의견을 혼동하거나 구분하여 의미를 변화시킬 수 있음.
    6. 금융 전문 용어 vs 일반 용어 (Financial Jargon vs General Terms). 금융 전문 용어에서 일반 용어로 바꾸거나, 그 반대로 바꿔 의미를 변화시킬 수 있음.
    전부 해당하지 않는 경우엔 빈 리스트를 반환해주세요.

    예시: 
    입력: "The company is doing well financially."
    출력: [0, 1, 2, 3, 4, 5, 6]

    위 유형과 예시를 바탕으로, 다음의 텍스트를 분류해주세요.
    입력: {source_text}
    출력:
    """

LAW_CLS = """
    분류 유형:
    0: 법적 해석 및 판단 변화 (Legal Interpretation and Judgment Shifts). 기존의 법적 해석, 판단기준의 변화를 주어 의미를 변화시킬 수 있음.
    1: 형량/보상 적용 방식의 변화 (Sentencing Application Method Shifts). 문장의 형량, 보상 적용 방식의 변화를 주어 의미를 변화시킬 수 있음.
    2: 규정 및 절차적 명확화 (Procedural and Regulatory Clarifications). 절차, 규정 준수, 적용 범위를 명확화하거나 모호화하여 의미를 변화시킬 수 있음.
    전부 해당하지 않는 경우엔 빈 리스트를 반환해주세요.

    예시: 
    입력: "The court ruled in favor of the plaintiff."
    출력: [0, 1, 2]

    위 유형과 예시를 바탕으로, 다음의 텍스트를 분류해주세요.
    입력: {source_text}
    출력:
    """

DIS_CLS = """
    분류 유형:
    0: Sentiment Shifts (긍/부정 감정 변화). 문장의 긍정적/부정적 감정을 더 강하게 혹은 약하게게 표현하여 의미를 변화시킬 수 있음.
    1: 세부사항 추가 (Elaborated Details). 비즈니스 상황에 대해 추가적인 세부 정보 제공하여 의미를 변화시킬 수 있음.
    2: 계획 실현 (Plan Realization): 예측된 사건이 실제로 발생했음을 명시하여 의미를 변화시킬 수 있음.
    3: 새로운 상황 (Emerging Situations): 새롭게 등장한 비즈니스 상황을 기술하여 의미를 변화시킬 수 있음.
    전부 해당하지 않는 경우엔 빈 리스트를 반환해주세요.

    예시: 
    입력: "The company is doing well financially."
    출력: [0, 1, 2, 3]

    위 유형과 예시를 바탕으로, 다음의 텍스트를 분류해주세요.
    입력: {source_text}
    출력:
    """

NEWS_AUG = {
    0: """
    당신은 금융 언어 변환 전문가로서 부정적인 텍스트를 긍정적으로, 긍정적 텍스트를 부정적으로로 변환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 긍정적/부정적인 표현을 식별하고 이를 부정적/긍정적으로 바꾸는 방법을 고안하십시오.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 의미가 반대여야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    1: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트가 긍정적/부정적인지 파악하고, 더 강한 긍정적 또는 부정적인 표현을 사용하여 텍스트를 변환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.

    1. 비즈니스 운영의 동일한 측면에 초점을 맞추고, 표면적인 단어들은 높은 유사도를 가지지만, 다른 표현을 사용하여 긍정적 또는 부정적인 표현을 강조해야 합니다.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 의미가 반대여야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    2: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트의 비즈니스 상황에 대해 파악하고, 현 상황에 대해 다양한 세부사항을 적용하여 더 자세한 텍스트로 변환하는 역할을 수행합니다. 
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.

    1. 비즈니스 상황에 대해 파악한 내용을 바탕으로, 현 상황에 대해 세부사항을 적용하여 더욱 자세한 텍스트로 만드세요.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 의미가 반대여야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.
    
    다음은 비즈니스 상황의 예시입니다.
    (예시) 
    . 새로운 규제 도입 또는 기존 규제의 변경 
    . 자본 조달 전략(주식 발행, 부채 활용 등) 변화 
    . M&A(인수합병) 및 구조조정 
    . 시장 상황 변화(주가 변동성, 경제 불확실성 등)
    . 고객사의 신용등급 하락 또는 부도 위험 
    . 환율 변동, 무역 분쟁, 경제 제재 

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    3: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 기대하는 미래의 사건에 대하여, 이 사건이 이미 발생했거나 현재 진행 중인 텍스트로 변환하는 역할을 수행합니다. 
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 기대하는 미래의 사건을 식별하고 사건의 시점을 바꿔 텍스트를 변환하세요.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 의미가 반대여야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    4: """
    당신은 금융 언어 변환 전문가로서 새로운 정보를 도입하여, 입력 텍스트와 유사하지만 다른 텍스트를 만드는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 정보에 새로운 요소를 추가하거나, 대체 시나리오를 넣는 방식으로 텍스트 변환을 수행하세요.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 의미가 반대여야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """
}


RPT_AUG = {
    0: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 감정의 강도를 평가한 후, 해당 감정을 더 극단적이거나 더 완화된 형태로 표현하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델의 대조학습을 위한 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 긍정적 또는 부정적인 감정의 정도를 파악하십시오.
    2. 해당 감정의 강도를 더 극단적이거나 더 완화된 형태로 표현하는 3개의 변환된 텍스트를 생성하십시오.
    3. 각 변환된 텍스트는 원본 텍스트와 유사한 구성을 유지하되, 감정 표현의 강도가 달라야 합니다.
    4. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형태로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    5. 반드시 3개의 텍스트가 포함되어야 하며, 출력은 순수 파이썬 리스트여야 합니다.
    6. 변환된 텍스트는 입력 텍스트의 언어적 특성과 문법을 유지해야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    1: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에 비즈니스 상황과 관련된 추가적인 세부 정보를 삽입하여 텍스트를 변환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델의 대조학습을 위한 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 비즈니스 상황을 파악한 후, 그 상황에 어울리는 추가 세부 정보를 고려하십시오.
    2. 추가된 세부 정보가 포함된 3개의 변환된 텍스트를 생성하되, 원본 텍스트의 맥락과 구성을 유지하십시오.
    3. 변환된 텍스트는 파싱 가능한 파이썬 리스트 형태로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 반드시 3개의 텍스트가 포함되어야 하며, 출력은 순수 파이썬 리스트여야 합니다.
    5. 변환된 텍스트는 기존 텍스트의 문법과 어휘 규칙을 준수해야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    2: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 미래에 대한 예측을 실제로 발생한 사건으로 전환하여 서술하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델의 대조학습을 위한 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 예측된 사건이나 계획을 식별한 후, 해당 사건이 이미 발생했거나 진행 중임을 명시하는 방식으로 텍스트를 재구성하십시오.
    2. 3개의 변환된 텍스트를 생성하되, 각 텍스트는 원본 텍스트와 유사한 맥락을 유지해야 합니다.
    3. 변환된 텍스트는 파싱 가능한 파이썬 리스트 형태로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 반드시 3개의 텍스트가 포함되어야 하며, 출력은 순수 파이썬 리스트여야 합니다.
    5. 변환된 텍스트는 입력 텍스트의 언어적 특성을 유지해야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    3: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에 새로운 비즈니스 상황이나 변화를 도입하여 텍스트를 변환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델의 대조학습을 위한 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 현재의 비즈니스 상황을 파악한 후, 새로운 상황이나 환경 변화를 도입하는 방식으로 3개의 변환된 텍스트를 생성하십시오.
    2. 각 변환된 텍스트는 원본 텍스트와 유사한 구성을 유지하면서도 새로운 상황을 반영해야 합니다.
    3. 변환된 텍스트는 파싱 가능한 파이썬 리스트 형태로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 반드시 3개의 텍스트가 포함되어야 하며, 출력은 순수 파이썬 리스트여야 합니다.
    5. 변환된 텍스트는 기존 텍스트의 언어적 특성과 문법을 준수해야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    4: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트의 관점을 미시적 혹은 거시적 관점으로 전환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델의 대조학습을 위한 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 표현된 관점을 파악한 후, 미시적 관점에서 거시적 관점으로 또는 그 반대로 전환하는 3개의 변환된 텍스트를 생성하십시오.
    2. 각 변환된 텍스트는 원본 텍스트와 유사한 맥락을 유지하되, 관점이 명확하게 달라져야 합니다.
    3. 변환된 텍스트는 파싱 가능한 파이썬 리스트 형태로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 반드시 3개의 텍스트가 포함되어야 하며, 출력은 순수 파이썬 리스트여야 합니다.
    5. 변환된 텍스트는 기존 텍스트의 언어적 특성을 유지해야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    5: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 사실과 의견의 구분을 재구성하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델의 대조학습을 위한 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 사실과 의견을 식별한 후, 이들 요소의 구분 또는 혼합 방식을 달리하는 3개의 변환된 텍스트를 생성하십시오.
    2. 각 변환된 텍스트는 원본 텍스트와 유사하지만 사실과 의견의 표현 방식이 달라져야 합니다.
    3. 변환된 텍스트는 파싱 가능한 파이썬 리스트 형태로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 반드시 3개의 텍스트가 포함되어야 하며, 출력은 순수 파이썬 리스트여야 합니다.
    5. 변환된 텍스트는 기존 텍스트의 언어적 규칙과 문법을 준수해야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    6: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 사용된 용어를 금융 전문 용어와 일반 용어 간에 전환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델의 대조학습을 위한 Hard Negative를 만들기 위함입니다.

    1. 입력된 텍스트에서 금융 전문 용어와 일반 용어를 식별한 후, 한쪽에서 다른 쪽으로 용어를 전환하는 방식으로 3개의 변환된 텍스트를 생성하십시오.
    2. 각 변환된 텍스트는 원본 텍스트와 유사한 구조를 유지하되, 용어 선택이 명확히 달라져야 합니다.
    3. 변환된 텍스트는 파싱 가능한 파이썬 리스트 형태로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 반드시 3개의 텍스트가 포함되어야 하며, 출력은 순수 파이썬 리스트여야 합니다.
    5. 변환된 텍스트는 기존 텍스트의 문법과 어휘 규칙을 준수해야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """
}

LAW_AUG = {
    0: """
    당신은 법률 언어 변환 전문가로서, 입력된 텍스트의 기존 법적 해석 및 판단을 반전시키거나 변형하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    법률 언어 변환은 임베딩 모델의 대조학습(Hard Negative 샘플 생성을 위한) 목적으로 사용됩니다.

    1. 입력된 텍스트에서 핵심 법적 해석이나 판결의 요소를 식별한 후, 이를 반대의 논리나 해석으로 전환하여 새로운 법적 결론을 도출하십시오.
    2. 총 3개의 변환된 텍스트를 생성해야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하되 의미가 반전되어야 합니다.
    3. 변환된 텍스트는 파이썬 리스트 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 리스트 형식이어야 합니다.
    5. 변환된 텍스트의 언어는 원본의 법률 용어와 표현 방식을 준수해야 합니다.
    6. 입력: 뒤에 나오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수한 파이썬 리스트만 포함되어야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    1: """
    당신은 법률 언어 변환 전문가로서, 입력된 텍스트에서 형량 또는 보상 적용 방식에 관한 표현을 분석하고,
    이를 더 강화하거나 완화하는 방식으로 재구성하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    법률 언어 변환은 임베딩 모델의 대조학습(Hard Negative 샘플 생성을 위한) 목적으로 사용됩니다.

    1. 입력된 텍스트에서 형량 혹은 보상 관련 요소를 식별한 후, 이를 더 엄격하거나 완화된 방식으로 전환하여 표현하십시오.
    2. 총 3개의 변환된 텍스트를 생성해야 하며, 각 변환된 텍스트는 원본과 유사하지만 의미가 반전되어야 합니다.
    3. 변환된 텍스트는 파이썬 리스트 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트 형식이어야 합니다.
    5. 변환된 텍스트의 언어는 법률 문맥에 맞는 용어와 표현을 유지해야 합니다.
    6. 입력: 뒤에 나오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수한 파이썬 리스트만 포함되어야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    2: """
    당신은 법률 언어 변환 전문가로서, 입력된 텍스트에서 규정 및 절차와 관련된 내용을 파악하고,
    이를 명확하거나 모호한 방식으로 재구성하여 의미를 전환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    법률 언어 변환은 임베딩 모델의 대조학습(Hard Negative 샘플 생성을 위한) 목적으로 사용됩니다.

    1. 입력된 텍스트에서 규정, 절차 또는 관련 용어를 식별한 후, 이를 더 명확하거나 모호한 방식으로 전환하여 표현하십시오.
    2. 총 3개의 변환된 텍스트를 생성해야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 의미가 반전되어야 합니다.
    3. 변환된 텍스트는 파이썬 리스트 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트 형식이어야 합니다.
    5. 변환된 텍스트의 언어는 법률 문맥의 규범과 용어를 준수해야 합니다.
    6. 입력: 뒤에 나오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수한 파이썬 리스트만 포함되어야 합니다.

    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """
}

DIS_AUG = {
    0: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 긍정적 또는 부정적 감정의 강도를 조절하여, 감정을 더 강하게 혹은 약하게 표현하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.
    
    1. 입력된 텍스트에서 긍정적/부정적 감정 표현을 식별하고, 감정의 강도를 높이거나 낮추는 방안을 고안하십시오.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하되 감정의 강도가 달라야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.
    
    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    1: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 비즈니스 상황에 대해 추가적인 세부 정보를 제공하여, 텍스트를 더욱 풍부하게 변환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.
    
    1. 입력된 텍스트의 비즈니스 상황을 분석하고, 관련 세부 정보를 추가하는 방법을 고안하십시오.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 추가 세부 정보가 포함되어야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.
    
    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    2: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에서 예측된 사건이 실제로 발생했거나 진행 중임을 명시하여, 텍스트의 시제를 전환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.
    
    1. 입력된 텍스트에서 미래에 대한 예측을 식별하고, 해당 사건이 실제로 발생했거나 진행 중임을 나타내는 방식으로 텍스트를 변환하십시오.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 시제와 의미가 달라야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.
    
    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """,
    3: """
    당신은 금융 언어 변환 전문가로서 입력된 텍스트에 새로운 비즈니스 상황 또는 대체 시나리오를 도입하여, 텍스트를 변환하는 역할을 수행합니다.
    아래의 지시에 따라 텍스트 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.
    
    1. 입력된 텍스트에서 새로운 비즈니스 상황이나 대체 시나리오를 도입하는 방식을 고안하십시오.
    2. 총 3개의 변환된 텍스트를 만들어야 하며, 각 변환된 텍스트는 원본 텍스트와 유사하지만 의미가 달라야 합니다.
    3. 변환된 텍스트를 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 텍스트1', '변환된 텍스트2', '변환된 텍스트3'].
    4. 텍스트는 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    5. 변환된 텍스트의 언어는 기존 텍스트의 문법과 어휘를 따라야 합니다.
    6. 입력: 뒤에 오는 텍스트를 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.
    
    이제 다음의 텍스트를 변환하세요.
    입력: {source_text}
    출력:
    """
}

ETC_AUG = """
    당신은 금융 언어 변환 전문가로서, 입력된 문장을 거의 유사하지만, 미묘하게 의미가 다른 문장으로 변환하여야 합니다.
    아래의 지시에 따라 문장 변환을 수행하십시오.
    금융 언어 변환은 임베딩 모델을 대조학습하기 위해 Hard Negative를 만들기 위함입니다.
    1. 총 3개의 변환된 문장을 만들어야 하며, 각 변환된 문장은 원본 문장과 유사하지만 의미가 달라야 합니다.
    2. 변환된 문장을 파싱 가능한 파이썬 리스트의 형식으로 제출하십시오: ['변환된 문장1', '변환된 문장2', '변환된 문장3'].
    3. 문장은 반드시 3개여야 하며, 파이썬 리스트의 형태여야 합니다.
    4. 변환된 문장의 언어는 기존 문장의 문법과 어휘를 따라야 합니다.
    5. 입력: 뒤에 오는 문장을 변환하여야 하며, 출력: 이후 답변에는 순수히 파이썬 리스트만 포함되어야 합니다.
    
    이제 다음의 문장을 변환하세요.
    입력: {source_text}
    출력:
    """
