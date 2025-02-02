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