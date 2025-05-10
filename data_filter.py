import re
import pandas as pd
from typing import List, Tuple
from langdetect import detect,LangDetectException


def detect_language(df: pd.DataFrame) -> pd.DataFrame:
    """Detect text language and add 'lang' column safely."""
    print("Detecting language...")
    df = df.copy()
    langs: List[str] = []
    for text in df['text'].fillna(''):
        txt = text.strip()
        if not txt:
            langs.append('und')
        else:
            try:
                langs.append(detect(txt))
            except LangDetectException:
                langs.append('und')
    df['lang'] = langs
    return df

def base_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 데이터프레임을 한국어는 기존 카테고리별 필터로, 영어는 그대로 유지하고
    마지막에 텍스트 길이 조건(<30 또는 >=2000)으로 행을 제거합니다.
    """
    # 1) 언어 감지
    df = detect_language(df)

    # 2) 한국어 / 영어 분리
    ko_df = df[df['lang'] == 'ko']
    en_df = df[df['lang'] == 'en']

    # 한국어 처리: 카테고리별 필터
    ko_news   = news_filter(   ko_df[ko_df['category'] == '뉴스'].copy() )
    ko_law    = law_filter(    ko_df[ko_df['category'] == '법률'].copy() )
    ko_report = report_filter( ko_df[ko_df['category'] == '리포트'].copy() )
    ko_dart   = dart_filter(   ko_df[ko_df['category'] == '공시'].copy() )

    # 영어 처리: 그대로 통과
    en_filtered = en_df.copy()

    # 3) 합치기
    filtered_df = pd.concat([ko_news, ko_law, ko_report, ko_dart, en_filtered],
                             ignore_index=True)

    # 4) 전체 텍스트 길이 제한: 30 미만 또는 2000 이상은 제거
    lengths = filtered_df['text'].str.len().fillna(0)
    mask = (lengths >= 30) & (lengths < 2000)
    filtered_df = filtered_df[mask].reset_index(drop=True)

    return filtered_df

def base_text_cleaning(text: str) -> str:
    """기본 텍스트 정제를 수행하는 유틸리티 함수"""
    # HTML 태그 처리
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]*>", "", text)

    # URL 및 이메일 제거
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    # 공백 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()

    return text


def apply_common_filters(
    df: pd.DataFrame, min_length: int = 50
) -> tuple[pd.DataFrame, List[tuple]]:
    """공통 필터링 로직을 적용하는 유틸리티 함수"""
    print("Applying common filters...")
    filtered_df = df.copy()
    initial_count = len(filtered_df)
    removed_counts = []

    # 1. 기본 필터링
    filtered_df = filtered_df.dropna(subset=["text"])
    filtered_df = filtered_df[filtered_df["text"].str.len() >= min_length]
    removed_counts.append(("기본 필터링", initial_count - len(filtered_df)))
    initial_count = len(filtered_df)

    # 2. 저작권 문구 제거
    copyright_mask = filtered_df["text"].str.contains(
        r"저작권|Copyright|©|All rights reserved", regex=True, case=False
    )
    filtered_df = filtered_df[~copyright_mask]
    removed_counts.append(("저작권 문구 제거", initial_count - len(filtered_df)))
    initial_count = len(filtered_df)

    # 3. 특수문자 반복 제거
    special_char_mask = filtered_df["text"].str.contains(
        r"([^\w\s가-힣])\1{2,}", regex=True
    )
    filtered_df = filtered_df[~special_char_mask]
    removed_counts.append(("특수문자 반복 제거", initial_count - len(filtered_df)))
    initial_count = len(filtered_df)

    # 4. Q&A 형식 제거
    qa_mask = filtered_df["text"].str.contains(
        r"\n\s*Q:|Q[.:]|질문[.:]|\n\s*A:|A[.:]|답변[.:]|^\s*Q:|^\s*A:",
        regex=True,
        case=False,
    )
    filtered_df = filtered_df[~qa_mask]
    removed_counts.append(("Q&A 형식 제거", initial_count - len(filtered_df)))

    # 5. 텍스트 정제
    filtered_df["text"] = filtered_df["text"].apply(base_text_cleaning)
    filtered_df = filtered_df[filtered_df["text"].str.len() > 0]

    return filtered_df, removed_counts


def news_filter(df: pd.DataFrame) -> pd.DataFrame:
    """뉴스 데이터 필터링"""
    print("Filtering news data...")
    # 공통 필터 적용
    filtered_df, _ = apply_common_filters(df, min_length=50)

    # 뉴스 특화 필터링
    # 1. 광고 문구 제거
    ad_mask = filtered_df["text"].str.contains(
        r"광고|click here|지금 가입|무료 체험|[0-9]{2,3}%-[0-9]{2,3}%",
        regex=True,
        case=False,
    )
    filtered_df = filtered_df[~ad_mask]

    # 2. 과도한 줄바꿈 제거
    newline_ratio = filtered_df["text"].str.count("\n") / filtered_df["text"].str.len()
    filtered_df = filtered_df[
        ~((filtered_df["text"].str.count("\n") > 5) & (newline_ratio > 0.05))
    ]

    # 3. 금융 키워드 필터링
    finance_keywords = [
        "금융",
        "주식",
        "투자",
        "은행",
        "펀드",
        "증권",
        "금리",
        "시장",
        "경제",
        "채권",
        "자산",
        "리스크",
        "포트폴리오",
        "인플레이션",
        "환율",
        "재무",
        "회계",
        "보험",
        "대출",
        "신용",
        "거래소",
        "ETF",
        "외환",
        "FX",
        "부동산",
        "파생상품",
        "주가지수",
        "코스피",
        "코스닥",
        "나스닥",
        "다우",
        "S&P",
        "금융위기",
        "통화정책",
    ]

    filtered_df = filtered_df[
        filtered_df["text"]
        .str.lower()
        .apply(lambda x: any(keyword in x for keyword in finance_keywords))
    ]

    return filtered_df


def law_filter(df: pd.DataFrame, strict_mode: bool = True) -> pd.DataFrame:
    """법률 데이터 필터링"""
    print("Filtering legal data...")
    min_length = 100 if strict_mode else 50
    filtered_df, _ = apply_common_filters(df, min_length=min_length)

    # 법률 특화 필터링
    def get_legal_keywords() -> List[str]:
        return [
            # 기본 법률 용어
            "법률",
            "법",
            "조항",
            "판결",
            "판례",
            "소송",
            "계약",
            "손해배상",
            "위헌",
            "합헌",
            # 법원 관련
            "대법원",
            "고등법원",
            "지방법원",
            "법원",
            "재판",
            "판사",
            "검사",
            "변호사",
            # 법률 행위
            "위법",
            "적법",
            "합법",
            "유효",
            "무효",
            "취소",
            "해제",
            "해지",
            "중재",
            "청구",
            # 형사 관련
            "유죄",
            "무죄",
            "실형",
            "집행유예",
            "징역",
            "벌금",
            "구속",
            "기소",
            "공소",
            # 민사 관련
            "소유권",
            "임대차",
            "담보",
            "보증",
            "등기",
            "상속",
            "유언",
            "손해배상",
            # 행정 관련
            "행정",
            "허가",
            "인가",
            "등록",
            "행정처분",
            "행정심판",
            "행정소송",
            "과징금",
        ]

    # 법률 키워드 밀도 계산 및 필터링
    min_density = 0.3 if strict_mode else 0.2
    legal_keywords = get_legal_keywords()

    def calculate_legal_density(text: str) -> float:
        keyword_count = sum(1 for keyword in legal_keywords if keyword in text.lower())
        return keyword_count / (len(text) / 100)  # 100자당 키워드 수

    filtered_df = filtered_df[
        filtered_df["text"].apply(calculate_legal_density) >= min_density
    ]

    # 엄격 모드에서 추가 필터링
    if strict_mode:
        legal_structure_patterns = [
            r"제\d+조",
            r"제\d+항",
            r"제\d+호",
            r"대법원.*\d{4}[가-힣]{1,4}\d+",
            r"「.*」",
            r"『.*』",
            r"주\s*문|이\s*유|판단|결론",
        ]
        filtered_df = filtered_df[
            filtered_df["text"].apply(
                lambda x: any(
                    re.search(pattern, x) for pattern in legal_structure_patterns
                )
            )
        ]

    return filtered_df


def report_filter(df: pd.DataFrame, strict_mode: bool = True) -> pd.DataFrame:
    """리포트 데이터 필터링"""
    print("Filtering report data...")
    min_length = 100 if strict_mode else 50
    filtered_df, _ = apply_common_filters(df, min_length=min_length)

    # 리포트 특화 필터링
    # 1. 코드 블록 제거
    code_mask = filtered_df["text"].str.contains(
        r"```|def |function |class |import |if\s*\(|for\s*\(", regex=True
    )
    filtered_df = filtered_df[~code_mask]

    # 2. 리포트 패턴 검사
    def check_report_patterns(text: str) -> List[str]:
        patterns = []
        pattern_checks = {
            "시각자료": r"표\s*\d+|그림\s*\d+|차트|그래프|Figure|Table|출처|통계",
            "인용참조": r"\(\d{4}\)|et al\.|참고문헌|References|인용|출처:",
            "통계수치": r"\d+\.\d+%|\d+,\d+|\(p\s*<\s*0\.\d+\)|평균|표준편차|증가율",
            "분석비교": r"분석|비교|검토|조사|평가|측정|연구 방법|분석 방법",
            "기간참조": r"\d{4}년|\d{1,2}월|분기|전년 대비|전월 대비",
        }

        for name, pattern in pattern_checks.items():
            if re.search(pattern, text, re.IGNORECASE):
                patterns.append(name)

        return patterns

    min_patterns = 2 if strict_mode else 1
    filtered_df = filtered_df[
        filtered_df["text"].apply(
            lambda x: len(check_report_patterns(x)) >= min_patterns
        )
    ]

    return filtered_df


def dart_filter(df: pd.DataFrame, strict_mode: bool = True) -> pd.DataFrame:
    """공시(DART) 데이터 필터링"""
    print("Filtering DART data...")
    min_length = 200 if strict_mode else 50
    filtered_df, _ = apply_common_filters(df, min_length=min_length)

    # 공시 특화 필터링
    def check_dart_patterns(text: str) -> List[str]:
        patterns = []
        pattern_checks = {
            "공시양식": r"^[\[(【].*공시[\])】]|제\d+기|기업공시|분기보고서|감사보고서",
            "재무정보": r"\d{1,3}(,\d{3})+|\d+\.\d+%|매출액|영업이익|당기순이익|ROE|ROA",
            "회사정보": r"회사명|대표이사|본점소재지|IR담당자|상장주식수|종목코드",
            "주식정보": r"유상증자|무상증자|배당|주당|액면가|자기주식|전환사채",
            "이사회": r"이사회|결의사항|주주총회|감사위원회|사외이사|대표이사",
        }

        for name, pattern in pattern_checks.items():
            if re.search(pattern, text):
                patterns.append(name)

        return patterns

    min_patterns = 3 if strict_mode else 2
    filtered_df = filtered_df[
        filtered_df["text"].apply(lambda x: len(check_dart_patterns(x)) >= min_patterns)
    ]

    # 엄격 모드에서 추가 필터링
    if strict_mode:
        important_patterns = ["공시양식", "재무정보", "주식정보", "이사회"]
        filtered_df = filtered_df[
            filtered_df["text"].apply(
                lambda x: any(
                    pattern in check_dart_patterns(x) for pattern in important_patterns
                )
            )
        ]

    return filtered_df
