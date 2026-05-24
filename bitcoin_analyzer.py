"""
₿ 비트코인 투자 분석 대시보드
- 빗썸(Bithumb) API 현재가 자동 수집
- 기술적·심리·온체인·거시경제·기관동향 지표 통합
- 가중치 기반 예측 가격 산출
실행: streamlit run bitcoin_analyzer.py
"""

import os
import json
import html as html_mod
import xml.etree.ElementTree as ET
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from dotenv import load_dotenv
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

load_dotenv()

# Streamlit Cloud 배포 시: secrets → 환경변수로 자동 동기화
try:
    for _k in ["BITHUMB_ACCESS_KEY", "BITHUMB_SECRET_KEY", "OPENAI_API_KEY", "FRED_API_KEY", "NEWS_API_KEY"]:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

# ══════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="₿ BTC 투자 분석기",
    layout="wide",
    page_icon="₿",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* 라이트 테마 */
    .stApp { background-color: #ffffff; color: #111111; }
    section[data-testid="stSidebar"] { background-color: #f7f7f7; }

    .pred-box {
        border-radius: 14px;
        padding: 1.6rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .scenario-box {
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .indicator-row {
        background: #f0f0f0;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
    }
    hr { border-color: #dddddd; }
    /* 기술 용어 툴팁 — data-tip 속성 사용 (브라우저 기본 흰 박스 제거) */
    abbr[data-tip] {
        text-decoration: underline dotted #aaa;
        cursor: help;
        border-bottom: none;
        position: relative;
    }
    abbr[data-tip]:hover::after {
        content: attr(data-tip);
        position: absolute;
        left: 0;
        top: 1.6em;
        z-index: 9999;
        background: rgba(30,30,30,0.92);
        color: #fff;
        padding: 7px 11px;
        border-radius: 7px;
        font-size: 0.78rem;
        font-weight: 400;
        line-height: 1.55;
        width: 280px;
        white-space: normal;
        word-break: keep-all;
        box-shadow: 0 3px 10px rgba(0,0,0,0.35);
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 인증 (로그인)
# ══════════════════════════════════════════════

def _load_auth_config() -> dict:
    """
    config.yaml에서 인증 설정 로드.
    Streamlit Cloud 배포 시 COOKIE_KEY secret으로 쿠키 서명 키를 오버라이드.
    """
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=SafeLoader)
    except FileNotFoundError:
        st.error(
            "❌ **config.yaml 파일을 찾을 수 없습니다.**\n\n"
            "`python setup_users.py` 를 실행해서 첫 사용자를 등록해주세요."
        )
        st.stop()
    # Streamlit Cloud secrets에서 쿠키 서명 키 오버라이드
    try:
        if "COOKIE_KEY" in st.secrets and st.secrets["COOKIE_KEY"]:
            cfg["cookie"]["key"] = st.secrets["COOKIE_KEY"]
    except Exception:
        pass
    return cfg


def _setup_authenticator() -> stauth.Authenticate:
    """Authenticate 인스턴스 생성 (session_state에 캐시)"""
    if "_authenticator" not in st.session_state:
        cfg = _load_auth_config()
        st.session_state["_authenticator"] = stauth.Authenticate(
            credentials=cfg["credentials"],
            cookie_name=cfg["cookie"]["name"],
            cookie_key=cfg["cookie"]["key"],
            cookie_expiry_days=cfg["cookie"]["expiry_days"],
            auto_hash=False,   # config.yaml에 이미 bcrypt 해시 저장됨
        )
    return st.session_state["_authenticator"]


# ══════════════════════════════════════════════
# 데이터 수집
# ══════════════════════════════════════════════

@st.cache_data(ttl=60)
def get_btc_usd() -> float | None:
    """바이낸스 Spot API — BTC/USDT 현재가"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            timeout=5,
        )
        return float(r.json()["price"])
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_usd_krw() -> float | None:
    """USD/KRW 실시간 환율 (exchangerate-api, 무료)"""
    try:
        r = requests.get(
            "https://open.er-api.com/v6/latest/USD", timeout=6
        )
        return float(r.json()["rates"]["KRW"])
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_bithumb_ticker() -> dict | None:
    """빗썸 BTC 현재가 조회"""
    try:
        res = requests.get(
            "https://api.bithumb.com/public/ticker/BTC_KRW", timeout=6
        )
        j = res.json()
        if j.get("status") == "0000":
            d = j["data"]
            return {
                "current":      float(d["closing_price"]),
                "open":         float(d["opening_price"]),
                "high":         float(d["max_price"]),
                "low":          float(d["min_price"]),
                "volume_24h":   float(d["units_traded_24H"]),
                "change_rate":  float(d["fluctate_rate_24H"]),
                "change_amt":   float(d["fluctate_24H"]),
                "prev_close":   float(d["prev_closing_price"]),
            }
    except Exception as e:
        st.warning(f"빗썸 API 오류: {e}")
    return None


@st.cache_data(ttl=300)
def get_bithumb_ohlcv() -> pd.DataFrame | None:
    """빗썸 일봉 캔들 데이터 (최근 365일로 제한)"""
    try:
        res = requests.get(
            "https://api.bithumb.com/public/candlestick/BTC_KRW/24h", timeout=10
        )
        j = res.json()
        if j.get("status") == "0000":
            df = pd.DataFrame(
                j["data"],
                columns=["timestamp", "open", "close", "high", "low", "volume"],
            )
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(np.int64), unit="ms"
            )
            for c in ["open", "close", "high", "low", "volume"]:
                df[c] = df[c].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            # 최근 365일만 사용 (버튼 범위: 1주/1개월/3개월/6개월/전체=1년)
            cutoff = df["timestamp"].iloc[-1] - pd.Timedelta(days=365)
            return df[df["timestamp"] >= cutoff].reset_index(drop=True)
    except Exception as e:
        st.warning(f"OHLCV 오류: {e}")
    return None


# ── 뉴스 관련 상수 ────────────────────────────────────────────────────────────
_NEWS_QUERIES = {
    "⚖️ 규제·정치": (
        "bitcoin regulation OR crypto law OR SEC crypto OR bitcoin ban OR "
        "crypto policy OR congress bitcoin OR bitcoin legislation"
    ),
    "🏦 기관·투자": (
        "bitcoin ETF OR institutional bitcoin OR MicroStrategy bitcoin OR "
        "bitcoin fund OR BlackRock bitcoin OR hedge fund crypto"
    ),
    "📊 시장·가격": (
        "bitcoin price OR bitcoin bull OR bitcoin ATH OR bitcoin crash OR "
        "bitcoin halving OR bitcoin whale"
    ),
}

_POSITIVE_KW = {
    "adopt", "approve", "approve", "bullish", "surge", "gain", "rally",
    "growth", "positive", "support", "soar", "record", "high", "milestone",
    "integration", "partnership", "launch", "buy", "accumulate", "upgrade",
}
_NEGATIVE_KW = {
    "ban", "reject", "bearish", "fall", "crash", "negative", "restrict",
    "loss", "drop", "plunge", "hack", "scam", "fraud", "penalty", "fine",
    "sell", "outflow", "bankruptcy", "warning", "risk",
}


# ── 기술 용어 툴팁 사전 ───────────────────────────────────────────────────────
_TOOLTIPS: dict[str, str] = {
    "RSI": (
        "RSI (상대강도지수): 0~100 범위. "
        "30 이하 → 과매도(매수 신호), 70 이상 → 과매수(매도 신호). "
        "14일 기준이 표준."
    ),
    "MACD": (
        "MACD (이동평균 수렴·발산): 단기(12일) EMA - 장기(26일) EMA 차이. "
        "히스토그램 양수 → 상승 모멘텀, 음수 → 하락 모멘텀."
    ),
    "MA": (
        "이동평균선 (MA): 일정 기간 종가 평균. "
        "MA20(단기)·MA50(중기)·MA200(장기)가 모두 우상향이면 '정배열'로 강세 신호."
    ),
    "MA200": (
        "MA200 (200일 이동평균): 장기 추세선. "
        "현재가가 MA200 위 → 장기 강세장, MA200 아래 → 장기 약세장 신호."
    ),
    "볼린저밴드": (
        "볼린저 밴드: MA20 ± 2×표준편차 구간. "
        "가격이 하단 접촉 → 과매도, 상단 접촉 → 과매수. "
        "밴드 수축 후 확장 시 큰 변동 예고."
    ),
    "공포탐욕": (
        "Fear & Greed Index: 0(극도 공포) ~ 100(극도 탐욕). "
        "역발상 지표: 극도 공포(0~25) → 매수 기회, 극도 탐욕(75~100) → 매도 주의."
    ),
    "OI": (
        "OI (미결제약정, Open Interest): 청산되지 않은 선물 계약 총수. "
        "OI 증가 + 가격 상승 → 강세 확인. "
        "OI 감소 → 포지션 청산(변동성 주의)."
    ),
    "펀딩레이트": (
        "펀딩 레이트: 선물 롱/숏 포지션 간 정기 지급 비용. "
        "양수(롱이 지불) = 롱 과열 = 단기 하락 위험. "
        "음수(숏이 지불) = 숏 과열 = 단기 반등 가능."
    ),
    "테이커": (
        "테이커 매수/매도 비율: 시장가(공격적) 주문 중 매수/매도 비율. "
        "1.0 초과 → 공격적 매수 우세(강세), 1.0 미만 → 공격적 매도 우세(약세)."
    ),
    "롱숏": (
        "Top Trader 롱/숏 비율: 바이낸스 상위 트레이더(기관·고래)의 롱/숏 계좌 비율. "
        "역발상 지표: 롱 과도(>1.5) → 조정 위험, 숏 과도(<0.7) → 반등 가능."
    ),
    "DXY": (
        "DXY (달러 인덱스): 주요 6개국 통화 대비 달러 강도 지수. "
        "달러 약세(하락) → 비트코인·금 등 위험자산 강세 경향."
    ),
    "해시레이트": (
        "해시레이트: 비트코인 네트워크의 초당 총 연산 처리량(TH/s). "
        "상승 = 채굴자 참여 증가 = 네트워크 신뢰 증가 = 장기 강세 신호."
    ),
    "넷플로우": (
        "거래소 넷플로우: 거래소 입금 - 출금 BTC 수량. "
        "순유출(음수) → BTC 보유 심리 강세. 순유입(양수) → 매도 대기 증가."
    ),
    "금리": (
        "연준 기준금리 (Fed Funds Rate): 미국 연방준비제도 기준금리. "
        "인하 → 유동성 증가 → 위험자산(BTC) 강세. 인상 → 유동성 축소 → 약세 압력."
    ),
    "CPI": (
        "CPI (소비자물가지수): 소비자가 구매하는 상품·서비스의 평균 가격 변화율. "
        "CPI 상승 → 인플레 심화 → 금리인상 압력 → BTC 약세. "
        "CPI 하락 → 금리인하 기대 → 유동성 증가 → BTC 강세."
    ),
    "고용": (
        "고용지표 (실업률/NFP): 미국 노동시장 상황. "
        "실업률 상승·NFP 부진 → 경기침체 우려 → 금리인하 기대 → BTC 긍정. "
        "실업률 하락·NFP 호조 → 고용 과열 → 금리인상 압력 → BTC 부정."
    ),
    "지정학": (
        "지정학적 리스크: 전쟁, 분쟁, 제재 등 정치·군사적 불안정성. "
        "단기: 공포 → BTC 하락(위험자산 회피). "
        "장기: 달러 불신·탈중앙화 수요 → BTC 상승. "
        "이란·러시아 제재 → 해당 국가 BTC 수요 증가 효과도 있음."
    ),
    "클래리티법안": (
        "CLARITY Act (암호화폐 명확화 법안): 미국에서 BTC·ETH 등 암호화폐의 "
        "법적 지위와 규제 관할권을 명확히 하는 법안. "
        "통과 시 기관투자자 진입 장벽 완화 → 대형 호재. "
        "폐기 시 규제 불확실성 지속 → 약세 압력."
    ),
}


def _tt(label: str, key: str) -> str:
    """HTML <abbr data-tip="..."> 태그로 마우스오버 툴팁 반환.
    title 속성 대신 data-tip을 써서 브라우저 기본 흰 박스 툴팁을 제거."""
    tip = _TOOLTIPS.get(key, "")
    if not tip:
        return label
    tip_escaped = tip.replace('"', "&quot;").replace("'", "&#39;")
    return (
        f'<abbr data-tip="{tip_escaped}" '
        f'style="text-decoration:underline dotted #999;cursor:help;">'
        f"{label}</abbr>"
    )


def _news_sentiment(title: str) -> str:
    """뉴스 제목 키워드 기반 단순 감성 판별"""
    words = set(title.lower().split())
    pos = words & _POSITIVE_KW
    neg = words & _NEGATIVE_KW
    if pos and not neg:   return "🟢"
    if neg and not pos:   return "🔴"
    if pos and neg:       return "🟡"
    return "⚪"


@st.cache_data(ttl=1800)   # 30분 캐시
def get_btc_news() -> list[dict]:
    """NewsAPI로 카테고리별 비트코인 뉴스 수집 (RSS 폴백 포함)"""
    api_key = os.getenv("NEWS_API_KEY", "")
    results: list[dict] = []

    # ── 1차: NewsAPI ──────────────────────────
    if api_key:
        for category, query in _NEWS_QUERIES.items():
            try:
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 6,
                        "apiKey": api_key,
                    },
                    timeout=8,
                )
                for art in r.json().get("articles", []):
                    if "[Removed]" in (art.get("title") or ""):
                        continue
                    pub = art.get("publishedAt", "")[:10]
                    results.append({
                        "category":  category,
                        "title":     art.get("title", ""),
                        "source":    art.get("source", {}).get("name", ""),
                        "url":       art.get("url", ""),
                        "date":      pub,
                        "desc":      (art.get("description") or "")[:180],
                        "sentiment": _news_sentiment(art.get("title", "")),
                    })
            except Exception:
                pass

    # ── 2차: RSS 폴백 (API 키 없거나 실패 시) ──
    if not results:
        rss_feeds = [
            ("⚖️ 규제·정치", "https://news.google.com/rss/search?q=bitcoin+regulation+crypto+law&hl=en&gl=US&ceid=US:en"),
            ("🏦 기관·투자", "https://news.google.com/rss/search?q=bitcoin+ETF+institutional+MicroStrategy&hl=en&gl=US&ceid=US:en"),
            ("📊 시장·가격", "https://cointelegraph.com/rss"),
        ]
        for category, url in rss_feeds:
            try:
                r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                root = ET.fromstring(r.content)
                for item in root.findall(".//item")[:6]:
                    title = item.findtext("title", "")
                    link  = item.findtext("link", "")
                    date  = item.findtext("pubDate", "")[:16]
                    source = item.findtext("source", "") or url.split("/")[2]
                    results.append({
                        "category":  category,
                        "title":     title,
                        "source":    source,
                        "url":       link,
                        "date":      date,
                        "desc":      "",
                        "sentiment": _news_sentiment(title),
                    })
            except Exception:
                pass

    # 날짜 내림차순 정렬
    results.sort(key=lambda x: x["date"], reverse=True)

    # ── 제목 한글 번역 (상위 24건) ──────────────
    if results:
        top24 = results[:24]
        raw_titles = [it["title"] for it in top24]
        ko_map = _translate_headlines("\n||||\n".join(raw_titles))
        for it in results:
            it["title_ko"] = ko_map.get(it["title"], "")

    return results


@st.cache_data(ttl=1800)   # 뉴스 제목 조합이 같으면 30분 동안 캐시
def _cached_ai_summary(lines: str) -> dict:
    """캐시 가능한 OpenAI 호출 (문자열 키) — 예외는 캐시되지 않고 전파됨"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    try:
        prompt = f"""당신은 비트코인·암호화폐 시장 전문 애널리스트입니다.
아래 최신 뉴스 목록을 바탕으로 투자자에게 실질적으로 유용한 **상세 한국어 분석 보고서**를 작성하세요.

[뉴스 목록]
{lines}

[작성 지침]
- 각 항목은 **5~8문장** 분량으로 충분히 서술하세요.
- 뉴스에 등장하는 **구체적 수치, 기관명, 인물명, 법안명**을 반드시 인용하세요.
- 단순 나열이 아닌 **뉴스 간의 연관성과 시장 영향**을 분석하세요.
- 투자 판단에 직결되는 **시사점**을 각 항목 끝에 한 문장으로 덧붙이세요.

다음 JSON 형식으로 출력하세요:
{{
  "regulation": "미국·글로벌 규제 동향 상세 분석 — 주요 법안 진행 현황, SEC·CFTC·의회 움직임, 각국 정부 입장 변화, 투자 시사점 포함",
  "institutional": "기관·기업 투자 동향 상세 분석 — ETF 자금 유출입 수치, 주요 기업 BTC 매입/매도 내역, 헤지펀드·은행 포지션 변화, 투자 시사점 포함",
  "market": "시장·가격 동향 상세 분석 — 주요 가격 레벨·지지/저항선, 거래량 변화, 파생상품 시장 동향(OI·펀딩레이트), 투자 시사점 포함",
  "overall": "종합 전망 — 위 세 분야를 통합한 단기(1~2주)·중기(1~3개월) BTC 시장 시나리오, 핵심 촉매 및 반전 조건 포함",
  "signal": "강세 또는 약세 또는 중립",
  "key_risks": "투자자가 반드시 주의해야 할 핵심 리스크 2~3가지를 ① ② ③ 형식으로 서술"
}}"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.25,
            max_tokens=2400,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        raise  # 예외를 다시 던져서 캐시되지 않게 함


@st.cache_data(ttl=1800)   # 제목이 같으면 30분 캐시
def _translate_headlines(titles_str: str) -> dict[str, str]:
    """뉴스 제목 목록을 한국어로 일괄 번역 (GPT-4o-mini) — 예외는 캐시되지 않고 전파됨"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {}   # 키 없으면 번역 없이 진행 (AI 요약과 달리 필수 아님)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    titles = titles_str.split("\n||||\n")
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": (
                    "당신은 금융·경제 뉴스 전문 번역가입니다. "
                    "영어 뉴스 제목을 자연스러운 한국어로 번역하세요. "
                    "번호와 번역만 출력하고 다른 내용은 추가하지 마세요."
                ),
            }, {
                "role": "user",
                "content": numbered,
            }],
            temperature=0.1,
            max_tokens=1200,
        )
    except Exception:
        raise  # 예외를 다시 던져서 캐시되지 않게 함

    result: dict[str, str] = {}
    for line in resp.choices[0].message.content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # "1. 번역된 제목" 형태 파싱
        dot_pos = line.find(". ")
        if dot_pos > 0 and line[:dot_pos].isdigit():
            idx = int(line[:dot_pos]) - 1
            translated = line[dot_pos + 2:].strip()
            if 0 <= idx < len(titles):
                result[titles[idx]] = translated
    return result


def summarize_news_with_ai(news_items: list[dict]) -> dict:
    """뉴스 제목+설명을 문자열로 직렬화 후 캐시된 AI 요약 호출"""
    lines = "\n".join(
        f"[{it['category']}] {it['title']}"
        + (f" — {it['desc'][:80]}" if it.get("desc") else "")
        + f" ({it['source']}, {it['date']})"
        for it in news_items[:24]          # 24건으로 확대
    )
    try:
        return _cached_ai_summary(lines)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600)
def get_bithumb_ohlcv_monthly() -> pd.DataFrame | None:
    """빗썸 일봉 전체 데이터를 월봉으로 리샘플링 (장기 차트용, 1시간 캐시)
    - Bithumb 24h 엔드포인트는 제공 가능한 최대 일봉을 모두 반환 (보통 3000+일 ≈ 8년)
    - pandas resample('ME')로 월봉(시가·고가·저가·종가·거래량) 변환
    """
    try:
        res = requests.get(
            "https://api.bithumb.com/public/candlestick/BTC_KRW/24h", timeout=15
        )
        j = res.json()
        if j.get("status") == "0000":
            df = pd.DataFrame(
                j["data"],
                columns=["timestamp", "open", "close", "high", "low", "volume"],
            )
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(np.int64), unit="ms"
            )
            for c in ["open", "close", "high", "low", "volume"]:
                df[c] = df[c].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)

            # ── 월봉 리샘플링 ──────────────────────────────────────
            monthly = (
                df.set_index("timestamp")
                .resample("ME")           # 월말 기준으로 묶기
                .agg(
                    open=("open",   "first"),
                    high=("high",   "max"),
                    low=("low",     "min"),
                    close=("close", "last"),
                    volume=("volume","sum"),
                )
                .dropna(subset=["open", "close"])
                .reset_index()
            )
            return monthly
    except Exception as e:
        st.warning(f"월봉 데이터 오류: {e}")
    return None


@st.cache_data(ttl=300)
def get_binance_futures_history() -> dict:
    """선물 히스토리 데이터 (기관·고래 추이 분석용, 5분 캐시)
    Binance → 차단 시 Bybit 자동 폴백"""

    # ── Binance 시도 ───────────────────────────────────────────────────────────
    def _binance_history() -> dict:
        base = "https://fapi.binance.com"
        out: dict = {}

        def _df(url: str, params: dict) -> pd.DataFrame:
            return pd.DataFrame(requests.get(url, params=params, timeout=8).json())

        # ① OI 30일 일봉
        try:
            d = _df(f"{base}/futures/data/openInterestHist",
                    {"symbol": "BTCUSDT", "period": "1d", "limit": 30})
            d["ts"] = pd.to_datetime(d["timestamp"].astype(np.int64), unit="ms")
            d["oi"] = d["sumOpenInterest"].astype(float)
            out["oi"] = d[["ts", "oi"]].reset_index(drop=True)
        except Exception:
            out["oi"] = None

        # ② Top Trader L/S 비율 72시간 (1h)
        try:
            d = _df(f"{base}/futures/data/topLongShortAccountRatio",
                    {"symbol": "BTCUSDT", "period": "1h", "limit": 72})
            d["ts"]        = pd.to_datetime(d["timestamp"].astype(np.int64), unit="ms")
            d["ls_ratio"]  = d["longShortRatio"].astype(float)
            d["long_pct"]  = d["longAccount"].astype(float)
            d["short_pct"] = d["shortAccount"].astype(float)
            out["ls"] = d[["ts", "ls_ratio", "long_pct", "short_pct"]].reset_index(drop=True)
        except Exception:
            out["ls"] = None

        # ③ 테이커 매수/매도 비율 72시간 (1h)
        try:
            d = _df(f"{base}/futures/data/takerlongshortRatio",
                    {"symbol": "BTCUSDT", "period": "1h", "limit": 72})
            d["ts"]    = pd.to_datetime(d["timestamp"].astype(np.int64), unit="ms")
            d["ratio"] = d["buySellRatio"].astype(float)
            d["buy_v"] = d["buyVol"].astype(float)
            d["sel_v"] = d["sellVol"].astype(float)
            out["taker"] = d[["ts", "ratio", "buy_v", "sel_v"]].reset_index(drop=True)
        except Exception:
            out["taker"] = None

        # ④ 펀딩 레이트 최근 30회
        try:
            d = _df(f"{base}/fapi/v1/fundingRate",
                    {"symbol": "BTCUSDT", "limit": 30})
            d["ts"]   = pd.to_datetime(d["fundingTime"].astype(np.int64), unit="ms")
            d["rate"] = d["fundingRate"].astype(float) * 100
            out["funding"] = d[["ts", "rate"]].reset_index(drop=True)
        except Exception:
            out["funding"] = None

        # Binance가 차단되면 모든 항목이 None → 폴백 트리거
        if all(v is None for v in out.values()):
            raise RuntimeError("Binance blocked")
        return out

    # ── Bybit 폴백 ────────────────────────────────────────────────────────────
    def _bybit_history() -> dict:
        base = "https://api.bybit.com/v5"
        out: dict = {}

        # ① OI 30일 일봉
        try:
            r = requests.get(f"{base}/market/open-interest",
                params={"category": "linear", "symbol": "BTCUSDT",
                        "intervalTime": "1d", "limit": 30}, timeout=8)
            items = r.json()["result"]["list"]
            df = pd.DataFrame(items)
            df["ts"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
            df["oi"] = df["openInterest"].astype(float)
            out["oi"] = df[["ts", "oi"]].sort_values("ts").reset_index(drop=True)
        except Exception:
            out["oi"] = None

        # ② L/S 비율 72시간 (1h) — Bybit account-ratio
        try:
            r = requests.get(f"{base}/market/account-ratio",
                params={"category": "linear", "symbol": "BTCUSDT",
                        "period": "1h", "limit": 72}, timeout=8)
            items = r.json()["result"]["list"]
            df = pd.DataFrame(items)
            df["ts"]       = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
            df["long_pct"] = df["buyRatio"].astype(float)
            df["short_pct"]= df["sellRatio"].astype(float)
            df["ls_ratio"] = df["long_pct"] / df["short_pct"].replace(0, 1)
            out["ls"] = df[["ts", "ls_ratio", "long_pct", "short_pct"]].sort_values("ts").reset_index(drop=True)
        except Exception:
            out["ls"] = None

        # ③ 테이커 비율 — Bybit 공개 API 미지원, None 유지
        out["taker"] = None

        # ④ 펀딩 레이트 히스토리
        try:
            r = requests.get(f"{base}/market/funding/history",
                params={"category": "linear", "symbol": "BTCUSDT", "limit": 30},
                timeout=8)
            items = r.json()["result"]["list"]
            df = pd.DataFrame(items)
            df["ts"]   = pd.to_datetime(df["fundingRateTimestamp"].astype(np.int64), unit="ms")
            df["rate"] = df["fundingRate"].astype(float) * 100
            out["funding"] = df[["ts", "rate"]].sort_values("ts").reset_index(drop=True)
        except Exception:
            out["funding"] = None

        return out

    # ── 실행 순서: Binance → Bybit ───────────────────────────────────────────
    try:
        return _binance_history()
    except Exception:
        return _bybit_history()


@st.cache_data(ttl=300)
def get_binance_futures() -> dict | None:
    """선물 공개 API — 기관·고래 동향 자동 수집 (API 키 불필요)
    Binance → 차단 시 Bybit 자동 폴백"""

    # ── Binance 시도 ───────────────────────────────────────────────────────────
    def _from_binance() -> dict:
        base = "https://fapi.binance.com"

        r_ls = requests.get(
            f"{base}/futures/data/topLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 2}, timeout=6,
        )
        ls_data  = r_ls.json()
        ls_ratio = float(ls_data[-1]["longShortRatio"]) if ls_data else 1.0

        r_fr = requests.get(
            f"{base}/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"}, timeout=6,
        )
        funding_rate = float(r_fr.json().get("lastFundingRate", 0))

        r_tk = requests.get(
            f"{base}/futures/data/takerlongshortRatio",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 2}, timeout=6,
        )
        tk_data     = r_tk.json()
        taker_ratio = float(tk_data[-1]["buySellRatio"]) if tk_data else 1.0

        r_oi = requests.get(
            f"{base}/futures/data/openInterestHist",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 25}, timeout=6,
        )
        oi_data       = r_oi.json()
        oi_now        = float(oi_data[-1]["sumOpenInterest"]) if oi_data else 0
        oi_prev       = float(oi_data[0]["sumOpenInterest"])  if oi_data else 0
        oi_change_pct = (oi_now - oi_prev) / oi_prev * 100   if oi_prev else 0

        return {
            "ls_ratio":      ls_ratio,
            "funding_rate":  funding_rate,
            "taker_ratio":   taker_ratio,
            "oi_change_pct": oi_change_pct,
            "oi_now":        oi_now,
            "source":        "Binance",
        }

    # ── Bybit 폴백 ────────────────────────────────────────────────────────────
    def _from_bybit() -> dict:
        base = "https://api.bybit.com/v5"

        # 티커 (펀딩 레이트 포함)
        r_tick = requests.get(
            f"{base}/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"}, timeout=6,
        )
        tick         = r_tick.json()["result"]["list"][0]
        funding_rate = float(tick.get("fundingRate", 0))

        # L/S 비율
        r_ls   = requests.get(
            f"{base}/market/account-ratio",
            params={"category": "linear", "symbol": "BTCUSDT",
                    "period": "1h", "limit": 2}, timeout=6,
        )
        ls_items  = r_ls.json()["result"]["list"]
        buy_ratio = float(ls_items[0]["buyRatio"])
        sel_ratio = float(ls_items[0]["sellRatio"])
        ls_ratio  = buy_ratio / sel_ratio if sel_ratio > 0 else 1.0

        # OI 변화율 (25시간)
        r_oi = requests.get(
            f"{base}/market/open-interest",
            params={"category": "linear", "symbol": "BTCUSDT",
                    "intervalTime": "1h", "limit": 25}, timeout=6,
        )
        oi_items      = r_oi.json()["result"]["list"]   # 최신이 [0]
        oi_now        = float(oi_items[0]["openInterest"])
        oi_prev       = float(oi_items[-1]["openInterest"])
        oi_change_pct = (oi_now - oi_prev) / oi_prev * 100 if oi_prev else 0

        return {
            "ls_ratio":      ls_ratio,
            "funding_rate":  funding_rate,
            "taker_ratio":   1.0,          # Bybit 공개 API 미지원 → 중립
            "oi_change_pct": oi_change_pct,
            "oi_now":        oi_now,
            "source":        "Bybit",
        }

    # ── 실행 순서: Binance → Bybit ───────────────────────────────────────────
    try:
        return _from_binance()
    except Exception:
        try:
            return _from_bybit()
        except Exception:
            return None


def signal_ls_ratio(ratio: float) -> float:
    """Top Trader 롱/숏 비율 → 역발상 신호
    과도한 롱(ratio↑) → 스마트머니 숏 준비 → 하락 신호"""
    if ratio >= 2.0:  return -1.0
    if ratio >= 1.5:  return -0.4 - (ratio - 1.5) / 0.5 * 0.6
    if ratio >= 1.1:  return -(ratio - 1.1) / 0.4 * 0.4
    if ratio <= 0.5:  return  1.0
    if ratio <= 0.8:  return  0.4 + (0.8 - ratio) / 0.3 * 0.6
    if ratio <= 0.9:  return  (0.9 - ratio) / 0.1 * 0.4
    return 0.0


def signal_funding(rate: float) -> float:
    """펀딩 레이트 → 역발상 신호
    높은 양수 펀딩(롱 과열) → 하락 신호"""
    pct = rate * 100  # % 단위
    if pct >=  0.05:  return -1.0
    if pct >=  0.01:  return -pct / 0.05 * 0.7
    if pct <= -0.05:  return  1.0
    if pct <= -0.01:  return  abs(pct) / 0.05 * 0.7
    return 0.0


def signal_taker(ratio: float) -> float:
    """테이커 매수/매도 비율 → 직접 방향 신호
    ratio > 1 = 매수 우세 = 강세"""
    if ratio >= 1.8:  return  1.0
    if ratio >= 1.2:  return  0.3 + (ratio - 1.2) / 0.6 * 0.7
    if ratio >= 1.0:  return  (ratio - 1.0) / 0.2 * 0.3
    if ratio <= 0.4:  return -1.0
    if ratio <= 0.8:  return -0.3 - (0.8 - ratio) / 0.4 * 0.7
    if ratio <= 1.0:  return -(1.0 - ratio) / 0.2 * 0.3
    return 0.0


def signal_oi_change(oi_change_pct: float, price_change_pct: float) -> float:
    """OI 변화율 + 가격 방향 결합 신호
    OI증가 & 가격상승 = 강세 확인 / OI증가 & 가격하락 = 약세 확인"""
    # OI 변화 강도 (-1 ~ +1)
    oi_intensity = max(min(oi_change_pct / 5.0, 1.0), -1.0)
    # 가격 방향 부호
    price_sign = 1 if price_change_pct >= 0 else -1
    if abs(oi_change_pct) < 0.5:   # OI 변화 미미
        return 0.0
    return oi_intensity * price_sign * 0.8


@st.cache_data(ttl=3600)
def get_fear_greed() -> tuple[int, str]:
    """Alternative.me 공포/탐욕 지수"""
    try:
        res = requests.get(
            "https://api.alternative.me/fng/?limit=1", timeout=5
        )
        d = res.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return 50, "Neutral"


@st.cache_data(ttl=3600)
def get_fred_macro() -> dict:
    """FRED API — CPI·연방금리·실업률 자동 수집 및 BTC 신호 변환 (1시간 캐시)
    FRED_API_KEY 없으면 빈 dict 반환 → 수동 슬라이더로 폴백
    """
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return {}

    def _fetch(series_id: str, limit: int = 3) -> list[float]:
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": limit,
                },
                timeout=8,
            )
            return [
                float(o["value"])
                for o in r.json().get("observations", [])
                if o["value"] != "."
            ]
        except Exception:
            return []

    out: dict = {}

    # ① CPI (소비자물가지수, CPIAUCSL, 월별)
    cpi = _fetch("CPIAUCSL", 3)
    if len(cpi) >= 2:
        change = (cpi[0] - cpi[1]) / cpi[1] * 100   # 전월 대비 변화율 %
        out["cpi_value"]  = cpi[0]
        out["cpi_change"] = change
        # 인플레 높으면 금리인상 → BTC 하락 / 낮으면 완화 기대 → BTC 상승
        if change > 0.5:    out["cpi_signal"] = -1.0
        elif change > 0.3:  out["cpi_signal"] = -0.5
        elif change > 0.1:  out["cpi_signal"] = -0.2
        elif change < -0.1: out["cpi_signal"] =  0.5
        else:               out["cpi_signal"] =  0.2

    # ② 연방기준금리 (FEDFUNDS, 월별)
    rate = _fetch("FEDFUNDS", 3)
    if len(rate) >= 2:
        change = rate[0] - rate[1]
        out["rate_value"]  = rate[0]
        out["rate_change"] = change
        if change >= 0.5:    out["rate_signal"] = -1.0
        elif change > 0:     out["rate_signal"] = -0.5
        elif change <= -0.5: out["rate_signal"] =  1.0
        elif change < 0:     out["rate_signal"] =  0.5
        else:                out["rate_signal"] =  0.0

    # ③ 실업률 (UNRATE, 월별)
    unemp = _fetch("UNRATE", 3)
    if len(unemp) >= 2:
        change = unemp[0] - unemp[1]
        out["unemp_value"]  = unemp[0]
        out["unemp_change"] = change
        # 실업↑ → 경기침체 → 금리인하 기대 → BTC 상승 / 실업↓ → 고용과열 → BTC 하락
        if change >= 0.3:    out["emp_signal"] =  0.5
        elif change > 0.0:   out["emp_signal"] =  0.2
        elif change <= -0.3: out["emp_signal"] = -0.5
        elif change < 0.0:   out["emp_signal"] = -0.2
        else:                out["emp_signal"] =  0.0

    return out


@st.cache_data(ttl=3600)
def get_geo_news_signal() -> dict:
    """지정학 리스크 뉴스 수집 → OpenAI LLM 감성 분석 → BTC 영향 점수 (1시간 캐시)
    NEWS_API_KEY 없으면 RSS 폴백, OPENAI_API_KEY 없으면 키워드 폴백
    """
    news_key   = os.getenv("NEWS_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    GEO_QUERIES = [
        "Iran war OR Iran sanctions OR Iran nuclear bitcoin",
        "Russia Ukraine war crypto OR Russia sanctions bitcoin",
        "geopolitical risk bitcoin OR war crypto market",
        "China Taiwan OR Middle East conflict oil bitcoin",
    ]

    headlines: list[str] = []

    # ── 1차: NewsAPI ──────────────────────────────
    if news_key:
        for q in GEO_QUERIES:
            try:
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": q, "language": "en",
                        "sortBy": "publishedAt", "pageSize": 3,
                        "apiKey": news_key,
                    },
                    timeout=8,
                )
                for art in r.json().get("articles", []):
                    t = art.get("title", "")
                    if t and "[Removed]" not in t:
                        headlines.append(t)
            except Exception:
                pass

    # ── 2차: RSS 폴백 ─────────────────────────────
    if not headlines:
        try:
            r = requests.get(
                "https://news.google.com/rss/search?q=geopolitical+risk+bitcoin+war&hl=en&gl=US&ceid=US:en",
                timeout=8, headers={"User-Agent": "Mozilla/5.0"},
            )
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:8]:
                t = item.findtext("title", "")
                if t:
                    headlines.append(t)
        except Exception:
            pass

    if not headlines:
        return {"signal": 0.0, "summary": "뉴스 없음 — 수동 설정 사용", "headlines": []}

    # ── OpenAI LLM 점수화 ─────────────────────────
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            bullet = "\n".join(f"- {h}" for h in headlines[:10])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content":
                    f"""다음 지정학 뉴스들이 비트코인 가격에 미치는 단기 영향을 분석하세요.

뉴스 목록:
{bullet}

다음 JSON 형식으로만 답변하세요:
{{
  "signal": -1.0에서 +1.0 사이 숫자 (전쟁심화·제재강화=음수, 협상·완화=양수, 혼조=0에 가깝게),
  "summary": "비트코인에 미치는 영향 2문장 한국어 요약",
  "key_event": "가장 중요한 이벤트 한 문장 한국어"
}}"""
                }],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=300,
            )
            result = json.loads(resp.choices[0].message.content)
            result["headlines"] = headlines[:10]
            return result
        except Exception:
            pass

    # ── 키워드 폴백 ───────────────────────────────
    neg_kw = {"war", "attack", "sanction", "nuclear", "missile",
              "conflict", "crisis", "bomb", "invasion", "threat"}
    pos_kw = {"ceasefire", "peace", "agreement", "deal",
              "withdraw", "calm", "resolve", "diplomacy", "talks"}
    score = 0.0
    for h in headlines[:10]:
        words = set(h.lower().split())
        score -= len(words & neg_kw) * 0.15
        score += len(words & pos_kw) * 0.15
    return {
        "signal": max(-1.0, min(1.0, score)),
        "summary": "키워드 기반 분석 (LLM 미사용)",
        "key_event": "",
        "headlines": headlines[:10],
    }


# ══════════════════════════════════════════════
# 기술 지표 계산
# ══════════════════════════════════════════════

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calc_bollinger(
    series: pd.Series, period: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + k * std, ma, ma - k * std


# ══════════════════════════════════════════════
# 지표 → 신호 변환  (-1.0 ~ +1.0)
# ══════════════════════════════════════════════

def signal_rsi(rsi: float) -> float:
    """RSI 30 이하 → +1(매수), 70 이상 → -1(매도)"""
    if rsi <= 20:  return  1.0
    if rsi <= 30:  return  0.6 + (30 - rsi) / 10 * 0.4
    if rsi >= 80:  return -1.0
    if rsi >= 70:  return -0.6 - (rsi - 70) / 10 * 0.4
    # 30~70 선형 보간
    return (50 - rsi) / 20 * 0.6


def signal_macd(hist: float, macd: float, sig: float) -> float:
    """히스토그램 방향과 크기로 신호 산출"""
    ref = max(abs(macd), abs(sig), 1.0)
    raw = hist / ref
    return max(min(raw, 1.0), -1.0)


def signal_ma(price: float, ma20: float, ma50: float, ma200: float) -> float:
    """이동평균선 배열 상태로 신호 산출"""
    pts = 0.0
    pts += 0.35 * (1 if price > ma200 else -1)
    pts += 0.30 * (1 if price > ma50  else -1)
    pts += 0.25 * (1 if price > ma20  else -1)
    pts += 0.10 * (1 if ma20  > ma50  else -1)   # 정배열 여부
    return max(min(pts, 1.0), -1.0)


def signal_bb(price: float, upper: float, mid: float, lower: float) -> float:
    """볼린저 밴드 위치로 신호 산출"""
    band_half = (upper - lower) / 2
    if band_half == 0:
        return 0.0
    return max(min((mid - price) / band_half, 1.0), -1.0)


def signal_fg(fg: int) -> float:
    """공포탐욕 역발상 지표  0(극공포)→+1  100(극탐욕)→-1"""
    return (50 - fg) / 50.0


# ══════════════════════════════════════════════
# 예측 계산
# ══════════════════════════════════════════════

def predict(
    current_price: float,
    signals_weights: list[tuple[float, float]],
    volatility_pct: float,
) -> tuple[float, float, float]:
    """
    Returns (predicted_price, change_pct, composite_signal)
    """
    total_w = sum(w for _, w in signals_weights)
    if total_w == 0:
        return current_price, 0.0, 0.0
    composite = sum(s * w for s, w in signals_weights) / total_w
    change = composite * (volatility_pct / 100)
    return current_price * (1 + change), change * 100, composite


# ══════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════

def signal_color(s: float) -> str:
    if s >  0.15: return "#26c77a"
    if s < -0.15: return "#ef5350"
    return "#ffa726"


def signal_label(s: float) -> str:
    if s >= 0.6:  return "🚀 강세"
    if s >= 0.2:  return "📈 약세상"
    if s <= -0.6: return "📉 강하락"
    if s <= -0.2: return "↘ 약하락"
    return "⚖️ 중립"


def krw(v: float) -> str:
    return f"₩{v:,.0f}"


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════

def main():
    # ══════════════════════════════════════════
    # 로그인 인증
    # ══════════════════════════════════════════
    authenticator = _setup_authenticator()

    # 로그인 폼 렌더링 (쿠키가 유효하면 자동 로그인, 아니면 폼 표시)
    authenticator.login(
        location="main",
        max_login_attempts=5,
        fields={
            "Form name": "₿ BTC 투자 분석기",
            "Username":  "아이디",
            "Password":  "비밀번호",
            "Login":     "로그인",
        },
        key="login_widget",
    )

    auth_status = st.session_state.get("authentication_status")

    if auth_status is False:
        st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")
        st.stop()

    if auth_status is None:
        # 로그인 폼만 보이고 나머지는 렌더링하지 않음
        st.stop()

    # ── 로그아웃 (사이드바) ───────────────────
    with st.sidebar:
        st.markdown(f"👤 **{st.session_state.get('name', '')}**")
        st.caption(f"@{st.session_state.get('username', '')}")
        st.markdown("---")
        authenticator.logout(
            button_name="🚪 로그아웃",
            location="sidebar",
            key="logout_btn",
        )

    # ══════════════════════════════════════════
    # 대시보드 메인
    # ══════════════════════════════════════════

    # ── 헤더 ──────────────────────────────────
    hdr_l, hdr_r = st.columns([7, 3])
    with hdr_l:
        st.title("₿ 비트코인 투자 분석 대시보드")
        st.caption("빗썸(Bithumb) 실시간 연동 · 다중 지표 가중치 기반 예측")
    with hdr_r:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # ── 데이터 로딩 ───────────────────────────
    with st.spinner("빗썸 · 바이낸스 API에서 데이터 불러오는 중…"):
        ticker  = get_bithumb_ticker()
        btc_usd = get_btc_usd()
        usd_krw = get_usd_krw()
        df_raw  = get_bithumb_ohlcv()
        fg_val, fg_cls = get_fear_greed()
        futures = get_binance_futures()
        fred    = get_fred_macro()         # FRED: CPI·금리·실업률 (키 없으면 {})
        geo     = get_geo_news_signal()    # 지정학 뉴스 LLM 분석

    if ticker is None or df_raw is None:
        st.error("❌ 데이터 수집 실패. 네트워크를 확인하거나 새로고침 해주세요.")
        st.stop()

    price = ticker["current"]

    # ── 기술 지표 계산 ─────────────────────────
    df = df_raw.copy()
    df["RSI14"] = calc_rsi(df["close"], 14)
    df["RSI7"]  = calc_rsi(df["close"],  7)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = calc_macd(df["close"])
    df["MA20"]  = df["close"].rolling(20).mean()
    df["MA50"]  = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    df["BB_u"], df["BB_m"], df["BB_l"] = calc_bollinger(df["close"])

    row = df.iloc[-1]
    rsi_v    = row["RSI14"]
    macd_v   = row["MACD"]
    macd_s   = row["MACD_sig"]
    macd_h   = row["MACD_hist"]
    ma20, ma50, ma200 = row["MA20"], row["MA50"], row["MA200"]
    bb_u, bb_m, bb_l  = row["BB_u"],  row["BB_m"],  row["BB_l"]

    # 자동 신호 — 기술·심리 지표
    s_rsi  = signal_rsi(rsi_v)
    s_macd = signal_macd(macd_h, macd_v, macd_s)
    s_ma   = signal_ma(price, ma20, ma50, ma200)
    s_bb   = signal_bb(price, bb_u, bb_m, bb_l)
    s_fg   = signal_fg(fg_val)

    # 자동 신호 — 기관·고래 (바이낸스 선물 데이터)
    if futures:
        s_ls_ratio = signal_ls_ratio(futures["ls_ratio"])
        s_funding  = signal_funding(futures["funding_rate"])
        s_taker    = signal_taker(futures["taker_ratio"])
        s_oi       = signal_oi_change(futures["oi_change_pct"], ticker["change_rate"])
        # 기관 신호 = 롱/숏 비율(역발상) + OI 변화 결합
        s_inst_auto  = (s_ls_ratio * 0.55 + s_oi * 0.45)
        # 고래 신호 = 테이커 비율(방향) + 펀딩 레이트(역발상)
        s_whale_auto = (s_taker * 0.60 + s_funding * 0.40)
    else:
        s_ls_ratio = s_funding = s_taker = s_oi = 0.0
        s_inst_auto = s_whale_auto = 0.0

    # 자동 신호 — FRED 거시경제 (API 키 있을 때만, 없으면 0 → 수동 슬라이더 사용)
    s_cpi_auto  = fred.get("cpi_signal",  None)  # None = 데이터 없음
    s_rate_auto = fred.get("rate_signal", None)
    s_emp_auto  = fred.get("emp_signal",  None)

    # 자동 신호 — 지정학 리스크 (뉴스 LLM)
    s_geo_auto = geo.get("signal", 0.0)

    # ══════════════════════════════════════════
    # 환율 정보 배너
    # ══════════════════════════════════════════
    if usd_krw and btc_usd and price:
        btc_impl_rate = price / btc_usd          # BTC 가격으로 역산한 환율
        kimchi = (btc_impl_rate / usd_krw - 1) * 100  # 김치 프리미엄
        kimchi_color  = "#e03131" if kimchi > 0 else "#26a65b"
        kimchi_symbol = "▲" if kimchi > 0 else "▼"
        st.markdown(
            f"""
            <div style="background:#f8f9fa; border:1px solid #e0e0e0;
                        border-radius:8px; padding:0.45rem 1rem;
                        display:flex; gap:2rem; align-items:center;
                        font-size:0.82rem; color:#444; margin-bottom:0.6rem;">
                <span>💱 <b>USD/KRW</b> &nbsp;{usd_krw:,.1f} 원</span>
                <span>🔄 <b>BTC 환산 환율</b> &nbsp;{btc_impl_rate:,.1f} 원</span>
                <span style="color:{kimchi_color};">
                    🌶 <b>김치 프리미엄</b> &nbsp;{kimchi_symbol}{abs(kimchi):.2f}%
                </span>
                <span style="margin-left:auto; color:#aaa; font-size:0.75rem;">
                    환율 1h 캐시 · 빗썸/바이낸스 기준
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════
    # 상단 KPI 카드
    # ══════════════════════════════════════════
    # ── 공통 KPI 카드 렌더러 ─────────────────
    def _kpi(label: str, value: str, sub: str = "", delta: str = "", delta_color: str = "#555") -> str:
        sub_html   = f"<p style='font-size:0.8rem;color:#666;margin:2px 0 0 0;'>{sub}</p>" if sub else ""
        delta_html = f"<p style='font-size:0.8rem;color:{delta_color};margin:5px 0 0 0;'>{delta}</p>" if delta else ""
        return (
            f"<div style='padding:6px 0 10px 0;'>"
            f"<p style='font-size:0.76rem;color:#888;margin:0;font-weight:500;'>{label}</p>"
            f"<p style='font-size:1.6rem;font-weight:700;margin:4px 0 0 0;line-height:1.15;'>{value}</p>"
            f"{sub_html}{delta_html}"
            f"</div>"
        )

    chg_emoji = "📈" if ticker["change_rate"] >= 0 else "📉"
    chg_color = "#26a65b" if ticker["change_rate"] >= 0 else "#e03131"

    rsi_delta_txt   = ("🟢 과매도" if rsi_v < 30 else "🔴 과매수" if rsi_v > 70 else "🟡 중립")
    rsi_delta_color = "#26a65b" if rsi_v < 30 else "#e03131" if rsi_v > 70 else "#b07d00"

    ma_above   = price > ma200
    ma_txt     = "🟢 MA200 위" if ma_above else "🔴 MA200 아래"
    ma_color   = "#26a65b" if ma_above else "#e03131"

    fg_color   = "#26a65b" if fg_val <= 30 else "#e03131" if fg_val >= 70 else "#b07d00"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(_kpi("₿ 현재가", krw(price),
                     sub=f"(${btc_usd:,.0f})" if btc_usd else "",
                     delta=f"{chg_emoji} {ticker['change_rate']:+.2f}%",
                     delta_color=chg_color), unsafe_allow_html=True)
    k2.markdown(_kpi("24h 고가", krw(ticker["high"])), unsafe_allow_html=True)
    k3.markdown(_kpi("24h 저가", krw(ticker["low"])),  unsafe_allow_html=True)
    k4.markdown(_kpi(_tt("RSI(14)", "RSI"), f"{rsi_v:.1f}",
                     delta=rsi_delta_txt, delta_color=rsi_delta_color), unsafe_allow_html=True)
    k5.markdown(_kpi(_tt("공포/탐욕", "공포탐욕"), str(fg_val),
                     delta=fg_cls, delta_color=fg_color), unsafe_allow_html=True)
    k6.markdown(_kpi(_tt("MA200 비교", "MA200"), krw(ma200),
                     delta=ma_txt, delta_color=ma_color), unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════
    # 차트  (단기 일봉 / 장기 월봉 자동 전환)
    # ══════════════════════════════════════════
    # 단기 → 일봉 + RSI + MACD  /  장기 → 월봉 + 이동평균만
    _SHORT_PERIODS = {"1주": 7, "1개월": 30, "3개월": 90, "6개월": 180, "1년": 365}
    _LONG_PERIODS  = {"5년": 60, "10년": 120, "전체": None}   # 월봉 개수
    _period_labels = list(_SHORT_PERIODS.keys()) + list(_LONG_PERIODS.keys())

    st.markdown(
        "<p style='margin:0 0 4px 0;font-size:0.8rem;color:#888;'>📅 차트 기간</p>",
        unsafe_allow_html=True,
    )
    _sel = st.radio(
        "차트기간",
        _period_labels,
        index=2,           # 기본값: 3개월
        horizontal=True,
        label_visibility="collapsed",
        key="chart_period",
    )

    # ── 장기 월봉 차트 ───────────────────────
    if _sel in _LONG_PERIODS:
        with st.spinner("월봉 데이터 불러오는 중…"):
            df_mo = get_bithumb_ohlcv_monthly()

        if df_mo is None:
            st.error("월봉 데이터를 불러올 수 없습니다.")
        else:
            n_mo = _LONG_PERIODS[_sel]
            cm = (df_mo.tail(n_mo) if n_mo else df_mo).copy()

            # 장기 이동평균 (월 기준)
            cm["MA6"]  = cm["close"].rolling(6).mean()    # 6개월
            cm["MA12"] = cm["close"].rolling(12).mean()   # 1년
            cm["MA24"] = cm["close"].rolling(24).mean()   # 2년
            cm["MA60"] = cm["close"].rolling(60).mean()   # 5년

            fig_lt = go.Figure()
            # 캔들스틱 (월봉)
            fig_lt.add_trace(go.Candlestick(
                x=cm["timestamp"],
                open=cm["open"], high=cm["high"],
                low=cm["low"],   close=cm["close"],
                name="BTC 월봉",
                increasing_line_color="#26c77a",
                decreasing_line_color="#ef5350",
            ))
            # 이동평균선
            for ma_col, ma_color, ma_w, ma_dash in [
                ("MA6",  "#42a5f5", 1.5, "solid"),
                ("MA12", "#ffb300", 1.8, "solid"),
                ("MA24", "#ab47bc", 2.0, "solid"),
                ("MA60", "#e53935", 2.0, "dash"),
            ]:
                if cm[ma_col].notna().any():
                    fig_lt.add_trace(go.Scatter(
                        x=cm["timestamp"], y=cm[ma_col],
                        name=ma_col,
                        line=dict(color=ma_color, width=ma_w, dash=ma_dash),
                    ))

            fig_lt.update_layout(
                title=dict(
                    text=f"BTC/KRW 월봉 차트 ({_sel})",
                    font=dict(size=14, color="#333"),
                ),
                height=560,
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#fafafa",
                xaxis_rangeslider_visible=False,
                xaxis=dict(
                    type="date",
                    tickformat="%Y년 %m월",
                    dtick="M12",           # 1년 간격 눈금
                    tickangle=-30,
                ),
                yaxis=dict(tickformat=","),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                margin=dict(t=60, b=40, l=10, r=10),
                font=dict(color="#222222"),
            )
            st.plotly_chart(fig_lt, use_container_width=True)
            _yr = len(cm) // 12
            _mo = len(cm) % 12
            _span = f"{_yr}년 {_mo}개월" if _mo else f"{_yr}년"
            st.caption(
                f"📌 월봉 차트 — MA6(6개월) · MA12(1년) · MA24(2년) · MA60(5년, 점선) "
                f"| 총 {len(cm)}개월({_span}) 데이터 "
                f"| {cm['timestamp'].iloc[0].strftime('%Y.%m')} ~ {cm['timestamp'].iloc[-1].strftime('%Y.%m')}"
            )

    # ── 단기 일봉 차트 ───────────────────────
    else:
        _n   = _SHORT_PERIODS[_sel]
        c_all = df.tail(_n).copy()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.60, 0.20, 0.20],
            subplot_titles=["BTC/KRW 일봉 차트", "RSI(14)", "MACD"],
        )

        # 캔들스틱
        fig.add_trace(go.Candlestick(
            x=c_all["timestamp"],
            open=c_all["open"], high=c_all["high"],
            low=c_all["low"],   close=c_all["close"],
            name="BTC",
            increasing_line_color="#26c77a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # 이동평균선
        for col_name, color, width in [
            ("MA20",  "#42a5f5", 1.5),
            ("MA50",  "#ffb300", 1.5),
            ("MA200", "#ab47bc", 1.8),
        ]:
            fig.add_trace(go.Scatter(
                x=c_all["timestamp"], y=c_all[col_name],
                name=col_name, line=dict(color=color, width=width),
            ), row=1, col=1)

        # 볼린저 밴드 (fill)
        ts_fwd        = list(c_all["timestamp"])
        ts_rev        = list(c_all["timestamp"].iloc[::-1])
        bb_upper_vals = list(c_all["BB_u"])
        bb_lower_vals = list(c_all["BB_l"].iloc[::-1])
        fig.add_trace(go.Scatter(
            x=ts_fwd + ts_rev,
            y=bb_upper_vals + bb_lower_vals,
            fill="toself",
            fillcolor="rgba(100,100,220,0.12)",
            line=dict(color="rgba(100,100,220,0.4)", width=1),
            name="볼린저 밴드",
            hoverinfo="skip",
        ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=c_all["timestamp"], y=c_all["RSI14"],
            name="RSI(14)", line=dict(color="#ff7043", width=1.5),
        ), row=2, col=1)
        for y_val, line_color in [(70, "rgba(239,83,80,0.55)"), (30, "rgba(38,199,122,0.55)")]:
            fig.add_shape(
                type="line",
                xref="paper", x0=0, x1=1,
                yref="y2",    y0=y_val, y1=y_val,
                line=dict(dash="dash", color=line_color, width=1),
            )

        # MACD
        hist_colors = ["#26c77a" if v >= 0 else "#ef5350" for v in c_all["MACD_hist"]]
        fig.add_trace(go.Bar(
            x=c_all["timestamp"], y=c_all["MACD_hist"],
            name="히스토그램", marker_color=hist_colors, opacity=0.7,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=c_all["timestamp"], y=c_all["MACD"],
            name="MACD", line=dict(color="#42a5f5", width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=c_all["timestamp"], y=c_all["MACD_sig"],
            name="Signal", line=dict(color="#e65c00", width=1.5),
        ), row=3, col=1)

        fig.update_layout(
            height=700,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fafafa",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=45, b=10, l=10, r=10),
            font=dict(color="#222222"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # 가중치 설정 패널 + 예측
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("⚙️ 지표 가중치 설정 & 예측")

    left, right = st.columns([3, 2], gap="large")

    with left:
        # ── 기술적 지표 ──────────────────────
        with st.expander("📊 기술적 지표 (자동 계산)", expanded=True):
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.markdown(f"**{_tt('RSI(14)', 'RSI')}**", unsafe_allow_html=True)
                w_rsi = st.slider("RSI 가중치", 0, 100, 20, key="w_rsi",
                                  help="RSI (상대강도지수): 30 이하 과매도(매수 신호), 70 이상 과매수(매도 신호)")
                st.markdown(
                    f"<small>값: **{rsi_v:.1f}** &nbsp;|&nbsp; "
                    f"신호: <span style='color:{signal_color(s_rsi)}'>{s_rsi:+.2f} "
                    f"{signal_label(s_rsi)}</span></small>",
                    unsafe_allow_html=True,
                )
            with r1c2:
                st.markdown(f"**{_tt('MACD', 'MACD')}**", unsafe_allow_html=True)
                w_macd = st.slider("MACD 가중치", 0, 100, 20, key="w_macd",
                                   help="MACD: 단기(12일) - 장기(26일) EMA 차이. 히스토그램 방향과 크기로 모멘텀 판단")
                st.markdown(
                    f"<small>히스토: **{macd_h:,.0f}** &nbsp;|&nbsp; "
                    f"신호: <span style='color:{signal_color(s_macd)}'>{s_macd:+.2f} "
                    f"{signal_label(s_macd)}</span></small>",
                    unsafe_allow_html=True,
                )

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.markdown(f"**{_tt('이동평균선 (MA20/50/200)', 'MA')}**", unsafe_allow_html=True)
                w_ma = st.slider("MA 가중치", 0, 100, 15, key="w_ma",
                                 help="MA20(단기)·MA50(중기)·MA200(장기) 배열 상태. 정배열 → 강세 신호")
                st.markdown(
                    f"<small>{_tt('MA200', 'MA200')}: **{ma200:,.0f}** &nbsp;|&nbsp; "
                    f"신호: <span style='color:{signal_color(s_ma)}'>{s_ma:+.2f} "
                    f"{signal_label(s_ma)}</span></small>",
                    unsafe_allow_html=True,
                )
            with r2c2:
                st.markdown(f"**{_tt('볼린저 밴드', '볼린저밴드')}**", unsafe_allow_html=True)
                w_bb = st.slider("BB 가중치", 0, 100, 10, key="w_bb",
                                 help="볼린저 밴드: MA20 ± 2σ. 하단 근처 → 과매도, 상단 근처 → 과매수")
                st.markdown(
                    f"<small>밴드 위치: **{s_bb:+.2f}** &nbsp;|&nbsp; "
                    f"신호: <span style='color:{signal_color(s_bb)}'>"
                    f"{signal_label(s_bb)}</span></small>",
                    unsafe_allow_html=True,
                )

        # ── 시장 심리 ────────────────────────
        with st.expander("😰 시장 심리 지표", expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**{_tt('공포/탐욕 지수', '공포탐욕')}** *(자동)*", unsafe_allow_html=True)
                w_fg = st.slider("F&G 가중치", 0, 100, 15, key="w_fg")
                st.markdown(
                    f"<small>지수: **{fg_val}** ({fg_cls}) &nbsp;|&nbsp; "
                    f"신호: <span style='color:{signal_color(s_fg)}'>{s_fg:+.2f} "
                    f"{signal_label(s_fg)}</span></small>",
                    unsafe_allow_html=True,
                )
            with sc2:
                st.markdown("**소셜 미디어 감성** *(수동)*")
                w_social = st.slider("소셜 가중치", 0, 100, 10, key="w_social")
                s_social = st.select_slider(
                    "소셜 감성 신호",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "🔴 매우 부정", -0.5: "🟠 부정",
                         0.0: "🟡 중립",
                         0.5: "🟢 긍정",  1.0: "💚 매우 긍정",
                    }[x],
                    key="s_social",
                )

        # ── 온체인 지표 ──────────────────────
        with st.expander("🔗 온체인 지표 (수동 입력)", expanded=False):
            on1, on2 = st.columns(2)
            with on1:
                st.markdown(f"**{_tt('해시레이트 추세', '해시레이트')}**", unsafe_allow_html=True)
                w_hash = st.slider("해시레이트 가중치", 0, 100, 10, key="w_hash",
                                   help="해시레이트: 비트코인 채굴 연산력. 상승 → 채굴자 신뢰 증가 → 장기 강세 신호")
                s_hash = st.select_slider(
                    "해시레이트 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "📉 급감", -0.5: "↘ 감소",
                         0.0: "→ 중립",
                         0.5: "↗ 증가",  1.0: "📈 급증",
                    }[x],
                    key="s_hash",
                )
            with on2:
                st.markdown(f"**{_tt('거래소 넷플로우', '넷플로우')}**", unsafe_allow_html=True)
                w_flow = st.slider("넷플로우 가중치", 0, 100, 10, key="w_flow",
                                   help="거래소 넷플로우: 순유출(음수) → BTC 보유 강세, 순유입(양수) → 매도 대기 증가")
                s_flow = st.select_slider(
                    "넷플로우 방향 (유출=보유=강세)",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "🔴 대량 유입(매도)", -0.5: "↘ 유입(매도)",
                         0.0: "→ 중립",
                         0.5: "↗ 유출(보유)",  1.0: "🟢 대량 유출(보유)",
                    }[x],
                    key="s_flow",
                )

        # ── 거시경제 ─────────────────────────
        with st.expander("🌍 거시경제 지표 (FRED 자동 + 수동 조정)", expanded=False):
            # FRED 자동 수집 상태 배너
            if fred:
                fred_items = []
                if "cpi_value" in fred:
                    fred_items.append(
                        f"📊 CPI <b>{fred['cpi_value']:.1f}</b> "
                        f"({'<span style=\"color:#e03131\">▲' if fred['cpi_change']>0 else '<span style=\"color:#26a65b\">▼'}"
                        f"{abs(fred['cpi_change']):.2f}%</span>)"
                    )
                if "rate_value" in fred:
                    fred_items.append(
                        f"🏦 기준금리 <b>{fred['rate_value']:.2f}%</b>"
                    )
                if "unemp_value" in fred:
                    fred_items.append(
                        f"👷 실업률 <b>{fred['unemp_value']:.1f}%</b> "
                        f"({'<span style=\"color:#e03131\">▲' if fred['unemp_change']>0 else '<span style=\"color:#26a65b\">▼'}"
                        f"{abs(fred['unemp_change']):.1f}%p</span>)"
                    )
                if fred_items:
                    st.markdown(
                        "<div style='background:#e8f4fd;border:1px solid #90caf9;"
                        "border-radius:7px;padding:0.45rem 0.9rem;font-size:0.82rem;"
                        "color:#1a237e;margin-bottom:0.7rem;'>"
                        "🤖 <b>FRED 자동 수집</b> &nbsp;|&nbsp; "
                        + " &nbsp;·&nbsp; ".join(fred_items)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("💡 FRED_API_KEY를 .env에 설정하면 CPI·금리·실업률이 자동 수집됩니다.")

            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"**{_tt('달러 인덱스 (DXY)', 'DXY')}**", unsafe_allow_html=True)
                w_dxy = st.slider("DXY 가중치", 0, 100, 8, key="w_dxy",
                                  help="DXY (달러 인덱스): 달러 약세(하락) → 비트코인·금 강세 경향")
                s_dxy = st.select_slider(
                    "DXY 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "🔴 달러 급강세", -0.5: "↘ 달러 강세",
                         0.0: "→ 중립",
                         0.5: "↗ 달러 약세",  1.0: "🟢 달러 급약세",
                    }[x],
                    key="s_dxy",
                )
            with mc2:
                st.markdown(f"**{_tt('연준 금리 방향', '금리')}**", unsafe_allow_html=True)
                w_rate = st.slider("금리 가중치", 0, 100, 8, key="w_rate",
                                   help="연준 기준금리: 인하 → 유동성 증가 → BTC 강세, 인상 → 약세 압력")
                # FRED 자동값이 있으면 표시, 슬라이더는 수동 조정용
                if s_rate_auto is not None:
                    _rc = signal_color(s_rate_auto)
                    st.markdown(
                        f"<small>🤖 FRED 자동: <b><span style='color:{_rc}'>"
                        f"{s_rate_auto:+.2f} {signal_label(s_rate_auto)}</span></b> "
                        f"(수동 슬라이더로 조정 가능)</small>",
                        unsafe_allow_html=True,
                    )
                _rate_default = round(s_rate_auto * 2) / 2 if s_rate_auto is not None else 0.0
                _rate_default = max(-1.0, min(1.0, _rate_default))
                s_rate = st.select_slider(
                    "금리 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=_rate_default,
                    format_func=lambda x: {
                        -1.0: "🔴 급인상", -0.5: "↘ 인상 기조",
                         0.0: "→ 동결",
                         0.5: "↗ 인하 기조", 1.0: "🟢 급인하",
                    }[x],
                    key="s_rate",
                )

            mc3, mc4 = st.columns(2)
            with mc3:
                st.markdown("**S&P500 추세**")
                w_sp500 = st.slider("S&P500 가중치", 0, 100, 7, key="w_sp500",
                                    help="주식시장 상승 → 위험자산 선호 → BTC 긍정")
                s_sp500 = st.select_slider(
                    "S&P500 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "🔴 급락", -0.5: "↘ 하락",
                         0.0: "→ 횡보",
                         0.5: "↗ 상승", 1.0: "🟢 급등",
                    }[x],
                    key="s_sp500",
                )
            with mc4:
                st.markdown("**금(GOLD) 추세**")
                w_gold = st.slider("금 가중치", 0, 100, 5, key="w_gold",
                                   help="안전자산 선호도 반영")
                s_gold = st.select_slider(
                    "금 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "🔴 급락", -0.5: "↘ 하락",
                         0.0: "→ 횡보",
                         0.5: "↗ 상승", 1.0: "🟢 급등",
                    }[x],
                    key="s_gold",
                )

            mc5, mc6 = st.columns(2)
            with mc5:
                st.markdown(f"**{_tt('CPI (소비자물가지수)', 'CPI')}**", unsafe_allow_html=True)
                w_cpi = st.slider("CPI 가중치", 0, 100, 8, key="w_cpi",
                                  help="CPI 상승 → 금리인상 압력 → BTC 약세 / 하락 → 완화 기대 → BTC 강세")
                if s_cpi_auto is not None:
                    _cc = signal_color(s_cpi_auto)
                    st.markdown(
                        f"<small>🤖 FRED 자동: CPI {fred.get('cpi_value','?'):.1f} "
                        f"({fred.get('cpi_change',0):+.2f}%) → "
                        f"<b><span style='color:{_cc}'>{s_cpi_auto:+.2f} {signal_label(s_cpi_auto)}</span></b></small>",
                        unsafe_allow_html=True,
                    )
                _cpi_default = round((s_cpi_auto or 0.0) * 2) / 2
                _cpi_default = max(-1.0, min(1.0, _cpi_default))
                s_cpi = st.select_slider(
                    "CPI 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=_cpi_default,
                    format_func=lambda x: {
                        -1.0: "🔴 급등(인플레심화)", -0.5: "↘ 상승",
                         0.0: "→ 안정",
                         0.5: "↗ 하락",  1.0: "🟢 급락(디플레)",
                    }[x],
                    key="s_cpi",
                )
            with mc6:
                st.markdown(f"**{_tt('고용지표 (실업률)', '고용')}**", unsafe_allow_html=True)
                w_emp = st.slider("고용 가중치", 0, 100, 7, key="w_emp",
                                  help="실업률 상승 → 금리인하 기대 → BTC 긍정 / 하락(과열) → 금리인상 압력")
                if s_emp_auto is not None:
                    _ec = signal_color(s_emp_auto)
                    st.markdown(
                        f"<small>🤖 FRED 자동: 실업률 {fred.get('unemp_value','?'):.1f}% "
                        f"({fred.get('unemp_change',0):+.1f}%p) → "
                        f"<b><span style='color:{_ec}'>{s_emp_auto:+.2f} {signal_label(s_emp_auto)}</span></b></small>",
                        unsafe_allow_html=True,
                    )
                _emp_default = round((s_emp_auto or 0.0) * 2) / 2
                _emp_default = max(-1.0, min(1.0, _emp_default))
                s_emp = st.select_slider(
                    "고용 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=_emp_default,
                    format_func=lambda x: {
                        -1.0: "🔴 완전고용(인상압력)", -0.5: "↘ 강한고용",
                         0.0: "→ 중립",
                         0.5: "↗ 실업증가",  1.0: "🟢 침체(인하기대)",
                    }[x],
                    key="s_emp",
                )

        # ── 지정학·규제 리스크 ────────────────
        with st.expander("🌐 지정학·규제 리스크 (뉴스 자동분석 + 수동 조정)", expanded=False):
            geo1, geo2 = st.columns(2)

            with geo1:
                st.markdown(f"**{_tt('지정학적 리스크', '지정학')}**", unsafe_allow_html=True)
                w_geo = st.slider("지정학 가중치", 0, 100, 8, key="w_geo",
                                  help="전쟁·제재·분쟁 등 지정학 리스크. 단기 하락·장기 BTC 수요 증가 효과")

                # 뉴스 자동 분석 결과 표시
                geo_sig_color = signal_color(s_geo_auto)
                geo_summary = geo.get("summary", "분석 데이터 없음")
                geo_key_event = geo.get("key_event", "")
                geo_headlines = geo.get("headlines", [])

                if geo_headlines:
                    st.markdown(
                        f"<div style='background:#fff8e1;border:1px solid #ffe082;"
                        f"border-radius:7px;padding:0.5rem 0.8rem;font-size:0.8rem;"
                        f"margin-bottom:0.5rem;'>"
                        f"🤖 <b>뉴스 자동 분석</b> → "
                        f"<span style='color:{geo_sig_color};font-weight:700;'>"
                        f"{s_geo_auto:+.2f} {signal_label(s_geo_auto)}</span><br>"
                        f"<span style='color:#555;'>{geo_summary}</span>"
                        + (f"<br><span style='color:#777;font-size:0.75rem;'>📌 {geo_key_event}</span>" if geo_key_event else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("📰 수집된 지정학 뉴스 헤드라인", expanded=False):
                        for h in geo_headlines[:8]:
                            st.markdown(f"- {h}")
                else:
                    st.caption("💡 NEWS_API_KEY를 설정하면 지정학 뉴스가 자동 수집됩니다.")

                _geo_default = round(s_geo_auto * 2) / 2
                _geo_default = max(-1.0, min(1.0, _geo_default))
                s_geo = st.select_slider(
                    "지정학 리스크 방향 (수동 조정)",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=_geo_default,
                    format_func=lambda x: {
                        -1.0: "🔴 전쟁확전·제재강화", -0.5: "↘ 긴장고조",
                         0.0: "→ 현상유지",
                         0.5: "↗ 협상진행",  1.0: "🟢 평화협정·완화",
                    }[x],
                    key="s_geo",
                )

            with geo2:
                st.markdown(f"**{_tt('클래리티법안 (CLARITY Act)', '클래리티법안')}**",
                            unsafe_allow_html=True)
                w_clarity = st.slider("클래리티법안 가중치", 0, 100, 7, key="w_clarity",
                                     help="미국 암호화폐 법적 지위 명확화 법안. 통과 시 기관 진입 장벽 완화 → 대형 호재")

                st.markdown(
                    "<div style='background:#f3e5f5;border:1px solid #ce93d8;"
                    "border-radius:7px;padding:0.5rem 0.8rem;font-size:0.8rem;"
                    "margin-bottom:0.5rem;'>"
                    "📋 <b>법안 진행 단계별 BTC 영향</b><br>"
                    "<span style='color:#555;'>"
                    "발의(+0.3) → 위원회통과(+0.6) → 하원통과(+0.8) → 서명(+1.0)<br>"
                    "청문회부정(-0.5) → 폐기(-0.8)"
                    "</span></div>",
                    unsafe_allow_html=True,
                )
                s_clarity = st.select_slider(
                    "클래리티법안 진행 상황",
                    options=[-1.0, -0.5, -0.2, 0.0, 0.3, 0.6, 0.8, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "🔴 법안폐기·강력규제",
                        -0.5: "🟠 청문회 부정적",
                        -0.2: "↘ 규제강화 우려",
                         0.0: "→ 미확인/현상유지",
                         0.3: "↗ 법안발의",
                         0.6: "🔵 위원회통과",
                         0.8: "💙 하원/상원통과",
                         1.0: "🟢 대통령서명·확정",
                    }[x],
                    key="s_clarity",
                )

                # 기타 규제 이슈 수동 입력
                st.markdown("**기타 규제 이슈**")
                w_reg = st.slider("규제 가중치", 0, 100, 5, key="w_reg",
                                  help="SEC·CFTC 결정, 국가별 규제 발표 등 기타 규제 이슈")
                s_reg = st.select_slider(
                    "규제 방향",
                    options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    value=0.0,
                    format_func=lambda x: {
                        -1.0: "🔴 강력규제·소송", -0.5: "↘ 규제강화",
                         0.0: "→ 현상유지",
                         0.5: "↗ 규제완화",  1.0: "🟢 전면허용",
                    }[x],
                    key="s_reg",
                )

        # ── 기관·고래 동향 (자동) ─────────────
        src_label = futures.get("source", "Binance") if futures else "?"
        with st.expander(f"🐋 기관·고래 동향 (선물 자동 수집 · {src_label})", expanded=True):
            if futures is None:
                st.warning("📡 선물 API(Binance·Bybit) 모두 응답 없음 — 가중치는 유지되나 신호값이 0으로 처리됩니다.", icon="⚠️")

            wh1, wh2 = st.columns(2)

            # ── 기관 동향 ──
            with wh1:
                st.markdown("**🏦 기관 동향** *(자동)*")
                w_inst = st.slider("기관 가중치", 0, 100, 12, key="w_inst")

                if futures:
                    # 롱/숏 비율 행
                    ls_c = signal_color(s_ls_ratio)
                    st.markdown(
                        f"<small>📊 {_tt('Top Trader 롱/숏', '롱숏')}: <b>{futures['ls_ratio']:.3f}</b>"
                        f"&nbsp;→&nbsp;<span style='color:{ls_c}'>{s_ls_ratio:+.2f} {signal_label(s_ls_ratio)}</span></small>",
                        unsafe_allow_html=True,
                    )
                    # OI 변화율 행
                    oi_c = signal_color(s_oi)
                    oi_arrow = "📈" if futures["oi_change_pct"] >= 0 else "📉"
                    st.markdown(
                        f"<small>{oi_arrow} {_tt('OI 24h 변화', 'OI')}: <b>{futures['oi_change_pct']:+.2f}%</b>"
                        f"&nbsp;→&nbsp;<span style='color:{oi_c}'>{s_oi:+.2f} {signal_label(s_oi)}</span></small>",
                        unsafe_allow_html=True,
                    )
                    inst_c = signal_color(s_inst_auto)
                    st.markdown(
                        f"<small>🔷 <b>기관 종합 신호: "
                        f"<span style='color:{inst_c}'>{s_inst_auto:+.2f} {signal_label(s_inst_auto)}</span></b></small>",
                        unsafe_allow_html=True,
                    )

            # ── 고래 동향 ──
            with wh2:
                st.markdown("**🐋 고래 동향** *(자동)*")
                w_whale = st.slider("고래 가중치", 0, 100, 12, key="w_whale")

                if futures:
                    # 테이커 비율 행
                    tk_c = signal_color(s_taker)
                    st.markdown(
                        f"<small>⚡ {_tt('테이커 매수/매도', '테이커')}: <b>{futures['taker_ratio']:.3f}</b>"
                        f"&nbsp;→&nbsp;<span style='color:{tk_c}'>{s_taker:+.2f} {signal_label(s_taker)}</span></small>",
                        unsafe_allow_html=True,
                    )
                    # 펀딩 레이트 행
                    fr_c = signal_color(s_funding)
                    fr_pct = futures["funding_rate"] * 100
                    st.markdown(
                        f"<small>💰 {_tt('펀딩 레이트', '펀딩레이트')}: <b>{fr_pct:+.4f}%</b>"
                        f"&nbsp;→&nbsp;<span style='color:{fr_c}'>{s_funding:+.2f} {signal_label(s_funding)}</span></small>",
                        unsafe_allow_html=True,
                    )
                    whale_c = signal_color(s_whale_auto)
                    st.markdown(
                        f"<small>🔷 <b>고래 종합 신호: "
                        f"<span style='color:{whale_c}'>{s_whale_auto:+.2f} {signal_label(s_whale_auto)}</span></b></small>",
                        unsafe_allow_html=True,
                    )

            # 데이터 출처 안내
            st.caption("데이터 출처: Binance Futures 공개 API (5분 캐시) — API 키 불필요")

    # ── 우측: 예측 결과 ────────────────────────
    with right:
        st.markdown("### 🎯 예측 설정")

        pred_period = st.radio(
            "📅 예측 기간",
            ["단기 (1~3일)", "중기 (1~2주)", "장기 (1~3개월)"],
            index=0,
            horizontal=True,
        )
        period_mult = {"단기 (1~3일)": 0.35, "중기 (1~2주)": 1.0, "장기 (1~3개월)": 3.2}
        p_mult = period_mult[pred_period]

        base_vol = st.slider(
            "📊 기준 변동폭 (%)",
            min_value=2, max_value=40, value=8,
            help="종합 신호 +1.0 일 때의 최대 예상 상승폭 (기간 배수 적용 전)",
        )
        vol = base_vol * p_mult

        st.markdown("---")

        # 신호·가중치 조합
        sw_pairs: list[tuple[float, float]] = [
            (s_rsi,    w_rsi),
            (s_macd,   w_macd),
            (s_ma,     w_ma),
            (s_bb,     w_bb),
            (s_fg,     w_fg),
            (s_social, w_social),
            (s_hash,   w_hash),
            (s_flow,   w_flow),
            (s_dxy,    w_dxy),
            (s_rate,   w_rate),
            (s_sp500,  w_sp500),
            (s_gold,   w_gold),
            (s_cpi,    w_cpi),       # CPI (FRED 자동 + 수동)
            (s_emp,    w_emp),       # 고용지표 (FRED 자동 + 수동)
            (s_geo,    w_geo),       # 지정학 리스크 (뉴스 LLM + 수동)
            (s_clarity, w_clarity),  # 클래리티법안 (수동)
            (s_reg,    w_reg),       # 기타 규제 (수동)
            (s_inst_auto,  w_inst),
            (s_whale_auto, w_whale),
        ]

        pred_price, chg_pct, composite = predict(price, sw_pairs, vol)
        total_w = sum(w for _, w in sw_pairs)

        if total_w == 0:
            st.warning("⚠️ 하나 이상의 가중치를 0 이상으로 설정하세요.")
        else:
            # 예측 박스 색상
            c = signal_color(composite)
            if composite > 0:
                bg = "#e8f5e9"
            elif composite < 0:
                bg = "#ffebee"
            else:
                bg = "#fffde7"

            st.markdown(
                f"""
                <div class="pred-box" style="background:{bg}; border:2px solid {c};">
                    <p style="color:#666; margin:0 0 0.4rem 0; font-size:0.85rem;">
                        {pred_period} 예측 가격
                    </p>
                    <h1 style="color:{c}; font-size:2.4rem; margin:0;">
                        {krw(pred_price)}
                    </h1>
                    <p style="color:{c}; font-size:1.1rem; margin:0.4rem 0 0.6rem 0;">
                        {chg_pct:+.2f}% &nbsp; {signal_label(composite)}
                    </p>
                    <p style="color:#888; font-size:0.8rem; margin:0;">
                        현재가 {krw(price)} → 변화 {krw(pred_price - price)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 종합 신호 게이지
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(composite * 100, 1),
                number={"font": {"size": 32, "color": c}},
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "종합 신호 지수 (-100 ~ +100)",
                       "font": {"size": 13, "color": "#555"}},
                gauge={
                    "axis": {"range": [-100, 100], "tickfont": {"size": 10}},
                    "bar": {"color": c, "thickness": 0.25},
                    "bgcolor": "#f5f5f5",
                    "bordercolor": "#cccccc",
                    "steps": [
                        {"range": [-100, -40], "color": "#ffcdd2"},
                        {"range": [ -40,  40], "color": "#f5f5e8"},
                        {"range": [  40, 100], "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "#333333", "width": 3},
                        "thickness": 0.8,
                        "value": composite * 100,
                    },
                },
            ))
            gauge.update_layout(
                height=250,
                paper_bgcolor="#ffffff",
                font={"color": "#222222"},
                margin=dict(t=50, b=20, l=30, r=30),
            )
            st.plotly_chart(gauge, use_container_width=True)

            # 지표별 기여도 바 차트
            names = [
                "RSI(14)", "MACD", "이동평균", "볼린저밴드",
                "공포/탐욕", "소셜감성", "해시레이트", "넷플로우",
                "DXY", "금리", "S&P500", "금(GOLD)",
                "CPI(소비자물가)", "고용지표",
                "지정학리스크", "클래리티법안", "기타규제",
                "기관동향(자동)", "고래동향(자동)",
            ]
            contribs = [
                s * w / total_w * 100 for s, w in sw_pairs
            ]
            active = [(n, c_) for n, c_, (_, w) in zip(names, contribs, sw_pairs) if w > 0]
            if active:
                act_names, act_vals = zip(*active)
                bar_colors = [signal_color(v / 100) for v in act_vals]

                bar_fig = go.Figure(go.Bar(
                    x=list(act_vals),
                    y=list(act_names),
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:+.1f}%" for v in act_vals],
                    textposition="outside",
                ))
                bar_fig.add_vline(x=0, line_color="#aaaaaa", line_width=1)
                bar_fig.update_layout(
                    title="지표별 기여도 (%)",
                    height=max(250, len(active) * 30 + 60),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#fafafa",
                    font={"color": "#222222", "size": 11},
                    margin=dict(t=40, b=10, l=10, r=60),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(bar_fig, use_container_width=True)

    # ══════════════════════════════════════════
    # 하단: 시나리오 분석
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 시나리오 분석")

    scenarios = [
        ("🚀 극강세", 0.85, "#e8f5e9", "#1b7a3e"),
        ("📈 강세",   0.45, "#f1f8f4", "#388e3c"),
        ("⚖️ 현재 분석", composite if total_w > 0 else 0.0, "#fffde7", "#b07d00"),
        ("↘ 약세",  -0.45, "#fff0f0", "#c62828"),
        ("📉 극약세", -0.85, "#ffebee", "#b71c1c"),
    ]

    sc_cols = st.columns(len(scenarios))
    for (sc_name, sc_sig, sc_bg, sc_c), col in zip(scenarios, sc_cols):
        sc_price = price * (1 + sc_sig * (vol / 100))
        sc_chg   = sc_sig * vol
        with col:
            st.markdown(
                f"""
                <div class="scenario-box" style="background:{sc_bg}; border:1px solid {sc_c};">
                    <p style="color:{sc_c}; margin:0; font-size:0.95rem;"><b>{sc_name}</b></p>
                    <h4 style="color:#111111; margin:0.4rem 0;">{krw(sc_price)}</h4>
                    <p style="color:{sc_c}; margin:0; font-size:0.9rem;">{sc_chg:+.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════
    # 기관·고래 동향 상세 분석
    # ══════════════════════════════════════════
    st.markdown("---")
    _src = futures.get("source", "Binance") if futures else "Bybit"
    st.subheader("🐋 기관·고래 동향 상세 분석")
    st.caption(f"데이터 출처: {_src} Futures 공개 API · 5분 캐시 · API 키 불필요")

    with st.spinner("선물 히스토리 불러오는 중…"):
        fhist = get_binance_futures_history()

    if futures:
        # ── 종합 신호 카드 ───────────────────
        def _big_signal_card(label: str, sig: float, detail: str) -> str:
            c   = signal_color(sig)
            bg  = "#e8f5e9" if sig > 0.15 else "#ffebee" if sig < -0.15 else "#fffde7"
            bar = int((sig + 1) / 2 * 100)
            return (
                f"<div style='background:{bg};border:2px solid {c};border-radius:12px;"
                f"padding:1rem 1.3rem;'>"
                f"<p style='font-size:0.8rem;color:#777;margin:0 0 0.3rem 0;'>{label}</p>"
                f"<div style='display:flex;align-items:baseline;gap:0.6rem;margin-bottom:0.5rem;'>"
                f"<span style='font-size:2rem;font-weight:700;color:{c};'>{sig:+.2f}</span>"
                f"<span style='font-size:1rem;color:{c};'>{signal_label(sig)}</span>"
                f"</div>"
                f"<div style='background:#e0e0e0;border-radius:4px;height:7px;margin-bottom:0.5rem;'>"
                f"<div style='background:{c};width:{bar}%;height:7px;border-radius:4px;'></div>"
                f"</div>"
                f"<p style='font-size:0.77rem;color:#666;margin:0;'>{detail}</p>"
                f"</div>"
            )

        fr_pct = futures["funding_rate"] * 100
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(
                _big_signal_card(
                    "🏦 기관 종합 신호 (L/S 역발상 + OI 방향)",
                    s_inst_auto,
                    f"Top Trader L/S: {futures['ls_ratio']:.3f} &nbsp;|&nbsp; OI 24h: {futures['oi_change_pct']:+.2f}%",
                ),
                unsafe_allow_html=True,
            )
        with bc2:
            st.markdown(
                _big_signal_card(
                    "🐋 고래 종합 신호 (테이커 방향 + 펀딩 역발상)",
                    s_whale_auto,
                    f"테이커 비율: {futures['taker_ratio']:.3f} &nbsp;|&nbsp; 펀딩 레이트: {fr_pct:+.4f}%",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # ── 4개 게이지 ───────────────────────
        def _mini_gauge(title: str, value: float,
                        lo: float, hi: float,
                        thr_lo: float, thr_hi: float,
                        unit: str = "", invert: bool = False) -> go.Figure:
            if invert:
                col = "#ef5350" if value >= thr_hi else "#26c77a" if value <= thr_lo else "#ffa726"
            else:
                col = "#26c77a" if value >= thr_hi else "#ef5350" if value <= thr_lo else "#ffa726"
            steps = [
                {"range": [lo, thr_lo],  "color": "#ffcdd2" if not invert else "#c8e6c9"},
                {"range": [thr_lo, thr_hi], "color": "#fffde7"},
                {"range": [thr_hi, hi],  "color": "#c8e6c9" if not invert else "#ffcdd2"},
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number={"suffix": unit, "font": {"size": 22, "color": col}},
                title={"text": title, "font": {"size": 11, "color": "#555"}},
                gauge={
                    "axis": {"range": [lo, hi], "tickfont": {"size": 9}},
                    "bar":  {"color": col, "thickness": 0.28},
                    "bgcolor": "#f5f5f5",
                    "bordercolor": "#ddd",
                    "steps": steps,
                    "threshold": {"line": {"color": col, "width": 3},
                                  "thickness": 0.85, "value": value},
                },
            ))
            fig.update_layout(
                height=210,
                paper_bgcolor="#ffffff",
                margin=dict(t=45, b=5, l=15, r=15),
                font={"color": "#333"},
            )
            return fig

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(
                _mini_gauge("Top Trader 롱/숏 비율",
                            futures["ls_ratio"], 0.4, 2.6, 0.8, 1.5, invert=True),
                use_container_width=True, key="wh_g1",
            )
            st.caption("🔴 1.5↑ 롱 과열(매도신호)　🟢 0.8↓ 숏 과열(매수신호)")
        with g2:
            st.plotly_chart(
                _mini_gauge("OI 24h 변화율", futures["oi_change_pct"],
                            -10, 10, -2, 2, unit="%"),
                use_container_width=True, key="wh_g2",
            )
            st.caption("🟢 양수 포지션 확대　🔴 음수 청산 흐름")
        with g3:
            st.plotly_chart(
                _mini_gauge("펀딩 레이트", fr_pct,
                            -0.12, 0.12, -0.03, 0.03, unit="%", invert=True),
                use_container_width=True, key="wh_g3",
            )
            st.caption("🔴 +0.03%↑ 롱 과열　🟢 음수 숏 과열")
        with g4:
            st.plotly_chart(
                _mini_gauge("테이커 매수/매도 비율",
                            futures["taker_ratio"], 0.4, 2.0, 0.85, 1.2),
                use_container_width=True, key="wh_g4",
            )
            st.caption("🟢 1.2↑ 공격적 매수　🔴 0.85↓ 공격적 매도")

        # ── 추이 차트 4종 ────────────────────
        st.markdown("##### 📈 히스토리 추이")
        hc1, hc2 = st.columns(2)

        # OI 30일 바 차트
        with hc1:
            if fhist.get("oi") is not None:
                d = fhist["oi"]
                colors = ["#26c77a" if d["oi"].iloc[i] >= d["oi"].iloc[max(i-1,0)]
                          else "#ef5350" for i in range(len(d))]
                fig_oi = go.Figure(go.Bar(
                    x=d["ts"], y=d["oi"],
                    marker_color=colors,
                    hovertemplate="%{x|%m/%d}<br>OI: %{y:,.0f}<extra></extra>",
                ))
                fig_oi.update_layout(
                    title="미결제약정(OI) 30일 추이",
                    height=260, paper_bgcolor="#fff", plot_bgcolor="#fafafa",
                    font=dict(size=11, color="#333"),
                    margin=dict(t=35, b=10, l=10, r=10),
                    showlegend=False, yaxis=dict(tickformat=".3s"),
                )
                st.plotly_chart(fig_oi, use_container_width=True, key="wh_oi")

        # L/S 롱·숏 비중 72h 영역 차트
        with hc2:
            if fhist.get("ls") is not None:
                d = fhist["ls"]
                fig_ls = go.Figure()
                fig_ls.add_trace(go.Scatter(
                    x=d["ts"], y=d["long_pct"] * 100,
                    fill="tozeroy", fillcolor="rgba(38,199,122,0.22)",
                    line=dict(color="#26c77a", width=1.8), name="롱(%)",
                    hovertemplate="%{x|%m/%d %H시}<br>롱: %{y:.1f}%<extra></extra>",
                ))
                fig_ls.add_trace(go.Scatter(
                    x=d["ts"], y=d["short_pct"] * 100,
                    fill="tozeroy", fillcolor="rgba(239,83,80,0.18)",
                    line=dict(color="#ef5350", width=1.8), name="숏(%)",
                    hovertemplate="%{x|%m/%d %H시}<br>숏: %{y:.1f}%<extra></extra>",
                ))
                fig_ls.add_hline(y=50, line_dash="dash", line_color="#aaa", line_width=1)
                fig_ls.update_layout(
                    title="Top Trader 롱/숏 비중 72h",
                    height=260, paper_bgcolor="#fff", plot_bgcolor="#fafafa",
                    font=dict(size=11, color="#333"),
                    margin=dict(t=35, b=10, l=10, r=10),
                    legend=dict(orientation="h", y=1.0, x=1, xanchor="right"),
                    yaxis=dict(range=[0, 100], ticksuffix="%"),
                )
                st.plotly_chart(fig_ls, use_container_width=True, key="wh_ls")

        hc3, hc4 = st.columns(2)

        # 테이커 비율 72h 바 차트
        with hc3:
            if fhist.get("taker") is not None:
                d = fhist["taker"]
                tk_c = ["#26c77a" if r >= 1.0 else "#ef5350" for r in d["ratio"]]
                fig_tk = go.Figure(go.Bar(
                    x=d["ts"], y=d["ratio"],
                    marker_color=tk_c,
                    hovertemplate="%{x|%m/%d %H시}<br>비율: %{y:.3f}<extra></extra>",
                ))
                fig_tk.add_hline(y=1.0, line_dash="dash", line_color="#555", line_width=1.5,
                                 annotation_text="기준선 1.0",
                                 annotation_font=dict(size=10, color="#555"))
                fig_tk.update_layout(
                    title="테이커 매수/매도 비율 72h",
                    height=260, paper_bgcolor="#fff", plot_bgcolor="#fafafa",
                    font=dict(size=11, color="#333"),
                    margin=dict(t=35, b=10, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_tk, use_container_width=True, key="wh_tk")

        # 펀딩 레이트 히스토리 바 차트
        with hc4:
            if fhist.get("funding") is not None:
                d = fhist["funding"]
                fr_c = ["#ef5350" if r > 0 else "#26c77a" for r in d["rate"]]
                fig_fr = go.Figure(go.Bar(
                    x=d["ts"], y=d["rate"],
                    marker_color=fr_c,
                    hovertemplate="%{x|%m/%d %H시}<br>%{y:+.4f}%<extra></extra>",
                ))
                fig_fr.add_hline(y=0, line_color="#aaa", line_width=1)
                fig_fr.add_hline(y=0.03, line_dash="dot", line_color="#ef5350", line_width=1,
                                 annotation_text="롱 과열 +0.03%",
                                 annotation_font=dict(size=10, color="#ef5350"))
                fig_fr.add_hline(y=-0.03, line_dash="dot", line_color="#26c77a", line_width=1,
                                 annotation_text="숏 과열 -0.03%",
                                 annotation_font=dict(size=10, color="#26c77a"))
                fig_fr.update_layout(
                    title="펀딩 레이트 추이 (약 10일)",
                    height=260, paper_bgcolor="#fff", plot_bgcolor="#fafafa",
                    font=dict(size=11, color="#333"),
                    margin=dict(t=35, b=10, l=10, r=10),
                    showlegend=False,
                    yaxis=dict(ticksuffix="%"),
                )
                st.plotly_chart(fig_fr, use_container_width=True, key="wh_fr")

    else:
        st.warning(
            "📡 **선물 히스토리 데이터를 불러올 수 없습니다.**\n\n"
            "Binance와 Bybit API 모두 응답하지 않았습니다. "
            "잠시 후 페이지를 새로고침 해주세요.",
            icon="⚠️",
        )

    # ══════════════════════════════════════════
    # 뉴스 & 정치 환경
    # ══════════════════════════════════════════
    st.markdown("---")
    # 헤더 + 도넛 차트 자리 예약 (데이터 로드 후 채움)
    _news_hdr_col, _donut_slot_col = st.columns([3, 1])
    with _news_hdr_col:
        st.subheader("📰 비트코인 주요 뉴스 & 정치 환경")
    _donut_slot = _donut_slot_col.empty()   # ← 나중에 도넛 차트를 여기에 삽입

    # 뉴스 로딩 + AI 요약 자동 실행
    _, btn_col = st.columns([8, 2])
    with btn_col:
        if st.button("🔄 뉴스 새로고침", key="news_refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("최신 뉴스 & AI 분석 불러오는 중… (첫 로드 시 10~20초 소요)"):
        news_items = get_btc_news()
        ai_sum     = summarize_news_with_ai(news_items) if news_items else {}

    # ── AI 요약 (뉴스 목록 위에 자동 표시) ───
    if ai_sum and "error" not in ai_sum:
        sig_raw    = ai_sum.get("signal", "중립")
        sig_emoji  = "🟢" if "강세" in sig_raw else "🔴" if "약세" in sig_raw else "🟡"
        sig_bg     = "#e8f5e9" if "강세" in sig_raw else "#ffebee" if "약세" in sig_raw else "#fffde7"
        sig_border = "#388e3c" if "강세" in sig_raw else "#c62828" if "약세" in sig_raw else "#b07d00"

        st.markdown(
            f"""
            <div style="background:{sig_bg}; border:2px solid {sig_border};
                        border-radius:12px; padding:1.1rem 1.4rem; margin-bottom:0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="color:{sig_border}; margin:0;">
                        {sig_emoji} AI 종합 전망: <b>{sig_raw}</b>
                    </h4>
                    <span style="font-size:0.75rem; color:#999;">30분 캐시 · GPT-4o-mini</span>
                </div>
                <p style="color:#555; margin:0.5rem 0 0 0; font-size:0.84rem;">
                    ⚠️ {ai_sum.get('key_risks', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def _ai_card(icon: str, title: str, text: str, bg: str, border: str) -> str:
            return (
                f"<div style='background:{bg};border:1px solid {border};"
                f"border-radius:10px;padding:1.1rem 1.3rem;'>"
                f"<p style='font-weight:700;margin:0 0 0.6rem 0;font-size:0.95rem;color:#222;'>"
                f"{icon} {title}</p>"
                f"<p style='margin:0;font-size:0.87rem;line-height:1.75;color:#333;white-space:pre-wrap;'>{text}</p>"
                f"</div>"
            )

        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown(
                _ai_card("⚖️", "규제·정치", ai_sum.get("regulation", ""),
                         "#f0f4ff", "#90caf9"),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.markdown(
                _ai_card("📊", "시장·가격", ai_sum.get("market", ""),
                         "#fff8e1", "#ffe082"),
                unsafe_allow_html=True,
            )
        with ai_c2:
            st.markdown(
                _ai_card("🏦", "기관·투자", ai_sum.get("institutional", ""),
                         "#f3e5f5", "#ce93d8"),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.markdown(
                _ai_card("🌐", "전반적 전망", ai_sum.get("overall", ""),
                         "#e8f5e9", "#a5d6a7"),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        st.markdown("---")

    elif ai_sum.get("error"):
        err_msg = ai_sum['error']
        if "OPENAI_API_KEY" in err_msg or "Missing credentials" in err_msg or "api_key" in err_msg.lower():
            with st.expander("🔑 AI 자동 요약을 활성화하려면 클릭", expanded=False):
                st.markdown(
                    "**OpenAI API 키**가 설정되지 않아 AI 요약 기능이 비활성화되어 있습니다.\n\n"
                    "**Streamlit Cloud 배포 시:**\n"
                    "앱 대시보드 → ⋮ → **Settings → Secrets** 에 아래 내용 추가:\n"
                    "```toml\n"
                    'OPENAI_API_KEY = "sk-..."\n'
                    "```\n\n"
                    "**로컬 실행 시:**\n"
                    "프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...` 추가"
                )
        else:
            st.warning(f"AI 요약 오류: {err_msg}")

    # ── 뉴스 탭 ──────────────────────────────
    if not news_items:
        st.warning("뉴스를 불러오지 못했습니다. 네트워크를 확인하거나 잠시 후 다시 시도하세요.")
    else:
        categories = list(_NEWS_QUERIES.keys())
        tab_all, tab_reg, tab_inst, tab_mkt = st.tabs(
            ["📋 전체", categories[0], categories[1], categories[2]]
        )

        def _render_news(items: list[dict]) -> None:
            if not items:
                st.caption("해당 카테고리의 뉴스가 없습니다.")
                return
            for it in items:
                sent_color = {
                    "🟢": "#e8f5e9", "🔴": "#ffebee",
                    "🟡": "#fffde7", "⚪": "#f5f5f5",
                }[it["sentiment"]]
                sent_border = {
                    "🟢": "#a5d6a7", "🔴": "#ef9a9a",
                    "🟡": "#ffe082", "⚪": "#e0e0e0",
                }[it["sentiment"]]

                # HTML 인젝션 방지: 사용자 데이터 이스케이프
                title_safe  = html_mod.escape(it['title'])
                source_safe = html_mod.escape(it['source'])
                url_safe    = html_mod.escape(it['url'])
                desc_safe   = html_mod.escape(it.get('desc', ''))

                cat_badge = (
                    f"<span style='background:#e3f2fd;color:#1565c0;"
                    f"border-radius:4px;padding:1px 7px;font-size:0.75rem;"
                    f"font-weight:600'>{it['category']}</span>"
                )
                title_ko = html_mod.escape(it.get("title_ko", ""))
                ko_section = (
                    f"<div style='color:#444;font-size:0.85rem;font-weight:500;"
                    f"margin-top:3px;line-height:1.45;'>{title_ko}</div>"
                    if title_ko and title_ko != title_safe else ""
                )
                desc_section = (
                    f"<div style='margin-top:3px;'>"
                    f"<small style='color:#555;'>{desc_safe}</small></div>"
                    if desc_safe else ""
                )
                # st.html() 사용: 마크다운 파서를 거치지 않아 빈 줄·들여쓰기로
                # HTML 블록이 코드블록으로 오해받는 CommonMark 파싱 버그 방지
                st.html(
                    f"<div style='background:{sent_color};border:1px solid {sent_border};"
                    f"border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                    f"<div style='flex:1;'>"
                    f"<a href='{url_safe}' target='_blank' "
                    f"style='color:#777;font-weight:400;font-size:0.82rem;"
                    f"text-decoration:none;line-height:1.4;'>"
                    f"{it['sentiment']} {title_safe}</a>"
                    f"{ko_section}"
                    f"<div style='margin-top:4px;'>"
                    f"<small style='color:#666;'>"
                    f"📰 {source_safe} &nbsp;·&nbsp; 📅 {it['date']}"
                    f"&nbsp;&nbsp;{cat_badge}"
                    f"</small></div>"
                    f"{desc_section}"
                    f"</div></div></div>"
                )

        with tab_all:
            # ── 감성 필터 ──────────────────────────
            _sent_map = {"🟢": "🟢 긍정", "🔴": "🔴 부정", "🟡": "🟡 혼조", "⚪": "⚪ 중립"}
            _cnt = {"🟢 긍정": 0, "🔴 부정": 0, "🟡 혼조": 0, "⚪ 중립": 0}
            for _it in news_items:
                _cnt[_sent_map[_it["sentiment"]]] += 1

            _filter_options = (
                [f"전체 ({len(news_items)}건)"]
                + [f"🟢 긍정 ({_cnt['🟢 긍정']}건)"]
                + [f"🔴 부정 ({_cnt['🔴 부정']}건)"]
                + [f"🟡 혼조 ({_cnt['🟡 혼조']}건)"]
                + [f"⚪ 중립 ({_cnt['⚪ 중립']}건)"]
            )
            _sel = st.radio(
                "감성 필터",
                _filter_options,
                horizontal=True,
                label_visibility="collapsed",
                key="news_sent_filter",
            )
            # 필터링
            _emoji_filter = (
                None  if _sel.startswith("전체") else
                "🟢"  if _sel.startswith("🟢") else
                "🔴"  if _sel.startswith("🔴") else
                "🟡"  if _sel.startswith("🟡") else "⚪"
            )
            _filtered = (
                news_items[:24] if _emoji_filter is None
                else [n for n in news_items if n["sentiment"] == _emoji_filter]
            )
            st.caption(f"총 {len(news_items)}건 표시 중 {len(_filtered)}건")
            _render_news(_filtered)
        with tab_reg:
            _render_news([n for n in news_items if n["category"] == categories[0]])
        with tab_inst:
            _render_news([n for n in news_items if n["category"] == categories[1]])
        with tab_mkt:
            _render_news([n for n in news_items if n["category"] == categories[2]])

        # 감성 분포 — 헤더 오른쪽 플레이스홀더에 삽입
        sent_counts = {"🟢 긍정": 0, "🔴 부정": 0, "🟡 혼조": 0, "⚪ 중립": 0}
        for it in news_items:
            _sk = {"🟢": "🟢 긍정", "🔴": "🔴 부정", "🟡": "🟡 혼조", "⚪": "⚪ 중립"}[it["sentiment"]]
            sent_counts[_sk] += 1

        donut = go.Figure(go.Pie(
            labels=list(sent_counts.keys()),
            values=list(sent_counts.values()),
            hole=0.58,
            marker_colors=["#66bb6a", "#ef5350", "#ffa726", "#bdbdbd"],
            textinfo="label+percent",
            textfont_size=10,
        ))
        donut.update_layout(
            title=dict(text="뉴스 감성 분포", font=dict(size=12), x=0.5),
            height=210,
            paper_bgcolor="#ffffff",
            font=dict(color="#222222"),
            margin=dict(t=30, b=0, l=0, r=0),
            showlegend=False,
        )
        # 헤더 행 오른쪽 칸에 삽입
        _donut_slot.plotly_chart(donut, use_container_width=True)

    # ══════════════════════════════════════════
    # 수동 메모 (사이드바)
    # ══════════════════════════════════════════
    with st.sidebar:
        st.header("📝 투자 메모")
        memo = st.text_area(
            "분석 노트",
            placeholder="현재 시장 상황, 판단 근거, 리스크 요소 등을 기록하세요...",
            height=200,
            key="memo",
        )

        st.markdown("---")
        st.markdown("### 📌 빠른 참조")
        st.markdown(f"- **현재가**: {krw(price)}")
        st.markdown(f"- **{_tt('RSI', 'RSI')}**: {rsi_v:.1f}", unsafe_allow_html=True)
        st.markdown(f"- **{_tt('공포/탐욕', '공포탐욕')}**: {fg_val} ({fg_cls})", unsafe_allow_html=True)
        st.markdown(f"- **{_tt('MA200', 'MA200')}**: {krw(ma200)}", unsafe_allow_html=True)
        st.markdown(f"- **{_tt('볼린저 상단', '볼린저밴드')}**: {krw(bb_u)}", unsafe_allow_html=True)
        st.markdown(f"- **{_tt('볼린저 하단', '볼린저밴드')}**: {krw(bb_l)}", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("⚠️ 이 분석은 투자 조언이 아닙니다. 투자 판단은 본인 책임입니다.")

    # ── 타임스탬프 ────────────────────────────
    st.markdown("---")
    st.caption(
        f"⏰ 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST  "
        f"| 데이터 출처: Bithumb API, Alternative.me"
    )


# ══════════════════════════════════════════════
if __name__ == "__main__":
    main()
