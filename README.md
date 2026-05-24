# ₿ 비트코인 투자 분석 대시보드

빗썸 실시간 시세 + 기술·심리·온체인·거시경제·지정학 지표를 통합한 BTC 투자 분석 앱

---

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. API 키 설정
```bash
# .env.example을 복사해서 .env 파일 생성
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```
`.env` 파일을 열고 발급받은 API 키를 입력하세요.

| API 키 | 필수 여부 | 발급 링크 |
|--------|---------|----------|
| `FRED_API_KEY` | 선택 | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) (무료) |
| `NEWS_API_KEY` | 선택 | [newsapi.org](https://newsapi.org/register) (무료) |
| `OPENAI_API_KEY` | 선택 | [platform.openai.com](https://platform.openai.com/api-keys) (유료) |
| `BITHUMB_ACCESS_KEY` | 선택 | [bithumb.com](https://www.bithumb.com) API 관리 |

> ⚠️ **API 키 없어도 실행 가능** — 키가 없는 기능은 수동 입력 모드로 자동 전환됩니다.

### 4. 앱 실행
```bash
streamlit run bitcoin_analyzer.py
```
브라우저에서 http://localhost:8501 접속

---

## 📊 주요 기능

- **실시간 시세** — 빗썸 BTC/KRW, 바이낸스 BTC/USDT, USD/KRW 환율
- **기술적 지표** — RSI, MACD, 이동평균, 볼린저밴드 (자동 계산)
- **시장 심리** — 공포/탐욕 지수, 소셜 감성
- **온체인 지표** — 해시레이트, 거래소 넷플로우
- **거시경제** — DXY, 금리, S&P500, 금, CPI, 고용지표 (FRED 자동 수집)
- **지정학·규제** — 이란/러시아 등 뉴스 LLM 자동 분석, 클래리티법안 슬라이더
- **기관·고래** — 바이낸스 선물 롱/숏 비율, OI, 펀딩레이트, 테이커 비율 (자동)
- **AI 뉴스 요약** — GPT-4o-mini 기반 한국어 분석 보고서
- **예측 가격** — 가중 합산 신호 기반 단기·중기·장기 예측

---

## ⚙️ 환경 요구사항

- Python 3.10 이상
- 인터넷 연결 (API 호출)
