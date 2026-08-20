"""
발전부문 배출권 가격 예측 MVP — 백엔드 API 서버 (FastAPI)
국립창원대학교 identity 팀

역할:
  1) 마스터셋 CSV를 읽고 OLS(statsmodels) + RandomForest(scikit-learn) 모델을 학습
  2) GET /api/predict 로 향후 1년 연쇄 릴레이 예측 결과를 JSON 으로 제공
  3) 프론트엔드(../frontend)를 정적 파일로 서빙 (같은 서버/같은 오리진)

실행:
  pip install -r requirements.txt
  uvicorn app:app --host 0.0.0.0 --port 8000
  → 브라우저에서 http://localhost:8000  (대시보드)
     API 단독 확인:  http://localhost:8000/api/predict
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "발전부문_배출권_분석_마스터셋.csv")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

FEATURES = ['할당량_여유_지수', 'X_원단위', '배출권가격_1달전', '정산기_시즌스위치', '리스크_2024더미']


# -------------------------------------------------------------
# 1. 마스터 데이터셋 로드 및 전처리 (원본 load_data 로직)
# -------------------------------------------------------------
def load_data():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    cols = list(df.columns)

    if '총_할당량' in df.columns and '총_배출량' in df.columns:
        df['할당량_여유_지수'] = df['총_할당량'] - df['총_배출량']
    elif '발전부문_수급과부족지수' in df.columns:
        df['할당량_여유_지수'] = df['발전부문_수급과부족지수'] * -1
    else:
        df['할당량_여유_지수'] = df[cols[1]] * -1

    df['y_가격'] = df['KRX_배출권가격'] if 'KRX_배출권가격' in df.columns else df[cols[-1]]
    df['X_원단위'] = df['원단위'] if '원단위' in df.columns else df[cols[1]]
    df['배출권가격_1달전'] = df['y_가격'].shift(1)
    df['정산기_시즌스위치'] = df['월'].isin([4, 5, 6]).astype(int) if '월' in df.columns else np.zeros(len(df))
    df['리스크_2024더미'] = (df['년도'] == 2024).astype(int) if '년도' in df.columns else np.zeros(len(df))

    if '년도' in df.columns and '월' in df.columns:
        df['날짜'] = pd.to_datetime(df['년도'].astype(int).astype(str) + '-' + df['월'].astype(int).astype(str) + '-01', errors='coerce')
    else:
        df['날짜'] = pd.date_range(start='2021-01-01', periods=len(df), freq='MS')

    return df.dropna(subset=['날짜', 'y_가격', '할당량_여유_지수', 'X_원단위', '배출권가격_1달전']).sort_values('날짜').reset_index(drop=True)


# -------------------------------------------------------------
# 2. 모델 학습 (서버 시작 시 1회)
# -------------------------------------------------------------
DF = load_data()
_X = DF[FEATURES]
_y = DF['y_가격']
OLS_MODEL = sm.OLS(_y, sm.add_constant(_X)).fit()
RF_MODEL = RandomForestRegressor(n_estimators=100, random_state=42)
RF_MODEL.fit(_X, _y)


# -------------------------------------------------------------
# 3. 향후 1년 연쇄 릴레이 예측 (원본 로직)
# -------------------------------------------------------------
def run_forecast(mode, model, sim_margin, sim_intensity, season_option):
    auto = (mode == 'auto')
    latest = DF.iloc[-1]
    latest_date = DF['날짜'].max()
    today = pd.Timestamp.now().normalize().replace(day=1)
    target_end = today + pd.DateOffset(months=12)

    rows = []
    lag = latest['y_가격']
    cur = latest_date + pd.DateOffset(months=1)
    while cur <= target_end:
        mon = cur.month
        if auto or season_option == 'auto':
            season = 1 if mon in [4, 5, 6] else 0
        elif season_option == 'force-on':
            season = 1
        else:
            season = 0

        hist_m = DF[DF['날짜'].dt.month == mon]
        base_intensity = hist_m['X_원단위'].mean() if len(hist_m) else latest['X_원단위']
        base_margin = hist_m['할당량_여유_지수'].mean() if len(hist_m) else latest['할당량_여유_지수']
        tightening = max(0, cur.year - latest_date.year) * 0.4
        risk = 1 if cur.year == 2024 else 0

        if auto:
            step_intensity = base_intensity
            step_margin = base_margin - tightening
        else:
            step_margin = base_margin + sim_margin - tightening
            step_intensity = base_intensity * (1 + sim_intensity / 100)

        base_input = pd.DataFrame([{'할당량_여유_지수': base_margin, 'X_원단위': base_intensity, '배출권가격_1달전': lag, '정산기_시즌스위치': season, '리스크_2024더미': risk}])
        step_input = pd.DataFrame([{'할당량_여유_지수': step_margin, 'X_원단위': step_intensity, '배출권가격_1달전': lag, '정산기_시즌스위치': season, '리스크_2024더미': risk}])

        if model == 'ols':
            ci = sm.add_constant(step_input, has_constant='add')
            ci['const'] = 1.0
            ci = ci[['const'] + FEATURES]
            pred = float(OLS_MODEL.predict(ci)[0])
        else:
            if auto:
                pred = float(RF_MODEL.predict(step_input)[0])
            else:
                rf_base = float(RF_MODEL.predict(base_input)[0])
                rf_raw = float(RF_MODEL.predict(step_input)[0])
                smooth = (sim_intensity * (rf_base * 0.015)) - (sim_margin * 45.0)
                pred = rf_raw + smooth * 0.3 if rf_raw != rf_base else rf_base + smooth

        pred = max(0.0, pred)
        rows.append({'날짜': cur, '예측가격': pred, '여유지수': float(step_margin), '정산기': int(season)})
        lag = pred
        cur = cur + pd.DateOffset(months=1)

    fut = pd.DataFrame(rows)
    iso = lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')
    rec = lambda r: {'date': iso(r['날짜']), '예측가격': r['예측가격'], '여유지수': r['여유지수'], '정산기': r['정산기']}

    next1y = [rec(r) for _, r in fut[fut['날짜'] >= today].iterrows()]
    next_month = [rec(r) for _, r in fut[fut['날짜'] > today].iterrows()]
    prices = [r['예측가격'] for r in next1y]

    return {
        'history': [{'date': iso(r['날짜']), 'y': float(r['y_가격'])} for _, r in DF.iterrows()],
        'latestDate': iso(latest_date),
        'today': iso(today),
        'targetEnd': iso(target_end),
        'future': [rec(r) for _, r in fut.iterrows()],
        'next1y': next1y,
        'nextMonth': next_month,
        'todayPred': next1y[0] if next1y else None,
        'nextPred': next_month[0] if next_month else (next1y[0] if next1y else None),
        'lastPred': next1y[-1] if next1y else None,
        'minP': min(prices) if prices else 0,
        'maxP': max(prices) if prices else 0,
        'dataName': os.path.basename(CSV_PATH),
    }


# -------------------------------------------------------------
# 4. FastAPI 앱
# -------------------------------------------------------------
app = FastAPI(title="배출권 가격 예측 MVP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "rows": int(len(DF)), "data": os.path.basename(CSV_PATH)}


@app.get("/api/predict")
def predict(
    mode: str = Query("auto", pattern="^(auto|manual)$"),
    model: str = Query("ols", pattern="^(ols|ai)$"),
    simMargin: float = 0.0,
    simIntensity: float = 0.0,
    seasonOption: str = Query("auto", pattern="^(auto|force-on|force-off)$"),
):
    return run_forecast(mode, model, simMargin, simIntensity, seasonOption)


# 프론트엔드 정적 서빙 (API 라우트 뒤에 마운트해야 함)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
