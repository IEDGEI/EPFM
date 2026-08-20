"""발전부문 배출권 가격 예측 API 및 8월 고도화 기능."""

from __future__ import annotations

import os
from threading import RLock
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from data_pipeline import (
    build_master_auto,
    build_master_from_raw,
    compare_forecast_actual,
    read_table_bytes,
    standardize_master,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "발전부문_배출권_분석_마스터셋.csv")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
FEATURES = ["할당량_여유_지수", "X_원단위", "배출권가격_1달전", "정산기_시즌스위치", "리스크_2024더미"]


def _model_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=["날짜", "y_가격", *FEATURES])
        .sort_values("날짜")
        .reset_index(drop=True)
    )


def _fit_models(df: pd.DataFrame, tune_rf: bool = True):
    train = _model_frame(df)
    if len(train) < 18:
        raise ValueError("모델 학습에 필요한 유효 데이터가 18개월 미만입니다.")
    x, y = train[FEATURES].astype(float), train["y_가격"].astype(float)
    # 가격을 로그 공간에서 학습해 장기 연쇄 예측이 음수로 붕괴하는 것을 방지한다.
    ols = sm.OLS(np.log1p(y), sm.add_constant(x, has_constant="add")).fit()
    base_rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if tune_rf and len(train) >= 30:
        search = GridSearchCV(
            base_rf,
            {"max_depth": [3, 5, None], "min_samples_leaf": [1, 2, 4]},
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=4),
            n_jobs=-1,
        ).fit(x, y)
        rf = search.best_estimator_
    else:
        rf = base_rf.set_params(max_depth=5, min_samples_leaf=2).fit(x, y)
    return train, ols, rf


class ModelState:
    def __init__(self):
        self.lock = RLock()
        self.version = 0
        self.replace(pd.read_csv(CSV_PATH, encoding="utf-8-sig"), os.path.basename(CSV_PATH))

    def replace(self, raw: pd.DataFrame, source: str, warnings: list[str] | None = None):
        standardized, local_warnings = standardize_master(raw)
        train, ols, rf = _fit_models(standardized, tune_rf=True)
        with self.lock:
            self.df = train
            self.ols = ols
            self.rf = rf
            self.source = source
            self.warnings = list(warnings or []) + local_warnings
            self.version += 1

    def snapshot(self):
        with self.lock:
            return self.df.copy(), self.ols, self.rf, self.source, list(self.warnings), self.version

    def status(self):
        df, ols, rf, source, warnings, version = self.snapshot()
        return {
            "status": "ok",
            "rows": int(len(df)),
            "data": source,
            "version": version,
            "periodStart": df["날짜"].min().strftime("%Y-%m"),
            "periodEnd": df["날짜"].max().strftime("%Y-%m"),
            "warnings": warnings,
            "modelInfo": {
                "olsAdjustedR2": float(ols.rsquared_adj),
                "rfMaxDepth": rf.get_params()["max_depth"],
                "rfMinSamplesLeaf": int(rf.get_params()["min_samples_leaf"]),
            },
        }


STATE = ModelState()


def _predict_row(model, model_name: str, row: pd.DataFrame) -> float:
    row = row[FEATURES].astype(float)
    if model_name == "ols":
        prediction = model.predict(sm.add_constant(row, has_constant="add"))
        log_value = float(prediction.iloc[0] if hasattr(prediction, "iloc") else prediction[0])
        return float(np.expm1(np.clip(log_value, 0, 20)))
    return float(model.predict(row)[0])


def _seasonal_inputs(df: pd.DataFrame, date: pd.Timestamp) -> tuple[float, float]:
    same_month = df[df["월"] == date.month]
    margin = float(same_month["할당량_여유_지수"].median())
    intensity = float(same_month["X_원단위"].median())
    yearly = df.groupby("년도")[["할당량_여유_지수", "X_원단위"]].mean().sort_index()
    margin_trend = float(yearly["할당량_여유_지수"].diff().median()) if len(yearly) > 1 else 0.0
    intensity_trend = float(yearly["X_원단위"].diff().median()) if len(yearly) > 1 else 0.0
    latest = df["날짜"].max()
    years_ahead = max(0.0, (date.year - latest.year) + (date.month - latest.month) / 12)
    return margin + margin_trend * years_ahead, intensity + intensity_trend * years_ahead


def run_forecast(mode: str, model: str, sim_margin: float, sim_intensity: float, season_option: str):
    df, ols, rf, source, warnings, version = STATE.snapshot()
    auto = mode == "auto"
    latest = df.iloc[-1]
    latest_date = df["날짜"].max()
    today = pd.Timestamp.now().normalize().replace(day=1)
    target_end = today + pd.DateOffset(months=12)
    chosen = ols if model == "ols" else rf
    rows = []
    lag = float(latest["y_가격"])
    current = latest_date + pd.DateOffset(months=1)
    while current <= target_end:
        if auto or season_option == "auto":
            season = int(current.month in (4, 5, 6))
        else:
            season = int(season_option == "force-on")
        base_margin, base_intensity = _seasonal_inputs(df, current)
        margin = base_margin if auto else base_margin + sim_margin * 10_000
        intensity = base_intensity if auto else base_intensity * (1 + sim_intensity / 100)
        step = pd.DataFrame([{
            "할당량_여유_지수": margin,
            "X_원단위": intensity,
            "배출권가격_1달전": lag,
            "정산기_시즌스위치": season,
            "리스크_2024더미": int(current.year == 2024),
        }])
        pred = max(0.0, _predict_row(chosen, model, step))
        rows.append({"날짜": current, "예측가격": pred, "여유지수": float(margin), "정산기": season})
        lag = pred
        current += pd.DateOffset(months=1)

    future = pd.DataFrame(rows)
    iso = lambda value: pd.Timestamp(value).strftime("%Y-%m-%d")
    rec = lambda row: {"date": iso(row["날짜"]), "예측가격": float(row["예측가격"]), "여유지수": float(row["여유지수"]), "정산기": int(row["정산기"])}
    next_year = [rec(row) for _, row in future[future["날짜"].between(today, target_end)].iterrows()]
    next_month = [row for row in next_year if pd.Timestamp(row["date"]) > today]
    prices = [row["예측가격"] for row in next_year]
    return {
        "history": [{"date": iso(row["날짜"]), "y": float(row["y_가격"])} for _, row in df.iterrows()],
        "latestDate": iso(latest_date), "today": iso(today), "targetEnd": iso(target_end),
        "future": [rec(row) for _, row in future.iterrows()], "next1y": next_year, "nextMonth": next_month,
        "todayPred": next_year[0] if next_year else None,
        "nextPred": next_month[0] if next_month else (next_year[0] if next_year else None),
        "lastPred": next_year[-1] if next_year else None,
        "minP": min(prices) if prices else 0, "maxP": max(prices) if prices else 0,
        "dataName": source, "dataVersion": version, "warnings": warnings,
    }


def historical_backtest(months: int = 12):
    df, *_ = STATE.snapshot()
    start = max(18, len(df) - months)
    if len(df) - start < 1:
        raise ValueError("백테스트에 사용할 데이터가 부족합니다.")
    records = []
    for index in range(start, len(df)):
        train, ols, rf = _fit_models(df.iloc[:index], tune_rf=False)
        test = df.iloc[[index]]
        for name, fitted in (("OLS", ols), ("RandomForest", rf)):
            pred = max(0.0, _predict_row(fitted, "ols" if name == "OLS" else "ai", test))
            actual = float(test["y_가격"].iloc[0])
            records.append({
                "date": test["날짜"].iloc[0].strftime("%Y-%m-%d"), "model": name,
                "forecast": pred, "actual": actual, "error": pred - actual,
                "ape": abs(pred - actual) / abs(actual) * 100 if actual else None,
            })
    detail = pd.DataFrame(records)
    metrics = []
    for name, group in detail.groupby("model"):
        metrics.append({
            "model": name,
            "MAE": float(mean_absolute_error(group["actual"], group["forecast"])),
            "RMSE": float(np.sqrt(mean_squared_error(group["actual"], group["forecast"]))),
            "MAPE": float(group["ape"].mean()), "BIAS": float(group["error"].mean()),
        })
    return {"months": len(detail["date"].unique()), "metrics": metrics, "detail": records}


app = FastAPI(title="배출권 가격 예측 MVP API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return STATE.status()


@app.get("/api/predict")
def predict(
    mode: str = Query("auto", pattern="^(auto|manual)$"),
    model: str = Query("ols", pattern="^(ols|ai)$"),
    simMargin: float = Query(0.0, ge=-30, le=30),
    simIntensity: float = Query(0.0, ge=-10, le=10),
    seasonOption: str = Query("auto", pattern="^(auto|force-on|force-off)$"),
):
    return run_forecast(mode, model, simMargin, simIntensity, seasonOption)


@app.post("/api/data/master")
async def upload_master(file: UploadFile = File(...)):
    try:
        raw = read_table_bytes(await file.read(), file.filename or "uploaded.csv")
        STATE.replace(raw, file.filename or "uploaded.csv")
        return STATE.status()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/data/auto-ingest")
async def auto_ingest(files: List[UploadFile] = File(...)):
    """한 선택창에서 받은 파일을 자동 분류해 마스터셋을 만들고 즉시 적용한다."""
    try:
        named_frames = []
        for file in files:
            name = file.filename or "uploaded.csv"
            named_frames.append((name, read_table_bytes(await file.read(), name)))
        if not named_frames:
            raise ValueError("선택된 파일이 없습니다.")

        # 완성된 표준 마스터셋 하나를 선택한 경우도 같은 창에서 바로 처리한다.
        if len(named_frames) == 1:
            name, frame = named_frames[0]
            try:
                STATE.replace(frame, name)
                result = STATE.status()
                result["classified"] = [{"filename": name, "type": "master", "label": "표준 마스터셋", "reason": "필수 컬럼과 수급지수 계산 근거 확인"}]
                return result
            except ValueError:
                pass

        master, warnings, classified = build_master_auto(named_frames)
        export = master.drop(columns=["날짜", "y_가격", "X_원단위", "배출권가격_1달전", "정산기_시즌스위치", "리스크_2024더미"], errors="ignore")
        STATE.replace(export, "자동 분류한 업로드 데이터", warnings)
        result = STATE.status()
        result["classified"] = classified
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/data/reset")
def reset_master():
    try:
        STATE.replace(pd.read_csv(CSV_PATH, encoding="utf-8-sig"), os.path.basename(CSV_PATH))
        return STATE.status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/backtest/historical")
def get_historical_backtest(months: int = Query(12, ge=3, le=24)):
    try:
        return historical_backtest(months)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/backtest/compare")
async def uploaded_backtest(forecast: UploadFile = File(...), actual: UploadFile = File(...)):
    try:
        pred = read_table_bytes(await forecast.read(), forecast.filename or "forecast.csv")
        real = read_table_bytes(await actual.read(), actual.filename or "actual.csv")
        detail, metrics = compare_forecast_actual(pred, real)
        return {"metrics": metrics, "detail": detail.replace({np.nan: None}).to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/data/build-master")
async def create_master(
    intensity: UploadFile = File(...), allocation: UploadFile = File(...),
    emissions: UploadFile = File(...), scope3: UploadFile = File(...),
    prices: List[UploadFile] = File(...), apply: bool = Query(True),
):
    try:
        master, warnings = build_master_from_raw(
            read_table_bytes(await intensity.read(), intensity.filename or "intensity.csv"),
            read_table_bytes(await allocation.read(), allocation.filename or "allocation.csv"),
            read_table_bytes(await emissions.read(), emissions.filename or "emissions.csv"),
            read_table_bytes(await scope3.read(), scope3.filename or "scope3.csv"),
            [read_table_bytes(await file.read(), file.filename or "price.csv") for file in prices],
        )
        export = master.drop(columns=["날짜", "y_가격", "X_원단위", "배출권가격_1달전", "정산기_시즌스위치", "리스크_2024더미"], errors="ignore")
        if apply:
            STATE.replace(export, "업로드 원본에서 생성한 표준 마스터셋", warnings)
        return {"rows": len(export), "warnings": warnings, "applied": apply, "csv": export.to_csv(index=False)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
