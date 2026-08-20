"""EPFM 데이터 업로드, 표준화, 원본 변환, 백테스트 도구."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MASTER_REQUIRED = ("년도", "월", "원단위", "KRX_배출권가격")


def read_table_bytes(payload: bytes, filename: str) -> pd.DataFrame:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("파일은 개당 15MB 이하여야 합니다.")
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(payload))
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(BytesIO(payload), encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV 인코딩을 판별하지 못했습니다. " + " | ".join(errors))


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def _year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")


def _column(df: pd.DataFrame, exact: Iterable[str], contains: Iterable[str] = ()) -> str:
    for name in exact:
        if name in df.columns:
            return name
    for token in contains:
        matches = [c for c in df.columns if token.lower() in str(c).lower()]
        if matches:
            return matches[0]
    raise ValueError("필수 컬럼을 찾지 못했습니다: " + ", ".join(exact))


def standardize_master(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    aliases = {"연도": "년도", "year": "년도", "month": "월", "배출권가격": "KRX_배출권가격", "수급과부족지수": "발전부문_수급과부족지수"}
    df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns}, inplace=True)
    missing = [c for c in MASTER_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("마스터셋 필수 컬럼 누락: " + ", ".join(missing))

    df["년도"] = _numeric(df["년도"])
    df["월"] = _numeric(df["월"])
    numeric_candidates = [
        "원단위", "월별가중치", "한수원_월별할당량", "수자원_월별배출량",
        "남부발전_월별Scope3", "동서발전_2023보정배출량", "발전부문_수급과부족지수",
        "할당량_여유_지수", "KRX_배출권가격", "총_할당량", "총_배출량",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = _numeric(df[col])
    invalid = df["년도"].isna() | ~df["월"].between(1, 12)
    if invalid.any():
        raise ValueError(f"년도/월이 잘못된 행이 {int(invalid.sum())}개 있습니다.")
    df[["년도", "월"]] = df[["년도", "월"]].astype(int)

    warnings: list[str] = []
    duplicated = df.duplicated(["년도", "월"], keep=False)
    if duplicated.any():
        value_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ("년도", "월")]
        df = df.groupby(["년도", "월"], as_index=False)[value_cols].mean()
        warnings.append(f"중복 년월 {int(duplicated.sum())}개 행을 평균값으로 통합했습니다.")
    df = df.sort_values(["년도", "월"]).reset_index(drop=True)

    if "월별가중치" not in df.columns:
        annual = df.groupby("년도")["원단위"].transform("sum")
        df["월별가중치"] = np.where(annual.ne(0), df["원단위"] / annual, np.nan)

    if "할당량_여유_지수" not in df.columns:
        if {"총_할당량", "총_배출량"}.issubset(df.columns):
            df["할당량_여유_지수"] = df["총_할당량"] - df["총_배출량"]
        elif "발전부문_수급과부족지수" in df.columns:
            df["할당량_여유_지수"] = -df["발전부문_수급과부족지수"]
        elif {"한수원_월별할당량", "수자원_월별배출량"}.issubset(df.columns):
            east = df.get("동서발전_2023보정배출량", pd.Series(0.0, index=df.index)).fillna(0)
            df["발전부문_수급과부족지수"] = df["수자원_월별배출량"] + east - df["한수원_월별할당량"]
            df["할당량_여유_지수"] = -df["발전부문_수급과부족지수"]
        else:
            raise ValueError("할당량 여유 지수를 계산할 수 없습니다. 수급지수 또는 할당량/배출량 컬럼이 필요합니다.")
    if "발전부문_수급과부족지수" not in df.columns:
        df["발전부문_수급과부족지수"] = -df["할당량_여유_지수"]

    feature_cols = [c for c in numeric_candidates if c in df.columns and c != "KRX_배출권가격"]
    corrected = pd.Series(False, index=df.index)
    for col in feature_cols:
        before = df[col].isna()
        if before.any():
            month_median = df.groupby("월")[col].transform("median")
            df[col] = df[col].fillna(month_median).interpolate(limit_direction="both")
            corrected |= before & df[col].notna()
    df["보정여부"] = np.where(corrected, "자동보정", "원본")
    if corrected.any():
        warnings.append(f"설명변수 결측 행 {int(corrected.sum())}개를 월 중앙값/시계열 보간으로 보정했습니다.")
    missing_price = int(df["KRX_배출권가격"].isna().sum())
    if missing_price:
        warnings.append(f"KRX 가격이 없는 {missing_price}개 행은 모델 학습에서 제외됩니다.")

    df["날짜"] = pd.to_datetime(df["년도"].astype(str) + "-" + df["월"].astype(str).str.zfill(2) + "-01")
    df["y_가격"] = df["KRX_배출권가격"]
    df["X_원단위"] = df["원단위"]
    df["배출권가격_1달전"] = df["y_가격"].shift(1)
    df["정산기_시즌스위치"] = df["월"].isin([4, 5, 6]).astype(int)
    df["리스크_2024더미"] = (df["년도"] == 2024).astype(int)
    return df, warnings


def build_master_from_raw(
    intensity: pd.DataFrame,
    allocation: pd.DataFrame,
    emissions: pd.DataFrame,
    scope3: pd.DataFrame,
    price_frames: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    intensity = intensity.copy()
    year_col = _column(intensity, ("구분", "연도"), ("구분", "연도"))
    intensity["_년도"] = _year(intensity[year_col])
    month_cols = {m: _column(intensity, (f"{m}월", str(m)), (f"{m}월",)) for m in range(1, 13)}
    melted = intensity.dropna(subset=["_년도"]).melt(
        id_vars=["_년도"], value_vars=list(month_cols.values()), var_name="_월", value_name="원단위"
    )
    melted["년도"] = melted["_년도"].astype(int)
    melted["월"] = melted["_월"].map({v: k for k, v in month_cols.items()})
    melted["원단위"] = _numeric(melted["원단위"])
    melted = melted.dropna(subset=["원단위"])[["년도", "월", "원단위"]]
    total = melted.groupby("년도")["원단위"].transform("sum")
    melted["월별가중치"] = melted["원단위"] / total

    a_year = _column(allocation, ("연도", "년도"), ("연도", "년도"))
    a_value = _column(allocation, ("무상할당량", "총_할당량", "할당량"), ("무상할당량", "할당량"))
    annual_a = pd.DataFrame({"년도": _year(allocation[a_year]), "무상할당량": _numeric(allocation[a_value])}).groupby("년도", as_index=False).mean()

    e_year = _column(emissions, ("연도", "년도"), ("연도", "년도"))
    e_value = _column(emissions, ("배출량(tCO2)", "배출량", "총_배출량"), ("배출량(tCO2)", "배출량"))
    annual_e = pd.DataFrame({"년도": _year(emissions[e_year]), "연간배출량": _numeric(emissions[e_value])}).groupby("년도", as_index=False).mean()

    s_year = _column(scope3, ("년도", "연도"), ("년도", "연도"))
    s_value = _column(scope3, ("스코프(Scope)3 배출량", "Scope3 배출량"), ("Scope)3", "Scope3", "scope3"))
    annual_s = pd.DataFrame({"년도": _year(scope3[s_year]), "연간Scope3": _numeric(scope3[s_value])}).groupby("년도", as_index=False).mean()

    prices = []
    for frame in price_frames:
        date_col = _column(frame, ("일자", "날짜", "date"), ("일자", "날짜"))
        value_col = _column(frame, ("종가", "KRX_배출권가격", "가격"), ("종가", "가격"))
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        part = pd.DataFrame({"년도": dates.dt.year, "월": dates.dt.month, "KRX_배출권가격": _numeric(frame[value_col])}).dropna()
        prices.append(part)
    if not prices:
        raise ValueError("KRX 가격 파일이 한 개 이상 필요합니다.")
    monthly_price = pd.concat(prices, ignore_index=True).groupby(["년도", "월"], as_index=False)["KRX_배출권가격"].mean()

    master = melted.merge(annual_a, on="년도", how="left").merge(annual_e, on="년도", how="left").merge(annual_s, on="년도", how="left")
    master = master.merge(monthly_price, on=["년도", "월"], how="left")
    first = int((monthly_price["년도"] * 100 + monthly_price["월"]).min())
    last = int((monthly_price["년도"] * 100 + monthly_price["월"]).max())
    master = master[(master["년도"] * 100 + master["월"]).between(first, last)].copy()
    for col in ("무상할당량", "연간배출량", "연간Scope3"):
        master[col] = master[col].interpolate(limit_direction="both")
    master["한수원_월별할당량"] = master["무상할당량"] * master["월별가중치"]
    master["수자원_월별배출량"] = master["연간배출량"] * master["월별가중치"]
    master["남부발전_월별Scope3"] = master["연간Scope3"] * master["월별가중치"]
    master["동서발전_2023보정배출량"] = 0.0
    master.drop(columns=["무상할당량", "연간배출량", "연간Scope3"], inplace=True)
    return standardize_master(master)


SOURCE_LABELS = {
    "price": "배출권 가격",
    "intensity": "원단위",
    "supply": "공급(할당량)",
    "demand": "수요(배출량)",
    "scope3": "Scope3(보조지표)",
}


def classify_source(df: pd.DataFrame, filename: str = "") -> tuple[str, str]:
    """파일명과 컬럼 구조를 함께 사용해 원본 데이터의 역할을 판별한다."""
    columns = [str(c).strip() for c in df.columns]
    joined = " ".join(columns).lower()
    name = filename.lower()
    month_count = sum(f"{month}월" in columns for month in range(1, 13))

    if month_count >= 6 and ("원단위" in joined or "원단위" in name):
        return "intensity", f"월별 원단위 컬럼 {month_count}개 확인"
    if (any(token in joined for token in ("종가", "krx_배출권가격", "가중평균")) and
            (any(token in joined for token in ("일자", "날짜")) or {"년도", "월"}.issubset(columns))):
        return "price", "거래일/년월과 가격 컬럼 확인"
    if "scope3" in name or "scope)3" in joined or "scope3" in joined:
        return "scope3", "Scope3 식별자 확인"
    if any(column in columns for column in ("배출량(tCO2)", "총_배출량", "배출량")) or "배출량" in name:
        return "demand", "배출량 컬럼 확인"
    if any(column in columns for column in ("무상할당량", "총_할당량", "할당량")) or "할당량" in name:
        return "supply", "할당량 컬럼 확인"
    raise ValueError(f"'{filename}'의 데이터 종류를 판별하지 못했습니다. 컬럼: {', '.join(columns[:8])}")


def _intensity_monthly(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    year_col = _column(work, ("구분", "연도"), ("구분", "연도"))
    work["_년도"] = _year(work[year_col])
    month_cols = {m: _column(work, (f"{m}월", str(m)), (f"{m}월",)) for m in range(1, 13)}
    melted = work.dropna(subset=["_년도"]).melt(
        id_vars=["_년도"], value_vars=list(month_cols.values()), var_name="_월", value_name="원단위"
    )
    melted["년도"] = melted["_년도"].astype(int)
    melted["월"] = melted["_월"].map({v: k for k, v in month_cols.items()})
    melted["원단위"] = _numeric(melted["원단위"])
    melted = melted.dropna(subset=["원단위"])[["년도", "월", "원단위"]]
    total = melted.groupby("년도")["원단위"].transform("sum")
    melted["월별가중치"] = melted["원단위"] / total
    return melted


def _price_monthly(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for frame in frames:
        value_col = _column(frame, ("KRX_배출권가격", "종가", "가격", "가중평균"), ("종가", "가격", "가중평균"))
        if {"년도", "월"}.issubset(frame.columns):
            year, month = _numeric(frame["년도"]), _numeric(frame["월"])
        else:
            date_col = _column(frame, ("일자", "날짜", "date"), ("일자", "날짜"))
            dates = pd.to_datetime(frame[date_col], errors="coerce")
            year, month = dates.dt.year, dates.dt.month
        parts.append(pd.DataFrame({"년도": year, "월": month, "KRX_배출권가격": _numeric(frame[value_col])}).dropna())
    result = pd.concat(parts, ignore_index=True)
    result[["년도", "월"]] = result[["년도", "월"]].astype(int)
    return result.groupby(["년도", "월"], as_index=False)["KRX_배출권가격"].mean()


def _quantity_column(df: pd.DataFrame, kind: str) -> str:
    if kind == "supply":
        return _column(df, ("무상할당량", "총_할당량", "할당량"), ("무상할당량", "할당량", "할당"))
    return _column(df, ("배출량(tCO2)", "총_배출량", "배출량"), ("배출량(tCO2)", "온실가스 배출량", "배출량"))


def _aggregate_quantity(frames: Iterable[pd.DataFrame], kind: str, timeline: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=timeline.index)
    for frame in frames:
        value_col = _quantity_column(frame, kind)
        year_col = _column(frame, ("연도", "년도", "구분"), ("연도", "년도", "구분"))
        years = _year(frame[year_col])
        values = _numeric(frame[value_col])
        if "월" in frame.columns:
            months = _numeric(frame["월"])
            monthly = pd.DataFrame({"년도": years, "월": months, "값": values}).dropna()
            monthly[["년도", "월"]] = monthly[["년도", "월"]].astype(int)
            monthly = monthly.groupby(["년도", "월"], as_index=False)["값"].sum()
            mapped = timeline[["년도", "월"]].merge(monthly, on=["년도", "월"], how="left")["값"].fillna(0)
        else:
            annual = pd.DataFrame({"년도": years, "값": values}).dropna().groupby("년도", as_index=False)["값"].sum()
            annual["년도"] = annual["년도"].astype(int)
            mapped = timeline[["년도"]].merge(annual, on="년도", how="left")["값"] * timeline["월별가중치"]
            mapped = mapped.fillna(0)
        total += mapped.to_numpy()
    return total


def build_master_auto(named_frames: Iterable[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """한 번에 받은 파일들을 자동 분류하고 표준 마스터셋을 생성한다."""
    buckets: dict[str, list[pd.DataFrame]] = {key: [] for key in SOURCE_LABELS}
    classified = []
    for filename, frame in named_frames:
        kind, reason = classify_source(frame, filename)
        buckets[kind].append(frame)
        classified.append({"filename": filename, "type": kind, "label": SOURCE_LABELS[kind], "reason": reason})

    required = {"price": "배출권 가격", "intensity": "원단위", "supply": "공급(할당량)", "demand": "수요(배출량)"}
    missing = [label for key, label in required.items() if not buckets[key]]
    if missing:
        raise ValueError("필수 데이터가 부족합니다: " + ", ".join(missing))
    if len(buckets["intensity"]) > 1:
        raise ValueError("원단위 파일은 한 개만 선택해 주세요.")

    timeline = _intensity_monthly(buckets["intensity"][0])
    prices = _price_monthly(buckets["price"])
    first = int((prices["년도"] * 100 + prices["월"]).min())
    last = int((prices["년도"] * 100 + prices["월"]).max())
    period = timeline["년도"] * 100 + timeline["월"]
    timeline = timeline[period.between(first, last)].copy().reset_index(drop=True)
    timeline["총_할당량"] = _aggregate_quantity(buckets["supply"], "supply", timeline)
    timeline["총_배출량"] = _aggregate_quantity(buckets["demand"], "demand", timeline)
    timeline = timeline.merge(prices, on=["년도", "월"], how="left")

    if buckets["scope3"]:
        scope = _aggregate_quantity(buckets["scope3"], "demand", timeline)
        timeline["남부발전_월별Scope3"] = scope
    standardized, warnings = standardize_master(timeline)
    if buckets["scope3"]:
        warnings.append("Scope3는 보조지표로 보존하고 할당량 수급 잔액에는 합산하지 않았습니다.")
    return standardized, warnings, classified


def _monthly_values(df: pd.DataFrame, candidates: Iterable[str]) -> pd.DataFrame:
    work = df.copy()
    if {"년도", "월"}.issubset(work.columns):
        year, month = _numeric(work["년도"]), _numeric(work["월"])
    else:
        date_col = _column(work, ("날짜", "일자", "예측 월"), ("날짜", "일자", "월"))
        text = work[date_col].astype(str).str.replace("년 ", "-", regex=False).str.replace("월", "", regex=False)
        dates = pd.to_datetime(text, errors="coerce")
        year, month = dates.dt.year, dates.dt.month
    value_col = _column(work, tuple(candidates), ("가격", "예측", "단가", "종가"))
    out = pd.DataFrame({"년도": year, "월": month, "값": _numeric(work[value_col])}).dropna()
    out[["년도", "월"]] = out[["년도", "월"]].astype(int)
    return out.groupby(["년도", "월"], as_index=False)["값"].mean()


def compare_forecast_actual(forecast: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    pred = _monthly_values(forecast, ("예측가격", "예측 단가(원)", "예측 단가", "forecast")).rename(columns={"값": "예측가격"})
    real = _monthly_values(actual, ("KRX_배출권가격", "실제가격", "종가", "actual")).rename(columns={"값": "실제가격"})
    merged = pred.merge(real, on=["년도", "월"], how="inner")
    if merged.empty:
        raise ValueError("예측 파일과 실제가 파일에서 일치하는 년월을 찾지 못했습니다.")
    merged["오차"] = merged["예측가격"] - merged["실제가격"]
    merged["절대오차"] = merged["오차"].abs()
    merged["절대오차율(%)"] = np.where(merged["실제가격"].ne(0), merged["절대오차"] / merged["실제가격"].abs() * 100, np.nan)
    metrics = {
        "MAE": float(merged["절대오차"].mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(merged["오차"])))),
        "MAPE": float(merged["절대오차율(%)"].mean()),
        "BIAS": float(merged["오차"].mean()),
    }
    return merged, metrics
