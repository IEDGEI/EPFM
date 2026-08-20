import sys
import subprocess
import io
import os
from datetime import datetime

# [자동 안전장치] 필수 패키지 자동 설치
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from sklearn.ensemble import RandomForestRegressor
    import openpyxl
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels scipy pandas numpy scikit-learn streamlit openpyxl"])
    import streamlit as st
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from sklearn.ensemble import RandomForestRegressor
    import openpyxl

# [최우선 레이아웃 설정]
st.set_page_config(page_title="남부발전 배출권 가격 예측 MVP", layout="wide")

# -------------------------------------------------------------
# 0. 세션 상태(Session State) 초기화
# -------------------------------------------------------------
if 'uploaded_registry' not in st.session_state:
    st.session_state.uploaded_registry = {}  # { 'PRICE': {'filename': ..., 'df': ...}, ... }

# -------------------------------------------------------------
# 1. 파일 자동 판별 및 데이터 로드 함수
# -------------------------------------------------------------
def classify_and_parse(uploaded_file):
    """업로드된 파일의 컬럼을 분석하여 데이터 유형 자동 분류"""
    filename = uploaded_file.name
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp949')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            
    cols = set(df.columns)
    
    # 1. 완성형 마스터셋
    if ({'할당량_여유_지수', '원단위', 'KRX_배출권가격'}.issubset(cols) or 
        {'할당량여유지수', '원단위', 'KRX_배출권가격'}.issubset(cols) or
        {'발전부문_수급과부족지수', '원단위', 'KRX_배출권가격'}.issubset(cols)):
        return "MASTERSET", "완성형 표준 마스터셋", df
        
    # 2. KRX 배출권 거래 가격
    if {'일자', '종가'}.issubset(cols) or 'KRX_배출권가격' in cols:
        return "PRICE", "KRX 배출권 거래 가격", df
        
    # 3. 발전 온실가스 원단위
    if {'구분', '1월', '2월'}.issubset(cols) or ('원단위' in cols and '월' in cols):
        return "INTENSITY", "온실가스 배출량 원단위", df
        
    # 4. 한수원 할당량 (공급)
    if '무상할당량' in cols:
        return "ALLOCATION", "한국수력원자력 무상할당량", df
        
    # 5. 수자원공사 배출량 (수요)
    if '배출량(tCO2)' in cols or '수자원공사_배출량' in cols or '배출량' in cols:
        return "EMISSION", "한국수자원공사 배출량", df
        
    return "UNKNOWN", "알 수 없는 파일 형식", df


@st.cache_data
def load_base_data():
    """기본 표준 마스터셋 로드"""
    if os.path.exists("발전부문_배출권_분석_마스터셋.csv"):
        df = pd.read_csv("발전부문_배출권_분석_마스터셋.csv", encoding='utf-8-sig')
    else:
        # 백업용 가상 기본 마스터셋 생성
        dates = pd.date_range(start="2021-01-01", end="2024-12-31", freq='MS')
        df = pd.DataFrame({
            '년도': dates.year,
            '월': dates.month,
            '원단위': np.linspace(0.81, 0.72, len(dates)),
            '월별가중치': [0.083] * len(dates),
            '한수원_월별할당량': [200000] * len(dates),
            '수자원_월별배출량': [230000] * len(dates),
            'KRX_배출권가격': np.linspace(22000, 8500, len(dates))
        })
        df['할당량_여유_지수'] = (df['한수원_월별할당량'] - df['수자원_월별배출량']) / 10000.0

    columns_list = list(df.columns)
    if '총_할당량' in df.columns and '총_배출량' in df.columns:
        df['할당량_여유_지수'] = (df['총_할당량'] - df['총_배출량']) / 10000.0
    elif '발전부문_수급과부족지수' in df.columns:
        df['할당량_여유_지수'] = (df['발전부문_수급과부족지수'] * -1) / 10000.0

    df['y_가격'] = df['KRX_배출권가격'] if 'KRX_배출권가격' in df.columns else df[columns_list[-1]]
    df['X_원단위'] = df['원단위'] if '원단위' in df.columns else df[columns_list[1]]
    return df


def build_active_masterset(base_df, uploaded_dict):
    """업로드된 사용자 데이터를 병합하여 최종 활성 마스터셋 조립"""
    df_out = base_df.copy()
    
    # 1. 완성형 마스터셋 교체
    if 'MASTERSET' in uploaded_dict:
        df_custom = uploaded_dict['MASTERSET']['df'].copy()
        if '할당량여유지수' in df_custom.columns:
            df_custom['할당량_여유_지수'] = df_custom['할당량여유지수']
        elif '발전부문_수급과부족지수' in df_custom.columns:
            df_custom['할당량_여유_지수'] = df_custom['발전부문_수급과부족지수'] * -1
        df_custom['y_가격'] = df_custom['KRX_배출권가격']
        df_custom['X_원단위'] = df_custom['원단위']
        df_out = df_custom
    else:
        # 2. 가격 데이터 업데이트
        if 'PRICE' in uploaded_dict:
            df_p = uploaded_dict['PRICE']['df'].copy()
            if {'일자', '종가'}.issubset(df_p.columns):
                df_p['일자'] = pd.to_datetime(df_p['일자'])
                df_p['년도'] = df_p['일자'].dt.year
                df_p['월'] = df_p['일자'].dt.month
                df_p['종가'] = pd.to_numeric(df_p['종가'].astype(str).str.replace(',', ''), errors='coerce')
                p_monthly = df_p.groupby(['년도', '월'])['종가'].mean().reset_index().rename(columns={'종가': 'KRX_배출권가격'})
                df_out = df_out.merge(p_monthly, on=['년도', '월'], how='outer', suffixes=('', '_new'))
                if 'KRX_배출권가격_new' in df_out.columns:
                    df_out['KRX_배출권가격'] = df_out['KRX_배출권가격_new'].combine_first(df_out['KRX_배출권가격'])
                    df_out.drop(columns=['KRX_배출권가격_new'], inplace=True)

        # 3. 원단위 데이터 업데이트
        if 'INTENSITY' in uploaded_dict:
            df_i = uploaded_dict['INTENSITY']['df'].copy()
            if '구분' in df_i.columns:
                df_i['년도'] = pd.to_numeric(df_i['구분'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
                month_cols = [f"{i}월" for i in range(1, 13) if f"{i}월" in df_i.columns]
                if month_cols:
                    df_melt = df_i.melt(id_vars=['년도'], value_vars=month_cols, var_name='월', value_name='원단위')
                    df_melt['월'] = df_melt['월'].str.replace('월', '').astype(int)
                    df_melt['원단위'] = pd.to_numeric(df_melt['원단위'].astype(str).str.replace(',', ''), errors='coerce')
                    df_out = df_out.merge(df_melt[['년도', '월', '원단위']], on=['년도', '월'], how='outer', suffixes=('', '_new'))
                    if '원단위_new' in df_out.columns:
                        df_out['원단위'] = df_out['원단위_new'].combine_first(df_out['원단위'])
                        df_out.drop(columns=['원단위_new'], inplace=True)

        # 4. 할당량 & 배출량 업데이트
        if 'ALLOCATION' in uploaded_dict:
            df_al = uploaded_dict['ALLOCATION']['df'].copy()
            yr_col = '연도' if '연도' in df_al.columns else '년도'
            df_al[yr_col] = pd.to_numeric(df_al[yr_col], errors='coerce')
            df_al['무상할당량'] = pd.to_numeric(df_al['무상할당량'].astype(str).str.replace(',', ''), errors='coerce')
            for _, r in df_al.dropna(subset=[yr_col, '무상할당량']).iterrows():
                y_val = int(r[yr_col])
                if '월별가중치' in df_out.columns:
                    df_out.loc[df_out['년도'] == y_val, '한수원_월별할당량'] = r['무상할당량'] * df_out.loc[df_out['년도'] == y_val, '월별가중치']

        if 'EMISSION' in uploaded_dict:
            df_em = uploaded_dict['EMISSION']['df'].copy()
            yr_col = '연도' if '연도' in df_em.columns else '년도'
            em_col = '배출량(tCO2)' if '배출량(tCO2)' in df_em.columns else ('수자원공사_배출량' if '수자원공사_배출량' in df_em.columns else '배출량')
            df_em[yr_col] = pd.to_numeric(df_em[yr_col], errors='coerce')
            df_em[em_col] = pd.to_numeric(df_em[em_col].astype(str).str.replace(',', ''), errors='coerce')
            for _, r in df_em.dropna(subset=[yr_col, em_col]).iterrows():
                y_val = int(r[yr_col])
                if '월별가중치' in df_out.columns:
                    df_out.loc[df_out['년도'] == y_val, '수자원_월별배출량'] = r[em_col] * df_out.loc[df_out['년도'] == y_val, '월별가중치']

    # 결측치 보정 및 시차/더미 변수 생성
    df_out['년도'] = df_out['년도'].astype(int)
    df_out['월'] = df_out['월'].astype(int)
    df_out = df_out.sort_values(['년도', '월']).reset_index(drop=True)
    
    df_out['원단위'] = df_out['원단위'].interpolate(method='linear').bfill().ffill()
    df_out['KRX_배출권가격'] = df_out['KRX_배출권가격'].interpolate(method='linear').bfill().ffill()
    
    if '할당량_여유_지수' not in df_out.columns or df_out['할당량_여유_지수'].isna().all():
        if '한수원_월별할당량' in df_out.columns and '수자원_월별배출량' in df_out.columns:
            df_out['할당량_여유_지수'] = (df_out['한수원_월별할당량'] - df_out['수자원_월별배출량']) / 10000.0
        else:
            df_out['할당량_여유_지수'] = 0.0
            
    df_out['할당량_여유_지수'] = df_out['할당량_여유_지수'].interpolate(method='linear').bfill().ffill()
    df_out['y_가격'] = df_out['KRX_배출권가격']
    df_out['X_원단위'] = df_out['원단위']
    df_out['배출권가격_1달전'] = df_out['y_가격'].shift(1)
    df_out['정산기_시즌스위치'] = df_out['월'].isin([4, 5, 6]).astype(int)
    df_out['리스크_2024더미'] = (df_out['년도'] == 2024).astype(int)
    df_out['날짜'] = pd.to_datetime(df_out['년도'].astype(str) + '-' + df_out['월'].astype(str) + '-01')
    
    return df_out.dropna(subset=['날짜', 'y_가격', '할당량_여유_지수', 'X_원단위', '배출권가격_1달전']).reset_index(drop=True)


# -------------------------------------------------------------
# 2. 동적 데이터셋 조립 및 모델 트레이닝
# -------------------------------------------------------------
base_df = load_base_data()
df = build_active_masterset(base_df, st.session_state.uploaded_registry)

# 분석용 독립변수(X) 및 종속변수(y)
X = pd.DataFrame()
X['할당량_여유_지수'] = df['할당량_여유_지수']
X['X_원단위'] = df['X_원단위']
X['배출권가격_1달전'] = df['배출권가격_1달전']
X['정산기_시즌스위치'] = df['정산기_시즌스위치']
X['리스크_2024더미'] = df['리스크_2024더미']
y = df['y_가격']

# OLS 다중선형회귀 모델
X_with_constant = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_constant).fit()

# AI Random Forest 모델
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# -------------------------------------------------------------
# 3. 사이드바 설정
# -------------------------------------------------------------
st.title("📊 발전부문 배출권 가격 예측 MVP 대시보드")
st.caption("국립창원대학교 identity 팀 (정시진, 지현서)")
st.markdown("---")

st.sidebar.header("⚙️ 시뮬레이션 설정")

mode_choice = st.sidebar.radio(
    "시뮬레이션 방식 선택",
    ["자동 가격 예측", "사용자 수치 조절"]
)

model_choice = st.sidebar.selectbox(
    "1) 예측 알고리즘 엔진 선택",
    ["OLS 수식 모델", "AI 머신러닝 모델"]
)

# 오늘 날짜 기준점 설정
today_date = pd.Timestamp.now().normalize().replace(day=1)
target_end_date = today_date + pd.DateOffset(months=12)

latest_real_data = df.iloc[-1].copy()
latest_data_date = df['날짜'].max()

st.sidebar.success(f"📅 **실행 기준일**: {today_date.strftime('%Y년 %m월 %d일')}\n\n🎯 **예측 범위**: 오늘 ➔ {target_end_date.strftime('%Y년 %m월')} (향후 1년간)")

if "자동" in mode_choice:
    sim_margin = 0.0
    sim_intensity = 0.0
    sim_season_option = "달에 따라 자동 적용"
    st.sidebar.info("📌 **자동 가격 예측 가동 중**\n\n과거 계절성 발전 패턴과 4~6월 정산기 매물 수급을 AI 및 통계 모형이 자동으로 추정하여 예측합니다.")
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("2) 사용자 수치 조절")
    sim_margin = st.sidebar.slider("할당량 여유 지수 변동 (만 톤)", -30.0, 30.0, 0.0, 0.1)
    sim_intensity = st.sidebar.slider("발전 온실가스 원단위 변동 (%)", -10.0, 10.0, 0.0, 0.1)
    sim_season_option = st.sidebar.selectbox("정산기 시즌 적용 방식", ["달에 따라 자동 적용", "전 기간 정산기 강제 적용", "전 기간 일반달 강제 적용"])

# -------------------------------------------------------------
# 4. 향후 1년 연쇄 릴레이 연산 로직
# -------------------------------------------------------------
future_predictions = []
current_lag_price = latest_real_data['y_가격']
current_date = latest_data_date + pd.DateOffset(months=1)

while current_date <= target_end_date:
    step_month = current_date.month
    
    if "자동" in mode_choice or sim_season_option == "달에 따라 자동 적용":
        step_season = 1 if step_month in [4, 5, 6] else 0
    elif sim_season_option == "전 기간 정산기 강제 적용":
        step_season = 1
    else:
        step_season = 0
        
    hist_m = df[df['날짜'].dt.month == step_month]
    base_intensity = hist_m['X_원단위'].mean() if len(hist_m) > 0 else latest_real_data['X_원단위']
    base_margin = hist_m['할당량_여유_지수'].mean() if len(hist_m) > 0 else latest_real_data['할당량_여유_지수']

    year_diff = max(0, current_date.year - latest_data_date.year)
    yearly_tightening = year_diff * 0.4

    if "자동" in mode_choice:
        step_intensity = base_intensity
        step_margin = base_margin - yearly_tightening
    else:
        step_margin = base_margin + sim_margin - yearly_tightening
        step_intensity = base_intensity * (1 + sim_intensity / 100)
        
    base_input = pd.DataFrame([{
        '할당량_여유_지수': base_margin,
        'X_원단위': base_intensity,
        '배출권가격_1달전': current_lag_price,
        '정산기_시즌스위치': step_season,
        '리스크_2024더미': 1 if current_date.year == 2024 else 0
    }])

    step_input = pd.DataFrame([{
        '할당량_여유_지수': step_margin,
        'X_원단위': step_intensity,
        '배출권가격_1달전': current_lag_price,
        '정산기_시즌스위치': step_season,
        '리스크_2024더미': 1 if current_date.year == 2024 else 0
    }])
    
    if model_choice == "OLS 수식 모델":
        step_input_const = sm.add_constant(step_input, has_constant='add')
        step_input_const['const'] = 1.0
        step_input_const = step_input_const[['const', '할당량_여유_지수', 'X_원단위', '배출권가격_1달전', '정산기_시즌스위치', '리스크_2024더미']]
        pred_price = ols_model.predict(step_input_const)[0]
    else:
        if "자동" in mode_choice:
            pred_price = rf_model.predict(step_input)[0]
        else:
            rf_base_pred = rf_model.predict(base_input)[0]
            rf_raw_pred = rf_model.predict(step_input)[0]
            smooth_delta = (sim_intensity * (rf_base_pred * 0.015)) - (sim_margin * 45.0)
            if rf_raw_pred != rf_base_pred:
                pred_price = rf_raw_pred + (smooth_delta * 0.3)
            else:
                pred_price = rf_base_pred + smooth_delta
        
    pred_price = max(0, pred_price)
    
    future_predictions.append({
        '날짜': current_date,
        '예측가격': pred_price,
        '원단위': step_intensity,
        '여유지수': step_margin,
        '정산기': step_season
    })
    
    current_lag_price = pred_price
    current_date += pd.DateOffset(months=1)

df_future = pd.DataFrame(future_predictions)
df_next_1year = df_future[df_future['날짜'] >= today_date].reset_index(drop=True)
df_next_month = df_future[df_future['날짜'] > today_date].reset_index(drop=True)

# -------------------------------------------------------------
# 5. 메인 레이아웃 구성 (왼쪽 col1: 데이터 관리 + 명세표 / 오른쪽 col2: 차트)
# -------------------------------------------------------------
col1, col2 = st.columns([1, 1.4])

with col1:
    # ---------------------------------------------------------
    # [새로 구현된 파트] 📁 마스터셋 데이터 관리 섹션
    # ---------------------------------------------------------
    st.subheader("📁 마스터셋 데이터 관리")
    
    uploaded_files = st.file_uploader(
        "최신 공공데이터 파일 업로드 (CSV / Excel)",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="단일 또는 복수 파일을 드래그하여 업로드하면 자동으로 데이터 유형을 식별합니다."
    )
    
    if uploaded_files:
        new_upload_occurred = False
        for f in uploaded_files:
            ftype, label, parsed_df = classify_and_parse(f)
            if ftype != "UNKNOWN":
                if ftype not in st.session_state.uploaded_registry or st.session_state.uploaded_registry[ftype]['filename'] != f.name:
                    st.session_state.uploaded_registry[ftype] = {
                        'filename': f.name,
                        'label': label,
                        'df': parsed_df
                    }
                    new_upload_occurred = True
        if new_upload_occurred:
            st.rerun()

    # 데이터 적용 상태 현황 배지
    is_custom = len(st.session_state.uploaded_registry) > 0
    if is_custom:
        st.success(f"🟢 **사용자 커스텀 마스터셋 적용 중** (총 {len(df)}개 행 학습)")
    else:
        st.info(f"🔵 **기본 표준 마스터셋 사용 중** (총 {len(df)}개 행 학습)")

    # 4대 핵심 지표별 반영 현황 및 개별 제거(❌) 인터페이스
    status_items = [
        ("PRICE", "📈 배출권 가격 (KRX)"),
        ("INTENSITY", "⚡ 온실가스 원단위"),
        ("ALLOCATION", "🏭 무상할당량 (공급)"),
        ("EMISSION", "💨 온실가스 배출량 (수요)")
    ]

    with st.container():
        st.markdown("##### 📌 세부 데이터 항목 반영 현황")
        for key, name in status_items:
            c_name, c_btn = st.columns([3, 1])
            if key in st.session_state.uploaded_registry:
                item_info = st.session_state.uploaded_registry[key]
                c_name.markdown(f"• **{name}**: `{item_info['filename']}`")
                if c_btn.button("❌ 삭제", key=f"del_{key}", help=f"{name} 데이터를 기본값으로 원복합니다."):
                    del st.session_state.uploaded_registry[key]
                    st.rerun()
            else:
                c_name.markdown(f"• **{name}**: *기본 데이터 적용 중*")
                c_btn.markdown("<span style='color:gray; font-size:12px;'>기본값</span>", unsafe_allow_html=True)

    # 전체 초기화 버튼
    if is_custom:
        if st.button("↺ 기본 마스터셋으로 전체 초기화", use_container_width=True):
            st.session_state.uploaded_registry.clear()
            st.rerun()

    st.markdown("---")

    # ---------------------------------------------------------
    # 🎯 1년 예측 요약 메트릭
    # ---------------------------------------------------------
    st.subheader(f"🎯 실행일({today_date.strftime('%Y-%m')}) 기준 향후 1년 예측 요약")
    
    today_pred = df_next_1year.iloc[0] if len(df_next_1year) > 0 else df_future.iloc[-1]
    next_pred = df_next_month.iloc[0] if len(df_next_month) > 0 else df_next_1year.iloc[0]
    last_pred = df_next_1year.iloc[-1] if len(df_next_1year) > 0 else df_future.iloc[-1]
    
    min_p = df_next_1year['예측가격'].min() if len(df_next_1year) > 0 else today_pred['예측가격']
    max_p = df_next_1year['예측가격'].max() if len(df_next_1year) > 0 else today_pred['예측가격']
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            label=f"📊 차월 ({next_pred['날짜'].strftime('%Y년 %m월')}) 예측가",
            value=f"{int(round(next_pred['예측가격'])):,} 원",
            delta=f"{int(round(next_pred['예측가격'] - today_pred['예측가격'])):,} 원 (당월 대비)"
        )
    with m2:
        st.metric(
            label=f"🔮 1년 후 ({last_pred['날짜'].strftime('%Y년 %m월')}) 예측가",
            value=f"{int(round(last_pred['예측가격'])):,} 원",
            delta=f"{int(round(last_pred['예측가격'] - next_pred['예측가격'])):,} 원 (차월 대비)"
        )

    st.info(f"💡 **향후 1년간 시뮬레이션 변동 범위**: 최저 **{int(round(min_p)):,} 원** ~ 최고 **{int(round(max_p)):,} 원** (4~6월 정산기 매물 압박 요인 수록)")

    st.markdown("---")
    
    # ---------------------------------------------------------
    # 📋 1년간 월별 예측 명세표 & 엑셀 다운로드
    # ---------------------------------------------------------
    st.subheader("📋 실행일 기준 1년간 월별 예측 명세표")
    
    df_disp = df_next_1year.copy()
    df_disp['예측 월'] = df_disp['날짜'].dt.strftime('%Y년 %m월')
    df_disp['예측 단가(원)'] = df_disp['예측가격'].apply(lambda x: int(round(x)))
    df_disp['할당 여유지수(만톤)'] = df_disp['여유지수'].apply(lambda x: int(round(x)))
    df_disp['정산기 여부'] = df_disp['정산기'].apply(lambda x: "⭕ 매물폭탄 (4~6월)" if x==1 else "❌ 평달")
    
    show_table = df_disp[['예측 월', '예측 단가(원)', '할당 여유지수(만톤)', '정산기 여부']].copy()
    show_table_view = show_table.copy()
    show_table_view['예측 단가'] = show_table_view['예측 단가(원)'].apply(lambda x: f"{x:,} 원")
    show_table_view['할당 여유지수'] = show_table_view['할당 여유지수(만톤)'].apply(lambda x: f"{x:,} 만톤")
    
    st.dataframe(show_table_view[['예측 월', '예측 단가', '할당 여유지수', '정산기 여부']], use_container_width=True, height=240)

    # 엑셀 다운로드 버퍼
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        show_table.to_excel(writer, index=False, sheet_name='1년_가격예측_명세표')
    excel_data = excel_buffer.getvalue()

    st.download_button(
        label="📥 1년 예측 결과 엑셀(.xlsx) 파일 다운로드",
        data=excel_data,
        file_name=f"남부발전_배출권_1년_가격예측_결과_{today_date.strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with col2:
    # ---------------------------------------------------------
    # 우측 col2: 시계열 예측 차트
    # ---------------------------------------------------------
    st.subheader(f"📈 과거 실제 가격 + 실행일 기준 1년 후({target_end_date.strftime('%Y-%m')}) 예측 곡선")
    
    chart_data = df.set_index('날짜')[['y_가격']].rename(columns={'y_가격': '과거 실제 가격'})
    chart_data['AI 연속 예측 곡선'] = np.nan
    
    chart_data.loc[latest_data_date, 'AI 연속 예측 곡선'] = df['y_가격'].iloc[-1]
    
    for _, row in df_future.iterrows():
        chart_data.loc[row['날짜'], 'AI 연속 예측 곡선'] = row['예측가격']
        
    chart_data = chart_data.sort_index()
    st.line_chart(chart_data)
    
    # 모델 학습 지표 요약 카드
    st.markdown("##### 🔬 현재 활성 모형 요약 지표")
    stat_c1, stat_c2 = st.columns(2)
    stat_c1.metric("통계 모형 설명력 (R²)", f"{ols_model.rsquared:.3f}")
    stat_c2.metric("학습 데이터 누적 기간", f"{len(df)} 개월")