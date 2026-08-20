# 발전부문 배출권 가격 예측 MVP

국립창원대학교 identity 팀 · KRDS(대한민국 정부 디자인 시스템) 스타일 적용

**프론트엔드(React) + 백엔드(Python)로 나뉜 하나의 클라이언트-서버 시스템**입니다.
프론트엔드는 화면만 담당하고, 실제 예측 계산(OLS·RandomForest)은 백엔드가 sklearn으로
수행합니다. 프론트엔드는 백엔드의 API를 호출해 결과를 받아 그립니다.

```
deliverable/
├─ backend/    ← Python 서버 (FastAPI). 예측 계산 + 프론트엔드 서빙
│  ├─ app.py
│  ├─ requirements.txt
│  └─ 발전부문_배출권_분석_마스터셋.csv   (모델 학습용 데이터)
└─ frontend/   ← React 화면 (단일 index.html). 백엔드 API를 호출
   └─ index.html
```

---

## 접속 주소

- **배포된 서비스(운영):** https://epfm.onrender.com/
  - 예측 API(JSON): https://epfm.onrender.com/api/predict
  - 상태 확인: https://epfm.onrender.com/api/health
  - (Render 무료 플랜은 유휴 상태에서 첫 접속 시 서버가 깨어나며 수십 초 걸릴 수 있습니다.)

## 로컬 실행 방법 (서버 하나만 켜면 끝)

백엔드 서버가 API와 프론트엔드 화면을 함께 제공합니다.

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

브라우저에서 **http://localhost:8000** 접속 → 대시보드가 열립니다.
(내 컴퓨터에서 실행할 때의 주소이며, 운영 접속은 위 배포 주소를 사용하세요.)

- 대시보드 화면:      http://localhost:8000/
- 예측 API(JSON):     http://localhost:8000/api/predict
- 상태 확인:          http://localhost:8000/api/health

## 동작 구조

1. 브라우저가 백엔드에서 `frontend/index.html`을 받습니다.
2. 화면의 컨트롤(모드·엔진·슬라이더)을 바꿀 때마다 프론트엔드가
   `/api/predict?mode=...&model=...` 를 호출합니다.
3. 백엔드가 statsmodels OLS / scikit-learn RandomForest로 향후 1년 연쇄 릴레이
   예측을 계산해 JSON으로 응답하고, 프론트엔드가 표·지표·차트로 그립니다.
4. **백엔드에 연결하지 못하면** 프론트엔드는 대시보드 대신
   "서버에 연결할 수 없습니다" 화면과 "다시 시도" 버튼을 표시합니다.

## 8월 고도화 기능

왼쪽 **데이터 추가/제거**와 대시보드 아래 **8월 모델 검증** 영역에서 다음 기능을 사용할 수 있습니다.

- **원본 파일 자동 인식:** 파일 선택창 하나에서 배출권 가격·원단위·공급(할당량)·수요(배출량) 파일을 함께 고르면 백엔드가 파일명과 컬럼 구조로 종류를 판별합니다.
- **마스터셋 동적 갱신:** 자동 분류한 파일을 월별로 동기화하고 결측치와 수급지수를 계산한 뒤 OLS와 RandomForest를 즉시 재학습합니다. 완성된 표준 마스터셋 한 개를 올리는 방식도 지원합니다.
- **예측 오차 백테스트:** 사전 예측 파일과 KRX 실제가 파일을 년월 기준으로 1:1 매칭하여 MAE, RMSE, MAPE, 평균 편향을 계산합니다. 파일이 없어도 최근 12개월 워크포워드 검증을 실행할 수 있습니다.

업로드 데이터는 서버 프로세스 메모리에 반영됩니다. Render 재배포·재시작 시에는 저장소의 기본 CSV로 복원되므로, 생성한 표준 CSV를 내려받아 검증한 뒤 영구 반영하려면 `backend/발전부문_배출권_분석_마스터셋.csv`를 교체해 커밋하세요.

### 추가 API

- `POST /api/data/master` — 표준 마스터셋 업로드 및 재학습
- `POST /api/data/auto-ingest` — 복수 파일 자동 분류·표준화·재학습
- `POST /api/data/reset` — 기본 마스터셋 복원
- `GET /api/backtest/historical?months=12` — 시계열 워크포워드 검증
- `POST /api/backtest/compare` — 예측/실제 파일 비교
- `POST /api/data/build-master?apply=true` — 원본 파일에서 표준 마스터셋 생성

## 배포 시 참고

- 프론트엔드와 백엔드를 **다른 서버/도메인**에 올리는 경우, `frontend/index.html`
  상단의 `window.__API_BASE__` 값을 백엔드 API 주소로 바꾸고(CORS는 허용되어 있음),
  그 파일을 정적 호스팅하면 됩니다. 같은 서버로 배포하면(위 기본 방식) 수정 불필요.
- `frontend/index.html`을 그냥 더블클릭(file://)하면 API를 호출하지 못해
  오류 화면이 나옵니다 — 반드시 백엔드 서버를 통해 여세요.
- **현재 배포:** Render(https://epfm.onrender.com/)에 백엔드+프론트엔드를 한 서버로
  배포했습니다. Render 시작 명령 예시: `uvicorn app:app --host 0.0.0.0 --port $PORT`
  (Render가 주입하는 `$PORT` 환경변수를 사용). 프론트엔드는 같은 오리진의 `/api`를
  호출하므로 `window.__API_BASE__` 수정 없이 동작합니다.
