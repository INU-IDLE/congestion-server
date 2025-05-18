import pandas as pd
import numpy as np
import joblib
from catboost import Pool
from fastapi import FastAPI, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# ✅ CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 호선별 차량 수
line_car_count = {
    2: 10, 3: 10, 4: 10,
    5: 8, 6: 8, 7: 8,
    8: 6
}

# 피처 순서 및 범주형 피처 목록
feature_order_map = {
    line: [f'congestionCar_{i}_fft' for i in range(1, count + 1)] +
          ['역번호', '시간대', 'updnLine', '요일_카테고리', 'is_weekend', 'time_segment', 'hour']
    for line, count in line_car_count.items()
}
cat_features_map = {
    line: ['역번호', 'updnLine', '요일_카테고리', 'time_segment']
    for line in line_car_count
}

# 월별 가중치
month_weight_map = {
    1: 1.00,
    2: 0.965,
    3: 1.035,
    4: 1.141,
    5: 1.176,
    6: 1.106,
    7: 1.059,
    8: 1.012,
    9: 1.165,
    10: 1.188,
    11: 1.235,
    12: 1.188
}

# 모델 & 데이터 로딩
general_models = {}
congested_models = {}
fft_data_by_line = {}

for line in line_car_count:
    gen_path = f"general_models/general_models_{line}.pkl"
    con_path = f"congested_models/congested_models_{line}.pkl"
    fft_path = f"fourier_transformed_data/fourier_transformed_data_{line}.csv"

    if os.path.exists(gen_path):
        general_models[line] = joblib.load(gen_path)
        print(f"[✅ 일반 모델 로딩 완료] line {line}")
    if os.path.exists(con_path):
        congested_models[line] = joblib.load(con_path)
        print(f"[✅ 고혼잡 모델 로딩 완료] line {line}")
    if os.path.exists(fft_path):
        fft_data_by_line[line] = pd.read_csv(fft_path)
        print(f"[✅ FFT 데이터 로딩 완료] line {line}")

# 시간대 파생 특성
def get_time_segment_and_flags(time_slot: int, weekday_type: str):
    hour = time_slot // 100
    if time_slot < 500: segment = '새벽'
    elif time_slot < 800: segment = '아침'
    elif time_slot < 1200: segment = '점심'
    elif time_slot < 1600: segment = '오후'
    elif time_slot < 2000: segment = '저녁'
    else: segment = '심야'
    is_weekend = int(weekday_type in ['토요일', '일요일'])
    return segment, is_weekend, hour

# 입력 feature 생성
def generate_features_dict(station_id, time_slot, updnLine, weekday_type, line):
    segment, is_weekend, hour = get_time_segment_and_flags(time_slot, weekday_type)
    updn_int = int(updnLine)

    base = {
        '역번호': str(station_id),
        '시간대': time_slot,
        'updnLine': updn_int,
        'is_weekend': is_weekend,
        'hour': hour,
        'time_segment': segment,
        '요일_카테고리': weekday_type
    }

    fft_df = fft_data_by_line.get(line)
    if fft_df is not None:
        group = fft_df[
            (fft_df['역번호'].astype(str) == str(station_id)) &
            (fft_df['updnLine'] == updn_int) &
            (fft_df['요일_카테고리'] == weekday_type)
        ].copy()
        group['시간대'] = group['시간대'].astype(int)

        if time_slot in group['시간대'].values:
            row = group[group['시간대'] == time_slot].iloc[0]
            for i in range(1, line_car_count[line] + 1):
                base[f'congestionCar_{i}_fft'] = row[f'congestionCar_{i}_fft']
        else:
            before = group[group['시간대'] < time_slot].tail(1)
            after = group[group['시간대'] > time_slot].head(1)
            if not before.empty and not after.empty:
                t1, t2 = before['시간대'].values[0], after['시간대'].values[0]
                r = (time_slot - t1) / (t2 - t1)
                for i in range(1, line_car_count[line] + 1):
                    col = f'congestionCar_{i}_fft'
                    v1, v2 = before[col].values[0], after[col].values[0]
                    base[col] = v1 * (1 - r) + v2 * r
            else:
                for i in range(1, line_car_count[line] + 1):
                    base[f'congestionCar_{i}_fft'] = 0.0
    else:
        for i in range(1, line_car_count[line] + 1):
            base[f'congestionCar_{i}_fft'] = 0.0

    return base

# API 엔드포인트
@app.get("/api/v1/congestion/real-time/car/{station_code}")
def predict_all_cars(
    station_code: int = Path(...),
    time_slot: int = Query(...),
    updnLine: str = Query(...),
    weekday_type: str = Query(...),
    line: int = Query(...),
    month: int = Query(...)
):
    if line not in general_models or line not in congested_models:
        return {"error": f"모델이 없습니다: line {line}"}

    gen_models = general_models[line]
    con_models = congested_models[line]
    car_count = line_car_count[line]
    features = feature_order_map[line]
    cat_features = cat_features_map[line]
    weight = month_weight_map.get(month, 1.0)

    predictions = {}
    f_dict = generate_features_dict(station_code, time_slot, updnLine, weekday_type, line)
    f = pd.DataFrame([f_dict])[features]

    for col in cat_features:
        f[col] = f[col].astype('category')

    # ✅ 평일 + 출퇴근 시간 조건일 때만 고혼잡 모델 사용
    is_peak = (
        weekday_type == "평일" and
        ((700 <= time_slot <= 900) or (1730 <= time_slot <= 1930))
    )
    model_group = con_models if is_peak else gen_models

    for car_no in range(1, car_count + 1):
        model_key = f"congestionCar_{car_no}"
        if model_key not in model_group:
            predictions[f"car_{car_no}"] = None
            continue

        pred = model_group[model_key].predict(Pool(f, cat_features=cat_features))[0]
        predictions[f"car_{car_no}"] = round(pred * weight, 2)

    return {
        "station": station_code,
        "time_slot": time_slot,
        "updnLine": updnLine,
        "weekday_type": weekday_type,
        "line": line,
        "month": month,
        "predictions": predictions
    }
