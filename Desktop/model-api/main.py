from fastapi import FastAPI, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
from catboost import Pool

app = FastAPI()
model_cache = {}
fft_data_by_line = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

line_car_count = {
    2: 10, 3: 10, 4: 10,
    5: 8, 6: 8, 7: 8,
    8: 6
}

feature_order_map = {
    line: [f'congestionCar_{i}_fft' for i in range(1, count + 1)] +
          ['역번호', '시간대', 'updnLine', '요일_카테고리', 'is_weekend', 'time_segment', 'hour']
    for line, count in line_car_count.items()
}

cat_features_map = {
    line: ['updnLine', '요일_카테고리', 'time_segment']
    for line in line_car_count
}

month_weight_map = {
    1: 0.85, 2: 0.82, 3: 0.88,
    4: 0.97, 5: 1.00, 6: 0.94,
    7: 0.90, 8: 0.86, 9: 0.99,
    10: 1.01, 11: 1.05, 12: 1.01
}

# Load models & FFT CSV
for line in line_car_count:
    model_path = f"catboost_models_{line}.pkl"
    fft_path = f"fourier_transformed_data/fourier_transformed_data_{line}.csv"
    if os.path.exists(model_path):
        model_cache[line] = joblib.load(model_path)
        print(f"[✅ 모델 로딩 완료] line {line}")
    if os.path.exists(fft_path):
        fft_data_by_line[line] = pd.read_csv(fft_path)
        print(f"[✅ FFT 데이터 로딩 완료] line {line}")

def get_time_segment_and_flags(time_slot: int, weekday_type: str):
    hour = time_slot // 100
    if time_slot < 500:
        segment = '새벽'
    elif time_slot < 800:
        segment = '아침'
    elif time_slot < 1200:
        segment = '점심'
    elif time_slot < 1600:
        segment = '오후'
    elif time_slot < 2000:
        segment = '저녁'
    else:
        segment = '심야'
    is_weekend = int(weekday_type in ['토요일', '일요일'])
    return segment, is_weekend, hour

def generate_features_dict(station_id, time_slot, updnLine, weekday_type, line):
    segment, is_weekend, hour = get_time_segment_and_flags(time_slot, weekday_type)
    updn_str = str(updnLine)

    base = {
        '역번호': station_id,
        '시간대': time_slot,
        'updnLine': updn_str,
        'is_weekend': is_weekend,
        'hour': hour,
        'time_segment': segment,
        '요일_카테고리': weekday_type
    }

    fft_df = fft_data_by_line.get(line)
    if fft_df is not None:
        try:
            updn_numeric = int(updnLine)
        except ValueError:
            updn_numeric = 1 if updnLine == '상행' else 0

        group = fft_df[
            (fft_df['역번호'] == station_id) &
            (fft_df['updnLine'] == updn_numeric) &
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
                t1 = before['시간대'].values[0]
                t2 = after['시간대'].values[0]
                r = (time_slot - t1) / (t2 - t1)
                print(f"[🔄 보간 적용] {t1}~{t2} → {time_slot} (비율: {r:.2f})")
                for i in range(1, line_car_count[line] + 1):
                    col = f'congestionCar_{i}_fft'
                    v1 = before[col].values[0]
                    v2 = after[col].values[0]
                    base[col] = v1 * (1 - r) + v2 * r
            else:
                print(f"[⚠️ 보간 불가] station={station_id}, time={time_slot}")
                for i in range(1, line_car_count[line] + 1):
                    base[f'congestionCar_{i}_fft'] = 0.0
    else:
        for i in range(1, line_car_count[line] + 1):
            base[f'congestionCar_{i}_fft'] = 0.0

    return base

@app.get("/api/v1/congestion/real-time/car/{station_code}")
def predict_all_cars(
    station_code: int = Path(...),
    time_slot: int = Query(...),
    updnLine: str = Query(...),
    weekday_type: str = Query(...),
    line: int = Query(...),
    month: int = Query(...)
):
    if line not in model_cache:
        return {"error": f"모델이 없습니다: line {line}"}

    models = model_cache[line]
    car_count = line_car_count[line]
    features = feature_order_map[line]
    cat_features = cat_features_map[line]
    weight = month_weight_map.get(month, 1.0)

    predictions = {}

    f_dict = generate_features_dict(station_code, time_slot, updnLine, weekday_type, line)
    f = pd.DataFrame([f_dict])[features]

    for col in cat_features:
        if col in f.columns:
            f[col] = f[col].astype('category')

    

    for car_no in range(1, car_count + 1):
        model_key = f"congestionCar_{car_no}"
        if model_key not in models:
            predictions[f"car_{car_no}"] = None
            continue

        print(f"✅ 모델 학습된 features: {models[model_key].feature_names_}")

        pred_log = models[model_key].predict(Pool(f, cat_features=cat_features))[0]
        predictions[f"car_{car_no}"] = round(np.expm1(pred_log) * weight, 2)

    return {
        "station": station_code,
        "time_slot": time_slot,
        "updnLine": updnLine,
        "weekday_type": weekday_type,
        "line": line,
        "month": month,
        "predictions": predictions
    }
