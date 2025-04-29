from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from catboost import Pool

app = FastAPI()
model_cache = {}

class Query(BaseModel):
    station_id: int
    time_slot: int
    updnLine: str
    weekday_type: str
    line: int

line_car_count = {
    2: 10,
    3: 10,
    4: 10,
    5: 8,
    6: 8,
    7: 8,
    8: 6
}

feature_order_map = {
    2: [f'congestionCar_{i}_fft' for i in range(1, 11)] + [
        '역번호', '시간대', 'updnLine', '요일_카테고리', 'is_weekend', 'time_segment', 'hour'
    ],
    3: [f'congestionCar_{i}_fft' for i in range(1, 11)] + [
        '역번호', '시간대', 'updnLine', '요일', 'is_weekend', 'time_segment', 'hour'
    ],
    4: [f'congestionCar_{i}_fft' for i in range(1, 11)] + [
        '역번호', '시간대', 'updnLine', '요일', 'is_weekend', 'time_segment', 'hour'
    ],
    5: [f'congestionCar_{i}_fft' for i in range(1, 9)] + [
        '역번호', '시간대', 'updnLine', '요일', 'is_weekend', 'time_segment', 'hour'
    ],
    6: [f'congestionCar_{i}_fft' for i in range(1, 9)] + [
        '역번호', '시간대', 'updnLine', '요일', 'is_weekend', 'time_segment', 'hour'
    ],
    7: [f'congestionCar_{i}_fft' for i in range(1, 9)] + [
        '역번호', '시간대', 'updnLine', '요일', 'is_weekend', 'time_segment', 'hour'
    ],
    8: [f'congestionCar_{i}_fft' for i in range(1, 7)] + [
        '역번호', '시간대', 'updnLine', '요일', 'is_weekend', 'time_segment', 'hour'
    ]
}

cat_features_map = {
    2: ['updnLine', '요일_카테고리', 'time_segment'],
    3: ['updnLine', '요일', 'time_segment'],
    4: ['updnLine', '요일', 'time_segment'],
    5: ['updnLine', '요일', 'time_segment'],
    6: ['updnLine', '요일', 'time_segment'],
    7: ['updnLine', '요일', 'time_segment'],
    8: ['updnLine', '요일', 'time_segment']
}

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
    is_weekend = 1 if weekday_type in ['토요일', '일요일'] else 0
    return segment, is_weekend, hour

def generate_features_dict(station_id, time_slot, updnLine, weekday_type, line, car_no):
    segment, is_weekend, hour = get_time_segment_and_flags(time_slot, weekday_type)
    base = {
        '역번호': station_id,
        '시간대': time_slot,
        'updnLine': str(updnLine),
        'is_weekend': is_weekend,
        'hour': hour
    }
    if line == 2:
        base['요일_카테고리'] = str(weekday_type)
    else:
        base['요일'] = str(weekday_type)
    base['time_segment'] = str(segment)

    for i in range(1, line_car_count[line] + 1):
        base[f'congestionCar_{i}_fft'] = 0.0

    if line == 2 and car_no == 10:
        base['congestionCar_10_special'] = hour * int(updnLine == '하행')

    return base

@app.post("/predict_all_cars")
def predict_all_cars(query: Query):
    line = query.line
    model_path = f"catboost_models_{line}.pkl"

    if line not in model_cache:
        if not os.path.exists(model_path):
            return {"error": f"모델 파일 {model_path} 이 존재하지 않아요."}
        model_cache[line] = joblib.load(model_path)

    models = model_cache[line]
    car_count = line_car_count.get(line, 6)
    features = feature_order_map[line]
    cat_features = cat_features_map[line]

    predictions = {}
    minute = query.time_slot % 100
    lower_min = 0 if minute < 30 else 30
    upper_min = 30 if lower_min == 0 else 60
    lower_time = (query.time_slot // 100) * 100 + lower_min
    upper_time = lower_time + 30 if upper_min == 30 else (query.time_slot // 100 + 1) * 100
    ratio = (query.time_slot - lower_time) / 30

    for car_no in range(1, car_count + 1):
        model_key = f"congestionCar_{car_no}"
        if model_key not in models:
            predictions[f"car_{car_no}"] = "모델 없음"
            continue

        f1_dict = generate_features_dict(query.station_id, lower_time, query.updnLine, query.weekday_type, line, car_no)
        f2_dict = generate_features_dict(query.station_id, upper_time, query.updnLine, query.weekday_type, line, car_no)

        f1 = pd.DataFrame([f1_dict])
        f2 = pd.DataFrame([f2_dict])

        if line == 2 and car_no == 10:
            f1 = f1[features + ['congestionCar_10_special']]
            f2 = f2[features + ['congestionCar_10_special']]
        else:
            f1 = f1[features]
            f2 = f2[features]

        for col in cat_features:
            if col in f1.columns:
                f1[col] = f1[col].astype('category')
                f2[col] = f2[col].astype('category')

        f1_pool = Pool(f1, cat_features=cat_features)
        f2_pool = Pool(f2, cat_features=cat_features)

        pred1 = models[model_key].predict(f1_pool)[0]
        pred2 = models[model_key].predict(f2_pool)[0]
        interpolated = pred1 * (1 - ratio) + pred2 * ratio
        predictions[f"car_{car_no}"] = np.expm1(interpolated)

    return {
        "station": query.station_id,
        "time_slot": query.time_slot,
        "updnLine": query.updnLine,
        "weekday_type": query.weekday_type,
        "line": query.line,
        "predictions": predictions
    }