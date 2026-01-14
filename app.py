# app.py
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from feature_builder import FeatureBuilderNoDelayRate
import __main__
__main__.FeatureBuilderNoDelayRate = FeatureBuilderNoDelayRate  # 兼容 joblib 反序列化

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== 兼容两种放法：根目录 或 results 文件夹 ======
CANDIDATE_MODEL_PATHS = [
    os.path.join(BASE_DIR, "results_cls_no_delayrate_fast_simple", "best_model_classifier.joblib"),
    os.path.join(BASE_DIR, "best_model_classifier.joblib"),
]
CANDIDATE_ROUTE_JSON_PATHS = [
    os.path.join(BASE_DIR, "results_cls_no_delayrate_fast_simple", "route_constraints.json"),
    os.path.join(BASE_DIR, "route_constraints.json"),
]
INDEX_HTML = os.path.join(BASE_DIR, "index.html")  # 可选：如果你想让 Render 同时托管前端

def pick_first_exist(paths, name):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} 不存在。尝试过：\n" + "\n".join(paths))

MODEL_PATH = pick_first_exist(CANDIDATE_MODEL_PATHS, "模型文件 best_model_classifier.joblib")
ROUTE_JSON = pick_first_exist(CANDIDATE_ROUTE_JSON_PATHS, "route_constraints.json")

# ====== 训练里出现过的列名（保持不变）======
DEP_BIN_COL = "起飞时间离散化"
ARR_BIN_COL = "到达时间离散化"
ARR_TIME_COL = "到达时间"

app = FastAPI(title="Flight Delay Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 加载模型 & 约束 ======
model = joblib.load(MODEL_PATH)
with open(ROUTE_JSON, "r", encoding="utf-8") as f:
    route_constraints = json.load(f)

# ====== 从 route_constraints.json 推导全量 airlines/airports/routes（不需要 Excel） ======
def build_all_from_constraints(constraints: dict):
    airlines = sorted(list(constraints.keys()))
    airports = set()
    routes = set()

    for air, from_dict in constraints.items():
        for frm, tos in (from_dict or {}).items():
            airports.add(str(frm))
            for t in tos or []:
                airports.add(str(t))
                routes.add(f"{frm}-{t}")

    return {
        "airlines": airlines,
        "airports": sorted(list(airports)),
        "routes": sorted(list(routes))
    }

ALL_LISTS = build_all_from_constraints(route_constraints)

# ====== 折扣：更细（8档） ======
def discount_suggestion(p: float):
    p = float(p)
    bins = [
        (0.10, "VeryLow",   0),
        (0.20, "Low",      -5),
        (0.30, "Medium",  -10),
        (0.40, "MedHigh", -15),
        (0.50, "High",    -25),
        (0.60, "VeryHigh",-35),
        (0.70, "Severe",  -45),
        (1.01, "Extreme", -55),
    ]
    for th, label, disc in bins:
        if p < th:
            return label, disc
    return "Extreme", -55

# ====== 时间段编码（上午/下午/晚上） ======
def label_to_code(label: str) -> int:
    s = str(label).strip()
    if s == "上午": return 0
    if s == "下午": return 1
    if s == "晚上": return 2
    raise HTTPException(status_code=422, detail="DepBin/ArrBin 只能是：上午 / 下午 / 晚上")

def code_to_label(code: int) -> str:
    return {0:"上午", 1:"下午", 2:"晚上"}.get(int(code), "未知")

# 训练时 dep/arr bin 可能是数字(0/1/2)也可能是中文字符串
# 没有 Excel 时没法 100% 自动判断，所以：默认按 string 传（更常见）
DEP_BIN_MODE = "string"
ARR_BIN_MODE = "string"

def encode_bin_from_label(label: str, mode: str):
    code = label_to_code(label)
    if mode == "numeric":
        return int(code)
    if mode == "string":
        return code_to_label(code)
    return None

# ====== 输入 ======
class PredictIn(BaseModel):
    Airline: str
    AirportFrom: str
    AirportTo: str
    DayOfWeek: int
    Length: float
    DepBin: str   # 上午/下午/晚上
    ArrBin: str   # 上午/下午/晚上

@app.get("/")
def home():
    # 如果你把 index.html 放在仓库根目录，就可以直接用 Render 的一个 URL 同时跑前后端
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML)
    return {"status": "ok", "message": "API is running. Go to /docs for testing."}

@app.get("/options")
def options():
    return {
        "constraints": route_constraints,
        "all": ALL_LISTS,
        "timebin_mode": {
            "dep_col": DEP_BIN_COL,
            "dep_mode": DEP_BIN_MODE,
            "arr_col": ARR_BIN_COL,
            "arr_mode": ARR_BIN_MODE
        }
    }

@app.get("/debug/expected")
def debug_expected():
    cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    return {
        "expected_columns": cols,
        "has_feature_names_in": bool(cols),
        "model_path_used": MODEL_PATH,
        "route_json_used": ROUTE_JSON
    }

@app.post("/predict")
def predict(x: PredictIn):
    # 组装 row（基础列）
    row = {
        "Airline": str(x.Airline),
        "AirportFrom": str(x.AirportFrom),
        "AirportTo": str(x.AirportTo),
        "DayOfWeek": int(x.DayOfWeek),
        "Length": float(x.Length),
    }

    # 如果训练里有 ARR_TIME_COL 但你现在没用到，就给 NaN
    row[ARR_TIME_COL] = np.nan

    # 按训练时类型编码（这里默认 string）
    dep_bin_val = encode_bin_from_label(x.DepBin, DEP_BIN_MODE)
    arr_bin_val = encode_bin_from_label(x.ArrBin, ARR_BIN_MODE)
    row[DEP_BIN_COL] = dep_bin_val
    row[ARR_BIN_COL] = arr_bin_val

    df = pd.DataFrame([row])

    # 对齐到模型训练时输入列：缺的补 np.nan，多的删
    expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    if expected:
        for c in expected:
            if c not in df.columns:
                df[c] = np.nan
        df = df[expected]

    try:
        p = float(model.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {repr(e)}")

    level, disc = discount_suggestion(p)

    return {
        "delay_probability": p,
        "delay_probability_percent": round(p * 100, 2),
        "risk_level": level,
        "discount_percent": disc,
        "dep_bin_value_sent": dep_bin_val,
        "arr_bin_value_sent": arr_bin_val,
    }
