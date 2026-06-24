"""
SIGAP Sekolah - FastAPI Backend API
REST API untuk integrasi dengan sistem lain.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="SIGAP AI API", version="3.0.0",
              description="Sistem Identifikasi Gejala Anak Berisiko Putus Sekolah")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
models_bundle = None
rf_model = None
df_data = None


class StudentInput(BaseModel):
    persentase_kehadiran: float = Field(..., ge=0, le=100)
    rata_rata_nilai: float = Field(..., ge=0, le=100)
    tren_nilai: float = Field(..., ge=-10, le=10)
    jumlah_mapel_di_bawah_kkm: int = Field(..., ge=0, le=12)
    status_kip: int = Field(..., ge=0, le=1)
    jarak_rumah_km: float = Field(..., ge=0.5, le=25)
    jumlah_pelanggaran: int = Field(..., ge=0, le=20)
    pekerjaan_ortu: int = Field(..., ge=0, le=3)
    pendidikan_ortu: int = Field(..., ge=0, le=4)
    jumlah_saudara: int = Field(..., ge=0, le=8)


class PredictionResponse(BaseModel):
    risk_score: float
    risk_label: str
    is_at_risk: bool


class WhatIfInput(BaseModel):
    student_data: Dict[str, Any]
    changes: Dict[str, Any]


@app.on_event("startup")
async def load_models():
    global models_bundle, rf_model, df_data
    try:
        path = os.path.join(MODEL_DIR, "sigap_models.pkl")
        if os.path.exists(path):
            models_bundle = joblib.load(path)
            rf_model = models_bundle["models"]["random_forest"]
        data_path = os.path.join(DATA_DIR, "data_siswa.csv")
        if os.path.exists(data_path):
            df_data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Warning: Could not load models: {e}")


@app.get("/")
async def root():
    return {"name": "SIGAP AI API", "version": "3.0.0", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "models_loaded": rf_model is not None,
            "data_loaded": df_data is not None, "timestamp": datetime.now().isoformat()}


@app.post("/api/predict/single", response_model=PredictionResponse)
async def predict_single(student: StudentInput):
    if rf_model is None or models_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    from model import predict_risk as pr, FEATURE_NAMES
    df = pd.DataFrame([student.dict()])
    result = pr(rf_model, df, model_name="ensemble", models_bundle=models_bundle)[0]
    return PredictionResponse(**result)


@app.post("/api/predict/batch")
async def predict_batch(students: List[StudentInput]):
    if rf_model is None or models_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    from model import predict_risk as pr
    df = pd.DataFrame([s.dict() for s in students])
    results = pr(rf_model, df, model_name="ensemble", models_bundle=models_bundle)
    return {"predictions": results, "total": len(results),
            "at_risk": sum(1 for r in results if r["is_at_risk"])}


@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    if rf_model is None or models_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    from model import predict_risk as pr, FEATURE_NAMES
    contents = await file.read()
    from io import StringIO
    df_up = pd.read_csv(StringIO(contents.decode("utf-8")))
    missing = [f for f in FEATURE_NAMES if f not in df_up.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
    results = pr(rf_model, df_up, model_name="ensemble", models_bundle=models_bundle)
    df_up["skor_risiko"] = [p["risk_score"] for p in results]
    df_up["label_risiko"] = [p["risk_label"] for p in results]
    return {"predictions": results, "total": len(results), "at_risk": sum(1 for r in results if r["is_at_risk"]),
            "data": df_up.to_dict("records")}


@app.get("/api/students")
async def get_students(limit: int = 50, offset: int = 0, risk_only: bool = False):
    if df_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    from model import predict_risk as pr
    preds = pr(rf_model, df_data, model_name="ensemble", models_bundle=models_bundle)
    df_result = df_data.copy()
    df_result["skor_risiko"] = [p["risk_score"] for p in preds]
    df_result["label_risiko"] = [p["risk_label"] for p in preds]
    if risk_only:
        df_result = df_result[df_result["label_risiko"] == "Berisiko Tinggi"]
    df_result = df_result.sort_values("skor_risiko", ascending=False)
    return {"students": df_result.iloc[offset:offset+limit].to_dict("records"),
            "total": len(df_result)}


@app.get("/api/students/{student_id}")
async def get_student(student_id: str):
    if df_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    if "id_siswa" not in df_data.columns:
        raise HTTPException(status_code=400, detail="No student ID column")
    row = df_data[df_data["id_siswa"] == student_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    from model import predict_risk as pr, explain_prediction as ep, get_top_factors as gtf, FEATURE_NAMES
    pred = pr(rf_model, row, model_name="ensemble", models_bundle=models_bundle)[0]
    explanation = ep(rf_model, row[FEATURE_NAMES], 0)
    top_factors = gtf(explanation, n=5)
    return {"student_id": student_id, "data": row.iloc[0].to_dict(),
            "prediction": pred, "explanation": explanation, "top_factors": top_factors}


@app.post("/api/simulator/run")
async def run_simulator(input_data: WhatIfInput):
    if models_bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    from model import what_if_simulation as wis
    result = wis(input_data.student_data, models_bundle, input_data.changes)
    return result


@app.get("/api/analytics/dashboard")
async def get_dashboard():
    if df_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    from model import predict_risk as pr
    from school_intelligence import SchoolRiskIntelligence
    preds = pr(rf_model, df_data, model_name="ensemble", models_bundle=models_bundle)
    from model import FEATURE_NAMES
    intel = SchoolRiskIntelligence(df_data, preds, FEATURE_NAMES)
    return intel.get_dashboard_data()


@app.get("/api/analytics/factors")
async def get_factors():
    if df_data is None or rf_model is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    from model import FEATURE_NAMES, FEATURE_LABELS
    importance = rf_model.feature_importances_
    return [{"feature": FEATURE_LABELS[f], "importance": round(float(importance[i])*100, 2)}
            for i, f in enumerate(FEATURE_NAMES)]


@app.get("/api/models/metrics")
async def get_model_metrics():
    if models_bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return models_bundle.get("metrics", {})
