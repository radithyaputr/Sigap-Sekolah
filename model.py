"""
SIGAP Sekolah - Machine Learning Engine v4.0
Dual-mode: SIGAP (10 features) + UCI (36 features)
Model Comparison: Random Forest, XGBoost, SVM + Ensemble Voting
SHAP Explainable AI for every prediction.
"""

import os
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, accuracy_score, recall_score,
    precision_score, f1_score, confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# ── SIGAP Mode (10 features, Indonesian SMK context) ──
SIGAP_FEATURE_NAMES = [
    "persentase_kehadiran", "rata_rata_nilai", "tren_nilai",
    "jumlah_mapel_di_bawah_kkm", "status_kip", "jarak_rumah_km",
    "jumlah_pelanggaran", "pekerjaan_ortu", "pendidikan_ortu", "jumlah_saudara",
]

SIGAP_FEATURE_LABELS = {
    "persentase_kehadiran": "Kehadiran (%)",
    "rata_rata_nilai": "Rata-rata Nilai",
    "tren_nilai": "Tren Nilai (3 bln)",
    "jumlah_mapel_di_bawah_kkm": "Mapel di Bawah KKM",
    "status_kip": "Penerima KIP/PIP",
    "jarak_rumah_km": "Jarak ke Sekolah (km)",
    "jumlah_pelanggaran": "Jumlah Pelanggaran",
    "pekerjaan_ortu": "Pekerjaan Orang Tua",
    "pendidikan_ortu": "Pendidikan Orang Tua",
    "jumlah_saudara": "Jumlah Saudara",
}

# ── UCI Mode (36 features, real dataset) ──
UCI_FEATURE_NAMES = [
    "Marital_status", "Application_mode", "Application_order", "Course",
    "Daytime_evening_attendance", "Previous_qualification", "Previous_qualification_grade",
    "Nacionality", "Mothers_qualification", "Fathers_qualification",
    "Mothers_occupation", "Fathers_occupation", "Admission_grade",
    "Displaced", "Educational_special_needs", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder",
    "Age_at_enrollment", "International",
    "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate", "Inflation_rate", "GDP",
]

UCI_FEATURE_LABELS = {
    "Marital_status": "Status Nikah",
    "Application_mode": "Mode Pendaftaran",
    "Application_order": "Urutan Pilihan",
    "Course": "Program Studi",
    "Daytime_evening_attendance": "Waktu Kuliah",
    "Previous_qualification": "Kualifikasi Sebelumnya",
    "Previous_qualification_grade": "Nilai Kualifikasi Sebelumnya",
    "Nacionality": "Kewarganegaraan",
    "Mothers_qualification": "Pendidikan Ibu",
    "Fathers_qualification": "Pendidikan Ayah",
    "Mothers_occupation": "Pekerjaan Ibu",
    "Fathers_occupation": "Pekerjaan Ayah",
    "Admission_grade": "Nilai Masuk",
    "Displaced": "Merantau",
    "Educational_special_needs": "Kebutuhan Khusus",
    "Debtor": "Memiliki Utang",
    "Tuition_fees_up_to_date": "SPP Terbayar",
    "Gender": "Jenis Kelamin",
    "Scholarship_holder": "Penerima Beasiswa",
    "Age_at_enrollment": "Usia Saat Daftar",
    "International": "Internasional",
    "Curricular_units_1st_sem_credited": "SKS Diakui (Sem 1)",
    "Curricular_units_1st_sem_enrolled": "SKS Diambil (Sem 1)",
    "Curricular_units_1st_sem_evaluations": "Evaluasi (Sem 1)",
    "Curricular_units_1st_sem_approved": "Lulus (Sem 1)",
    "Curricular_units_1st_sem_grade": "Rata-rata Nilai (Sem 1)",
    "Curricular_units_1st_sem_without_evaluations": "Tanpa Evaluasi (Sem 1)",
    "Curricular_units_2nd_sem_credited": "SKS Diakui (Sem 2)",
    "Curricular_units_2nd_sem_enrolled": "SKS Diambil (Sem 2)",
    "Curricular_units_2nd_sem_evaluations": "Evaluasi (Sem 2)",
    "Curricular_units_2nd_sem_approved": "Lulus (Sem 2)",
    "Curricular_units_2nd_sem_grade": "Rata-rata Nilai (Sem 2)",
    "Curricular_units_2nd_sem_without_evaluations": "Tanpa Evaluasi (Sem 2)",
    "Unemployment_rate": "Tingkat Pengangguran",
    "Inflation_rate": "Tingkat Inflasi",
    "GDP": "GDP",
}

# Backward compatibility aliases
FEATURE_NAMES = SIGAP_FEATURE_NAMES
FEATURE_LABELS = SIGAP_FEATURE_LABELS

RISK_LABELS = {0: "Aman", 1: "Berisiko Tinggi"}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def detect_mode(df: pd.DataFrame) -> str:
    """Detect whether data is SIGAP or UCI format."""
    if "Curricular_units_1st_sem_approved" in df.columns:
        return "uci"
    if "persentase_kehadiran" in df.columns:
        return "sigap"
    return "unknown"


def get_feature_names(mode: str = "sigap") -> list:
    if mode == "uci":
        return UCI_FEATURE_NAMES
    return SIGAP_FEATURE_NAMES


def get_feature_labels(mode: str = "sigap") -> dict:
    if mode == "uci":
        return UCI_FEATURE_LABELS
    return SIGAP_FEATURE_LABELS


def load_data(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "data_siswa.csv")
    if not os.path.exists(path):
        path_uci = os.path.join(DATA_DIR, "dataset.csv")
        if os.path.exists(path_uci):
            return load_uci_data(path_uci)
        raise FileNotFoundError(f"No dataset found in {DATA_DIR}")
    df = pd.read_csv(path)
    mode = detect_mode(df)
    if mode == "uci":
        return prepare_uci_dataframe(df)
    return df


def load_uci_data(path: str) -> pd.DataFrame:
    """Load and prepare UCI dataset."""
    df = pd.read_csv(path)
    return prepare_uci_dataframe(df)


def prepare_uci_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare UCI dataframe with Indonesian-style naming for compatibility."""
    target_col = "Target"
    if target_col in df.columns:
        target_map = {"Dropout": 1, "Enrolled": 0, "Graduate": 0}
        if df[target_col].dtype == object:
            df["putus_sekolah"] = df[target_col].map(target_map).fillna(0).astype(int)
        else:
            df["putus_sekolah"] = (df[target_col] == 1).astype(int)

    if "Age_at_enrollment" in df.columns:
        df["id_siswa"] = [f"UCI{i+1:04d}" for i in range(len(df))]
        df["nama"] = [f"Mahasiswa {i+1}" for i in range(len(df))]
        df["kelas"] = "N/A"

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")
        elif df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def prepare_features(df):
    mode = detect_mode(df)
    feature_names = get_feature_names(mode)

    available_features = [f for f in feature_names if f in df.columns]
    X = df[available_features].copy()

    y = None
    if "putus_sekolah" in df.columns:
        y = df["putus_sekolah"].copy()

    return X, y, mode


def train_all_models(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=random_state, eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)

    svm_base = SVC(
        kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
        random_state=random_state,
    )
    svm = CalibratedClassifierCV(svm_base, ensemble=False, cv=3)
    svm.fit(X_train_scaled, y_train)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("svm", svm)],
        voting="soft", weights=[2, 2, 1],
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)

    models = {"random_forest": rf, "xgboost": xgb, "svm": svm, "ensemble": ensemble}

    metrics_all = {}
    for name, m in models.items():
        if name == "svm":
            yp = m.predict(X_test_scaled)
        else:
            yp = m.predict(X_test)

        acc = round(accuracy_score(y_test, yp), 4)
        rec = round(recall_score(y_test, yp), 4)
        prec = round(precision_score(y_test, yp), 4)
        f1 = round(f1_score(y_test, yp), 4)

        try:
            if name == "svm":
                auc = round(roc_auc_score(y_test, m.predict_proba(X_test_scaled)[:, 1]), 4)
            else:
                auc = round(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]), 4)
        except Exception:
            auc = 0.0

        metrics_all[name] = {
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "auc_roc": auc,
            "confusion_matrix": confusion_matrix(y_test, yp).tolist(),
        }

    feature_names = list(X.columns)
    return models, metrics_all, scaler, X_test, y_test, feature_names


def train_model(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics, X_test, y_test


def save_model(model, path=None):
    if path is None:
        path = os.path.join(MODEL_DIR, "sigap_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def save_all_models(models, scaler, metrics, feature_names=None, path=None):
    if path is None:
        path = os.path.join(MODEL_DIR, "sigap_models.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {"models": models, "scaler": scaler, "metrics": metrics}
    if feature_names:
        bundle["feature_names"] = feature_names
    joblib.dump(bundle, path)
    return path


def load_model(path=None):
    if path is None:
        path = os.path.join(MODEL_DIR, "sigap_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_all_models(path=None):
    if path is None:
        path = os.path.join(MODEL_DIR, "sigap_models.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_risk(model, student_data, model_name="ensemble", models_bundle=None):
    if isinstance(student_data, dict):
        df = pd.DataFrame([student_data])
    else:
        df = student_data.copy()

    feature_names = models_bundle.get("feature_names", SIGAP_FEATURE_NAMES) if models_bundle else SIGAP_FEATURE_NAMES
    available = [f for f in feature_names if f in df.columns]
    X = df[available]

    if models_bundle and model_name in models_bundle["models"]:
        m = models_bundle["models"][model_name]
        scaler = models_bundle.get("scaler")
        if model_name == "svm" and scaler:
            X_input = scaler.transform(X)
        else:
            X_input = X
        probabilities = m.predict_proba(X_input)[:, 1]
        predictions = m.predict(X_input)
    else:
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)

    results = []
    for i in range(len(df)):
        results.append({
            "risk_score": round(float(probabilities[i]) * 100, 1),
            "risk_label": RISK_LABELS[predictions[i]],
            "is_at_risk": bool(predictions[i]),
        })
    return results


def explain_prediction(model, X, student_index=0):
    feature_names = list(X.columns)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        values = shap_values[1][student_index]
    else:
        values = shap_values[student_index]
    if hasattr(values, 'ndim') and values.ndim > 1:
        values = values.flatten()

    all_labels = {**SIGAP_FEATURE_LABELS, **UCI_FEATURE_LABELS}
    explanation = {}
    for i, feat in enumerate(feature_names):
        if i < len(values):
            val = values[i]
            if hasattr(val, 'item'):
                val = val.item()
            label = all_labels.get(feat, feat)
            explanation[label] = round(float(val), 4)
    return explanation


def get_top_factors(explanation, n=3):
    sorted_factors = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
    top = []
    for feat, shap_val in sorted_factors[:n]:
        direction = "meningkatkan" if shap_val > 0 else "menurunkan"
        top.append({
            "feature": feat, "impact": round(abs(shap_val) * 100, 1),
            "direction": direction,
        })
    return top


def generate_recommendations(top_factors, student_data):
    recommendations = []
    recommendations.append("Lakukan pemantauan intensif terhadap siswa ini selama 2 minggu ke depan.")

    kehadiran = student_data.get("persentase_kehadiran") or student_data.get("Curricular_units_1st_sem_approved", 100)
    if isinstance(kehadiran, (int, float)) and kehadiran < 75:
        recommendations.append("Hubungi orang tua/wali untuk memahami penyebab ketidakhadiran.")

    nilai = student_data.get("rata_rata_nilai") or student_data.get("Curricular_units_1st_sem_grade", 100)
    if isinstance(nilai, (int, float)) and nilai < 65:
        recommendations.append("Sediakan bimbingan belajar tambahan atau les privat.")

    bawah = student_data.get("jumlah_mapel_di_bawah_kkm", 0)
    if isinstance(bawah, (int, float)) and bawah > 3:
        recommendations.append("Identifikasi mata pelajaran yang paling lemah dan prioritaskan remedial.")

    kip = student_data.get("status_kip") or student_data.get("Scholarship_holder", 0)
    if kip == 1:
        recommendations.append("Pastikan bantuan KIP/PIP atau beasiswa telah tepat sasaran.")

    pelanggaran = student_data.get("jumlah_pelanggaran") or student_data.get("Curricular_units_1st_sem_without_evaluations", 0)
    if isinstance(pelanggaran, (int, float)) and pelanggaran > 5:
        recommendations.append("Lakukan pendekatan konseling untuk memahami penyebab perilaku.")

    recommendations.append("Catat setiap tindak lanjut di sistem untuk evaluasi berkala.")
    recommendations.append("Gunakan fitur What-If Simulator untuk mensimulasi dampak intervensi.")
    return recommendations


def what_if_simulation(student_data, models_bundle, changes):
    original_data = student_data.copy()
    modified_data = student_data.copy()
    for key, value in changes.items():
        modified_data[key] = value

    original_pred = predict_risk(None, pd.DataFrame([original_data]),
                                  model_name="ensemble", models_bundle=models_bundle)[0]
    simulated_pred = predict_risk(None, pd.DataFrame([modified_data]),
                                   model_name="ensemble", models_bundle=models_bundle)[0]

    contributions = {}
    for key in changes:
        single_change = original_data.copy()
        single_change[key] = changes[key]
        single_pred = predict_risk(None, pd.DataFrame([single_change]),
                                    model_name="ensemble", models_bundle=models_bundle)[0]
        contributions[key] = round(original_pred["risk_score"] - single_pred["risk_score"], 1)

    return {
        "original_score": original_pred["risk_score"],
        "simulated_score": simulated_pred["risk_score"],
        "reduction": round(original_pred["risk_score"] - simulated_pred["risk_score"], 1),
        "original_category": original_pred["risk_label"],
        "new_category": simulated_pred["risk_label"],
        "contributions": contributions,
    }


def full_pipeline(data_path=None, model_path=None):
    print("=" * 60)
    print("SIGAP Sekolah - Model Training Pipeline v4.0")
    print("=" * 60)
    print("\n[1/5] Loading data...")
    df = load_data(data_path)
    mode = detect_mode(df)
    print(f"  Mode: {mode.upper()}")
    print(f"  Loaded {len(df)} records")
    print("\n[2/5] Preparing features...")
    X, y, mode = prepare_features(df)
    print(f"  Features: {len(X.columns)}")
    print("\n[3/5] Training all models...")
    models, metrics, scaler, X_test, y_test, feature_names = train_all_models(X, y)
    for name, m in metrics.items():
        print(f"  {name}: Acc={m['accuracy']:.1%} Rec={m['recall']:.1%} AUC={m.get('auc_roc', 0):.3f}")
    print("\n[4/5] Saving models...")
    save_all_models(models, scaler, metrics, feature_names, model_path)
    print("\n[5/5] Generating sample explanation...")
    rf_model = models["random_forest"]
    explanation = explain_prediction(rf_model, X_test, 0)
    top = get_top_factors(explanation)
    for f in top:
        print(f"  - {f['feature']}: {f['impact']}% {f['direction']} risiko")
    print("\nTraining complete!")
    return models, metrics


if __name__ == "__main__":
    full_pipeline()
