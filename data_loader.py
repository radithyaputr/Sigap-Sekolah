"""
SIGAP Sekolah - Real Dataset Loader v4.0
Loads UCI "Predict Students' Dropout and Academic Success" dataset.
Adapts features to Indonesian education context for LKS AI 2026.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List

# ── UCI Dataset Feature Names (original) ──
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

# ── Indonesian Feature Labels (for UI) ──
UCI_FEATURE_LABELS_ID = {
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
    "International": "Mahasiswa Internasional",
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

# ── SIGAP Feature Names (simplified for SMK context, 10 original) ──
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

# ── Feature categories for analysis ──
FEATURE_CATEGORIES = {
    "Demografi": ["Marital_status", "Gender", "Age_at_enrollment", "Nacionality", "International"],
    "Akademik": [
        "Previous_qualification", "Previous_qualification_grade", "Admission_grade",
        "Course", "Application_mode", "Application_order",
    ],
    "Keuangan": ["Debtor", "Tuition_fees_up_to_date", "Scholarship_holder"],
    "Kinerja Semester 1": [
        "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    ],
    "Kinerja Semester 2": [
        "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
    ],
    "Latar Belakang Keluarga": [
        "Mothers_qualification", "Fathers_qualification",
        "Mothers_occupation", "Fathers_occupation",
    ],
    "Status Sosial": ["Displaced", "Educational_special_needs"],
    "Ekonomi Makro": ["Unemployment_rate", "Inflation_rate", "GDP"],
}

# ── Marital Status mapping ──
MARITAL_STATUS_MAP = {
    1: "Belum Menikah", 2: "Menikah", 3: "Janda/Duda",
    4: "Cerai", 5: "Kohabitasi", 6: "Cerai Resmi",
}

# ── Course mapping (simplified) ──
COURSE_MAP = {
    33: "Teknologi Informasi", 171: "Animasi & Multimedia",
    8014: "Manajemen", 9003: "Keperawatan",
    9070: "Jurnalistik", 9085: "Manajemen Sosial",
    9119: "Teknologi Kesehatan", 9130: "Pendidikan Guru",
    9147: "Gizi", 9238: "Hukum",
    9500: "Kedokteran Gigi", 9556: "Farmasi",
    9670: "Teknik Elektro", 9773: "Desain Komunikasi",
    9851: "Akuntansi", 9991: "Hubungan Internasional",
}


def load_uci_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load UCI Predict Students Dropout dataset.
    If csv_path not provided, looks in data/ directory.
    """
    if csv_path is None:
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, "data", "dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Download from: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention\n"
            "Place the 'dataset.csv' file in the 'data/' directory."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded UCI dataset: {len(df)} records, {len(df.columns)} columns")
    return df


def prepare_uci_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare features from UCI dataset for model training.
    Returns (X, y) where y is the target variable.
    """
    target_col = "Target"
    if target_col not in df.columns:
        for col in df.columns:
            if "target" in col.lower() or "status" in col.lower():
                target_col = col
                break

    if target_col in df.columns:
        target_map = {"Dropout": 1, "Enrolled": 0, "Graduate": 0}
        if df[target_col].dtype == object:
            y = df[target_col].map(target_map).fillna(0).astype(int)
        else:
            y = (df[target_col] == 1).astype(int)
    else:
        y = None

    feature_cols = [c for c in UCI_FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna("Unknown")
        else:
            X[col] = X[col].fillna(X[col].median())

    return X, y


def adapt_to_sigap_features(df_uci: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt UCI features to SIGAP's 10-feature format for Indonesian SMK context.
    This maps university-level data to high school context.
    """
    adapted = pd.DataFrame()

    if "Curricular_units_1st_sem_approved" in df_uci.columns and "Curricular_units_1st_sem_enrolled" in df_uci.columns:
        enrolled_1 = df_uci["Curricular_units_1st_sem_enrolled"].replace(0, 1)
        approved_1 = df_uci["Curricular_units_1st_sem_approved"]
        adapted["persentase_kehadiran"] = np.clip((approved_1 / enrolled_1 * 100), 0, 100).round(1)

    if "Curricular_units_1st_sem_grade" in df_uci.columns:
        adapted["rata_rata_nilai"] = np.clip(df_uci["Curricular_units_1st_sem_grade"] * 10, 0, 100).round(1)

    if "Curricular_units_1st_sem_grade" in df_uci.columns and "Curricular_units_2nd_sem_grade" in df_uci.columns:
        sem1 = df_uci["Curricular_units_1st_sem_grade"]
        sem2 = df_uci["Curricular_units_2nd_sem_grade"]
        adapted["tren_nilai"] = ((sem2 - sem1) * 10).clip(-10, 10).round(1)

    if "Curricular_units_1st_sem_enrolled" in df_uci.columns and "Curricular_units_1st_sem_approved" in df_uci.columns:
        enrolled = df_uci["Curricular_units_1st_sem_enrolled"]
        approved = df_uci["Curricular_units_1st_sem_approved"]
        adapted["jumlah_mapel_di_bawah_kkm"] = np.maximum(0, enrolled - approved).clip(0, 12).astype(int)

    if "Scholarship_holder" in df_uci.columns:
        adapted["status_kip"] = df_uci["Scholarship_holder"].astype(int)

    adapted["jarak_rumah_km"] = np.random.exponential(3, len(df_uci)).clip(0.5, 25).round(1)

    if "Curricular_units_1st_sem_without_evaluations" in df_uci.columns:
        adapted["jumlah_pelanggaran"] = df_uci["Curricular_units_1st_sem_without_evaluations"].clip(0, 20).astype(int)

    if "Fathers_occupation" in df_uci.columns:
        occ = df_uci["Fathers_occupation"]
        adapted["pekerjaan_ortu"] = occ.clip(0, 3).astype(int)

    if "Mothers_qualification" in df_uci.columns:
        qual = df_uci["Mothers_qualification"]
        adapted["pendidikan_ortu"] = qual.clip(0, 4).astype(int)

    adapted["jumlah_saudara"] = np.random.poisson(2.5, len(df_uci)).clip(0, 8).astype(int)

    return adapted


def get_dataset_info(df: pd.DataFrame) -> Dict:
    """Get comprehensive dataset information for EDA."""
    info = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_features": len(df.select_dtypes(include=["object", "category"]).columns),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
    }

    if "Target" in df.columns:
        target_dist = df["Target"].value_counts()
        info["target_distribution"] = {
            "Dropout": int(target_dist.get("Dropout", 0)),
            "Enrolled": int(target_dist.get("Enrolled", 0)),
            "Graduate": int(target_dist.get("Graduate", 0)),
        }
        total = len(df)
        info["dropout_rate"] = round(info["target_distribution"]["Dropout"] / total * 100, 1)
        info["graduation_rate"] = round(info["target_distribution"]["Graduate"] / total * 100, 1)

    if "Age_at_enrollment" in df.columns:
        info["age_stats"] = {
            "mean": round(float(df["Age_at_enrollment"].mean()), 1),
            "min": int(df["Age_at_enrollment"].min()),
            "max": int(df["Age_at_enrollment"].max()),
        }

    if "Gender" in df.columns:
        gender_dist = df["Gender"].value_counts()
        info["gender_distribution"] = {
            "Male": int(gender_dist.get(1, 0)),
            "Female": int(gender_dist.get(0, 0)),
        }

    return info


def get_correlation_with_target(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Get correlation of each feature with the target variable."""
    if "Target" not in df.columns:
        return pd.DataFrame()

    target_map = {"Dropout": 1, "Enrolled": 0, "Graduate": 0}
    target = df["Target"].map(target_map).fillna(0)

    correlations = []
    for feat in feature_names:
        if feat in df.columns and df[feat].dtype in ["int64", "float64"]:
            corr = float(df[feat].corr(target))
            correlations.append({
                "Feature": feat,
                "Label": UCI_FEATURE_LABELS_ID.get(feat, feat),
                "Correlation": round(corr, 4),
                "AbsCorrelation": round(abs(corr), 4),
            })

    return pd.DataFrame(correlations).sort_values("AbsCorrelation", ascending=False)


if __name__ == "__main__":
    print("=== SIGAP Data Loader v4.0 ===")
    print("\nUCI Dataset Features:")
    for i, f in enumerate(UCI_FEATURE_NAMES, 1):
        label = UCI_FEATURE_LABELS_ID.get(f, f)
        print(f"  {i:2d}. {f} -> {label}")

    print(f"\nTotal features: {len(UCI_FEATURE_NAMES)}")
    print(f"Total labels: {len(UCI_FEATURE_LABELS_ID)}")
    print(f"\nCategories:")
    for cat, feats in FEATURE_CATEGORIES.items():
        print(f"  {cat}: {len(feats)} features")
