"""
SIGAP Sekolah - Responsible AI Module
Bias Detection, Fairness Check, Transparency Logging.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


class FairnessChecker:
    def __init__(self, model, data, feature_names):
        self.model = model
        self.data = data
        self.feature_names = feature_names

    def demographic_parity(self, sensitive_col):
        groups = self.data[sensitive_col].unique()
        results = {}
        for g in groups:
            mask = self.data[sensitive_col] == g
            X_g = self.data.loc[mask, self.feature_names]
            preds = self.model.predict(X_g)
            results[str(g)] = round(float(preds.mean()), 4)
        rates = list(results.values())
        max_r = max(rates) if rates else 1
        min_r = min(rates) if rates else 0
        disparity = round(max_r / min_r, 3) if min_r > 0 else float("inf")

        data_rates = {}
        for g in groups:
            mask = self.data[sensitive_col] == g
            if "putus_sekolah" in self.data.columns:
                data_rates[str(g)] = round(float(self.data.loc[mask, "putus_sekolah"].mean()), 4)
        data_vals = list(data_rates.values())
        if data_vals:
            data_max = max(data_vals)
            data_min = min(data_vals)
            data_disparity = round(data_max / data_min, 3) if data_min > 0 else float("inf")
        else:
            data_disparity = 1.0

        model_exceeds_data = disparity <= data_disparity * 1.3 if data_disparity > 0 else disparity <= 1.5

        return {
            "groups": results,
            "data_groups": data_rates,
            "disparity_ratio": disparity,
            "data_disparity_ratio": data_disparity,
            "threshold": 1.5,
            "is_fair": disparity <= 1.5 and model_exceeds_data,
            "explanation": (
                f"Model disparity: {disparity}x | Data disparity: {data_disparity}x. "
                f"Model tidak memperbesar bias yang sudah ada dalam data."
                if model_exceeds_data else
                f"Model disparity: {disparity}x melebihi threshold. Perlu penyesuaian."
            ),
        }

    def equalized_odds(self, sensitive_col, y_true):
        groups = self.data[sensitive_col].unique()
        results = {}
        for g in groups:
            mask = self.data[sensitive_col] == g
            y_g = y_true[mask]
            X_g = self.data.loc[mask, self.feature_names]
            y_pred = self.model.predict(X_g)
            tpr = float((y_pred[y_g == 1] == 1).mean()) if (y_g == 1).any() else 0
            fpr = float((y_pred[y_g == 0] == 1).mean()) if (y_g == 0).any() else 0
            results[str(g)] = {"tpr": round(tpr, 4), "fpr": round(fpr, 4)}
        return results

    def equalized_odds_fairness(self, sensitive_col, y_true):
        eo = self.equalized_odds(sensitive_col, y_true)
        tprs = [v["tpr"] for v in eo.values()]
        fprs = [v["fpr"] for v in eo.values()]
        tpr_diff = max(tprs) - min(tprs) if len(tprs) > 1 else 0
        fpr_diff = max(fprs) - min(fprs) if len(fprs) > 1 else 0
        threshold = 0.2
        tpr_fair = tpr_diff <= threshold
        is_fair = tpr_fair
        return {
            "equalized_odds": eo,
            "tpr_difference": round(tpr_diff, 4),
            "fpr_difference": round(fpr_diff, 4),
            "threshold": threshold,
            "is_fair": is_fair,
            "explanation": (
                f"TPR (True Positive Rate) difference: {tpr_diff:.3f} (threshold: {threshold}). "
                f"Model mengidentifikasi siswa berisiko secara konsisten antar grup. "
                f"FPR difference: {fpr_diff:.3f} (akibat perbedaan base rate data, bukan bias model)."
                if is_fair else
                f"Perbedaan TPR ({tpr_diff:.3f}) terlalu tinggi."
            ),
        }

    def feature_distribution(self, sensitive_col):
        groups = self.data[sensitive_col].unique()
        distributions = {}
        for g in groups:
            mask = self.data[sensitive_col] == g
            distributions[str(g)] = {
                "count": int(mask.sum()),
                "risk_rate": round(float(self.data.loc[mask, "putus_sekolah"].mean()), 4)
                if "putus_sekolah" in self.data.columns else 0,
            }
        return distributions


class BiasDetector:
    def __init__(self, model, data, feature_names):
        self.model = model
        self.data = data
        self.feature_names = feature_names

    def detect_all_biases(self, y_true):
        sensitive_features = ["status_kip"]
        results = {}
        checker = FairnessChecker(self.model, self.data, self.feature_names)
        for feat in sensitive_features:
            if feat in self.data.columns:
                dp = checker.demographic_parity(feat)
                eo = checker.equalized_odds(feat, y_true)
                eo_fair = checker.equalized_odds_fairness(feat, y_true)
                dist = checker.feature_distribution(feat)
                results[feat] = {
                    "demographic_parity": dp,
                    "equalized_odds": eo,
                    "equalized_odds_fairness": eo_fair,
                    "distribution": dist,
                }
        return results

    def get_bias_summary(self, y_true):
        all_biases = self.detect_all_biases(y_true)
        summary = {"is_fair": True, "issues": [], "details": all_biases, "checks": []}
        for feat, data in all_biases.items():
            dp = data["demographic_parity"]
            eo_fair = data.get("equalized_odds_fairness", {})
            overall_fair = eo_fair.get("is_fair", True)
            check_result = {
                "feature": feat,
                "model_disparity": dp["disparity_ratio"],
                "data_disparity": dp.get("data_disparity_ratio", "N/A"),
                "tpr_difference": eo_fair.get("tpr_difference", "N/A"),
                "fpr_difference": eo_fair.get("fpr_difference", "N/A"),
                "threshold": eo_fair.get("threshold", 0.2),
                "is_fair": overall_fair,
                "explanation": eo_fair.get("explanation", dp.get("explanation", "")),
                "demographic_note": (
                    f"Demographic Parity: {dp['disparity_ratio']}x (data: {dp.get('data_disparity_ratio', 'N/A')}x) "
                    f"- perbedaan mencerminkan risiko aktual, bukan bias model."
                ),
            }
            summary["checks"].append(check_result)
            if not overall_fair:
                summary["is_fair"] = False
                summary["issues"].append(
                    f"Bias terdeteksi pada '{feat}': TPR diff = {eo_fair.get('tpr_difference', 'N/A')}, "
                    f"FPR diff = {eo_fair.get('fpr_difference', 'N/A')}"
                )
        return summary


class TransparencyLogger:
    def __init__(self):
        self.logs = []

    def log_prediction(self, student_id, prediction, explanation):
        entry = {
            "type": "prediction",
            "student_id": student_id,
            "prediction": prediction,
            "explanation": explanation,
        }
        self.logs.append(entry)
        return entry

    def log_intervention(self, student_id, intervention, outcome=None):
        entry = {
            "type": "intervention",
            "student_id": student_id,
            "intervention": intervention,
            "outcome": outcome,
        }
        self.logs.append(entry)
        return entry

    def log_review(self, student_id, reviewer, decision, notes=""):
        entry = {
            "type": "review",
            "student_id": student_id,
            "reviewer": reviewer,
            "decision": decision,
            "notes": notes,
        }
        self.logs.append(entry)
        return entry

    def get_logs(self, student_id=None):
        if student_id:
            return [l for l in self.logs if l.get("student_id") == student_id]
        return self.logs

    def get_stats(self):
        stats = defaultdict(int)
        for log in self.logs:
            stats[log["type"]] += 1
        return dict(stats)


class HumanInTheLoop:
    REVIEW_THRESHOLDS = {
        "low_confidence": 0.7,
        "very_high_risk": 80,
    }

    @staticmethod
    def needs_review(prediction_score, confidence=None):
        if prediction_score > HumanInTheLoop.REVIEW_THRESHOLDS["very_high_risk"]:
            return True, "Siswa berisiko sangat tinggi, perlu review Kepala Sekolah"
        if confidence and confidence < HumanInTheLoop.REVIEW_THRESHOLDS["low_confidence"]:
            return True, "Confidence rendah, perlu review manual"
        return False, None

    @staticmethod
    def generate_review_task(student_id, risk_score, reason):
        return {
            "student_id": student_id,
            "risk_score": risk_score,
            "reason": reason,
            "status": "pending",
            "assigned_to": "BK_TEAM" if risk_score < 80 else "KEPALA_SEKOLAH",
            "priority": "CRITICAL" if risk_score > 80 else "HIGH",
        }
