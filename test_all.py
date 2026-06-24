"""Quick test for SIGAP v4.0 - with dual-mode support"""
import sys
sys.path.insert(0, '.')
from model import train_all_models, load_data, prepare_features, save_all_models, load_all_models
from model import predict_risk, explain_prediction, get_top_factors, what_if_simulation
from model import detect_mode, get_feature_names, SIGAP_FEATURE_NAMES, UCI_FEATURE_NAMES
from fairness import BiasDetector, TransparencyLogger, HumanInTheLoop
from school_intelligence import SchoolRiskIntelligence
from pdf_report import generate_student_report, generate_parent_letter
from intervention import InterventionTracker, EarlyWarningSystem, ImpactMetrics, RiskMitigation, AppealSystem, AuditTrail

print("=== SIGAP v4.0 Test Suite ===")

print("\n[1] Loading data...")
import os
data_path = os.path.join(os.path.dirname(__file__), "data", "data_siswa.csv")
df = load_data()
mode = detect_mode(df)
feature_names = get_feature_names(mode)
print(f"    Mode: {mode.upper()}")
print(f"    Loaded {len(df)} records")
print(f"    Columns: {list(df.columns)}")
print(f"    Features: {len(feature_names)}")

print("\n[2] Training all models...")
X, y, data_mode = prepare_features(df)
models, metrics, scaler, X_test, y_test, fn = train_all_models(X, y)
for name, m in metrics.items():
    acc = m['accuracy'] * 100
    rec = m['recall'] * 100
    f1 = m['f1'] * 100
    auc = m.get('auc_roc', 0) * 100
    print(f"    {name}: Acc={acc:.1f}% Rec={rec:.1f}% F1={f1:.1f}% AUC={auc:.1f}%")

print("\n[3] Saving models...")
save_all_models(models, scaler, metrics, fn)
bundle = load_all_models()
print(f"    Models: {list(bundle['models'].keys())}")
print(f"    Feature names saved: {len(bundle.get('feature_names', []))}")

print("\n[4] Testing prediction...")
pred = predict_risk(None, df.head(1), model_name="ensemble", models_bundle=bundle)
print(f"    Score: {pred[0]['risk_score']:.1f}% Label: {pred[0]['risk_label']}")

print("\n[5] Testing SHAP explanation...")
X_sample = df[fn].iloc[[0]]
explanation = explain_prediction(bundle['models']['random_forest'], X_sample, 0)
top = get_top_factors(explanation, n=3)
for f in top:
    print(f"    {f['feature']}: {f['impact']:.1f}% {f['direction']}")

print("\n[6] Testing What-If simulation...")
row = df.iloc[0].to_dict()
changes = {fn[0]: float(df[fn[0]].iloc[0]) + 10}
sim = what_if_simulation(row, bundle, changes)
print(f"    Before: {sim['original_score']:.1f}% -> After: {sim['simulated_score']:.1f}% (reduction: {sim['reduction']:.1f}%)")

print("\n[7] Testing Fairness...")
y_true = df["putus_sekolah"]
detector = BiasDetector(bundle['models']['random_forest'], df, fn)
bias = detector.get_bias_summary(y_true)
print(f"    Fair: {bias['is_fair']} Issues: {len(bias['issues'])}")

print("\n[8] Testing School Intelligence...")
all_preds = predict_risk(None, df, model_name="ensemble", models_bundle=bundle)
intel = SchoolRiskIntelligence(df, all_preds, fn)
dashboard = intel.get_dashboard_data()
print(f"    Overview: {dashboard['overview']}")
print(f"    Heatmap entries: {len(dashboard['heatmap'])}")

print("\n[9] Testing PDF report...")
pred_single = predict_risk(None, df.head(1), model_name="ensemble", models_bundle=bundle)[0]
explanation = explain_prediction(bundle['models']['random_forest'], df[fn].iloc[[0]], 0)
top_factors = get_top_factors(explanation, n=5)
recommendations = ["Test recommendation 1", "Test recommendation 2"]
filepath = generate_student_report("TEST001", df.iloc[0].to_dict(), pred_single, top_factors, recommendations)
print(f"    PDF generated: {filepath}")
filepath2 = generate_parent_letter("TEST001", df.iloc[0].to_dict(), pred_single, top_factors)
print(f"    Parent letter: {filepath2}")

print("\n[10] Testing Human-in-the-loop...")
needs, reason = HumanInTheLoop.needs_review(85.0)
print(f"    Score 85% -> needs review: {needs}")

print("\n[11] Testing Intervention Tracker...")
tracker = InterventionTracker()
intv = tracker.add_intervention("STU0001", "Akademik", "Test intervention", priority="HIGH")
print(f"    Intervention added: {intv['id']}")
stats = tracker.get_stats()
print(f"    Stats: {stats}")

print("\n[12] Testing Early Warning System...")
alerts = EarlyWarningSystem.generate_alerts(all_preds[:5], df.head(5))
print(f"    Alerts generated: {len(alerts)}")

print("\n[13] Testing Impact Metrics...")
impact = ImpactMetrics.calculate_impact(tracker.interventions, all_preds)
print(f"    At risk: {impact['at_risk_students']}, Interventions: {impact['total_interventions']}")

print("\n[14] Testing Risk Mitigation...")
mitigations = RiskMitigation.get_all_mitigations()
status = RiskMitigation.get_implementation_status()
print(f"    Mitigations: {len(mitigations)}, Coverage: {status['coverage_percentage']}%")

print("\n[15] Testing Appeal System...")
appeal_system = AppealSystem()
appeal = appeal_system.submit_appeal("STU0001", "Test appeal")
print(f"    Appeal submitted: {appeal['id']}")

print("\n[16] Testing Audit Trail...")
audit = AuditTrail()
audit.log("TEST_ACTION", "test_user", {"detail": "test"})
audit_stats = audit.get_audit_stats()
print(f"    Audit logs: {audit_stats['total_logs']}")

print("\n=== ALL TESTS PASSED! ===")
