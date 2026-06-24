"""
SIGAP Sekolah - AI BK Assistant
Chatbot untuk membantu guru memahami siswa berisiko.
"""

import pandas as pd
from datetime import datetime


class AIBKAssistant:
    def __init__(self, models_bundle, data_df):
        self.models = models_bundle
        self.data = data_df
        self.conversation_history = {}

    def get_student_context(self, student_id):
        if "id_siswa" in self.data.columns:
            row = self.data[self.data["id_siswa"] == student_id]
        else:
            row = self.data.iloc[[student_id]] if isinstance(student_id, int) else None
        if row is None or row.empty:
            return None
        return row.iloc[0].to_dict()

    def analyze_student(self, student_id):
        ctx = self.get_student_context(student_id)
        if ctx is None:
            return {"error": "Siswa tidak ditemukan"}
        from model import predict_risk, explain_prediction, get_top_factors, FEATURE_NAMES, FEATURE_LABELS
        df_single = pd.DataFrame([ctx])
        X_single = df_single[FEATURE_NAMES]
        rf_model = self.models["models"]["random_forest"]
        pred = predict_risk(rf_model, df_single, model_name="ensemble", models_bundle=self.models)[0]
        explanation = explain_prediction(rf_model, X_single, 0)
        top_factors = get_top_factors(explanation, n=5)
        return {
            "student_id": student_id,
            "context": ctx,
            "prediction": pred,
            "explanation": explanation,
            "top_factors": top_factors,
        }

    def chat(self, student_id, user_message):
        analysis = self.analyze_student(student_id)
        if "error" in analysis:
            return {"response": analysis["error"], "type": "error"}

        msg_lower = user_message.lower()
        pred = analysis["prediction"]
        factors = analysis["top_factors"]
        ctx = analysis["context"]

        if any(w in msg_lower for w in ["mengapa", "kenapa", "sebab", "alasan", "penyebab"]):
            response = self._explain_risk(student_id, analysis)
            response["type"] = "explanation"
        elif any(w in msg_lower for w in ["rekomendasi", "saran", "solusi", "tindakan", "harus"]):
            response = self._recommend_interventions(student_id, analysis)
            response["type"] = "recommendation"
        elif any(w in msg_lower for w in ["ringkasan", "resume", "laporan", "summary"]):
            response = self._generate_case_summary(student_id, analysis)
            response["type"] = "summary"
        elif any(w in msg_lower for w in ["orang tua", "wali", "parent", "ibu", "bapak"]):
            response = self._parent_report(student_id, analysis)
            response["type"] = "parent_report"
        elif any(w in msg_lower for w in ["bandingkan", "compare", "vs", "perbandingan"]):
            response = self._compare_with_classmates(student_id, analysis)
            response["type"] = "comparison"
        elif any(w in msg_lower for w in ["halo", "hai", "hello", "hi"]):
            nama = ctx.get("nama", student_id)
            kelas = ctx.get("kelas", "")
            response = {
                "jawaban": f"Halo! Saya AI BK Assistant untuk siswa **{nama}** ({student_id}){(' - ' + kelas) if kelas else ''}. "
                           f"Siswa ini memiliki skor risiko **{pred['risk_score']}%** ({pred['risk_label']}). "
                           f"Silakan tanyakan tentang:\n"
                           f"- Mengapa siswa berisiko?\n"
                           f"- Rekomendasi intervensi\n"
                           f"- Ringkasan kasus\n"
                           f"- Laporan untuk orang tua\n"
                           f"- Perbandingan dengan teman sekelas",
            }
            response["type"] = "greeting"
        else:
            response = self._general_analysis(student_id, analysis)
            response["type"] = "general"

        if student_id not in self.conversation_history:
            self.conversation_history[student_id] = []
        self.conversation_history[student_id].append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": response.get("jawaban", ""),
        })
        return response

    def _explain_risk(self, student_id, analysis):
        pred = analysis["prediction"]
        factors = analysis["top_factors"]
        ctx = analysis["context"]
        nama = ctx.get("nama", student_id)
        kelas = ctx.get("kelas", "")

        lines = [f"## Analisis Risiko: {nama} ({student_id}){(' - ' + kelas) if kelas else ''}\n"]
        lines.append(f"**Skor Risiko:** {pred['risk_score']}% ({pred['risk_label']})\n")
        lines.append("### Faktor Utama yang Mempengaruhi:\n")

        for i, f in enumerate(factors, 1):
            emoji = "🔴" if f["direction"] == "meningkatkan" else "🟢"
            lines.append(f"{i}. {emoji} **{f['feature']}** — {f['impact']}% {f['direction']} risiko")

        lines.append("\n### Konteks Data Siswa:")
        if ctx.get("persentase_kehadiran"):
            avg_hadir = self.data["persentase_kehadiran"].mean() if "persentase_kehadiran" in self.data.columns else 85
            selisih = ctx["persentase_kehadiran"] - avg_hadir
            lines.append(f"- Kehadiran: **{ctx['persentase_kehadiran']}%** (rata-rata kelas: {avg_hadir:.0f}%, selisih: {selisih:+.1f}%)")
        if ctx.get("rata_rata_nilai"):
            avg_nilai = self.data["rata_rata_nilai"].mean() if "rata_rata_nilai" in self.data.columns else 70
            lines.append(f"- Nilai Rata-rata: **{ctx['rata_rata_nilai']}** (rata-rata kelas: {avg_nilai:.0f})")

        lines.append("\n> 💡 *Semua faktor ini dianalisis oleh model AI. Keputusan akhir tetap di tangan guru BK.*")
        return {"jawaban": "\n".join(lines), "faktor_utama": factors}

    def _recommend_interventions(self, student_id, analysis):
        pred = analysis["prediction"]
        ctx = analysis["context"]
        factors = analysis["top_factors"]
        nama = ctx.get("nama", student_id)
        kelas = ctx.get("kelas", "")

        lines = [f"## Rekomendasi Intervensi: {nama} ({student_id}){(' - ' + kelas) if kelas else ''}\n"]
        lines.append(f"**Status:** {pred['risk_score']}% — {pred['risk_label']}\n")
        lines.append("### Tindakan yang Direkomendasikan:\n")

        interventions = []
        if ctx.get("persentase_kehadiran", 100) < 75:
            interventions.append(("🔴 TINGGI", "Kehadiran",
                "Hubungi orang tua segera. Identifikasi penyebab ketidakhadiran. "
                "Jika masalah transportasi, koordinasi antar-jemput."))
        if ctx.get("rata_rata_nilai", 100) < 65:
            interventions.append(("🔴 TINGGI", "Akademik",
                "Susun program remedial. Sediakan bimbingan belajar 2x seminggu. "
                "Prioritaskan mata pelajaran dengan nilai terendah."))
        if ctx.get("jumlah_pelanggaran", 0) > 5:
            interventions.append(("🔴 TINGGI", "Kedisiplinan",
                "Lakukan konseling mendalam. Buat perjanjian perilaku dengan orang tua. "
                "Pantau mingguan."))
        if ctx.get("status_kip", 0) == 1:
            interventions.append(("🟡 SEDANG", "Ekonomi",
                "Pastikan bantuan KIP/PIP tepat sasaran. Koordinasi dinas sosial."))
        if ctx.get("jarak_rumah_km", 0) > 8:
            interventions.append(("🟡 SEDANG", "Transportasi",
                "Pertimbangkan solusi antar-jemput atau pembelajaran daring parsial."))
        if ctx.get("tren_nilai", 0) < -3:
            interventions.append(("🟡 SEDANG", "Tren",
                "Nilai menunjukkan tren menurun. Identifikasi perubahan kondisi siswa."))

        if not interventions:
            interventions.append(("🟢 RENDAH", "Umum",
                "Lanjutkan pemantauan rutin. Siswa dalam kondisi relatif aman."))

        for pri, area, action in interventions:
            lines.append(f"**{pri} — {area}**\n{action}\n")

        lines.append("\n### Jadwal Tindak Lanjut:")
        lines.append("1. **Minggu 1:** Evaluasi awal dan kontak orang tua")
        lines.append("2. **Minggu 2-4:** Implementasi program intervensi")
        lines.append("3. **Minggu 5-6:** Evaluasi pertengahan")
        lines.append("4. **Minggu 8:** Evaluasi akhir dan penyesuaian")

        return {"jawaban": "\n".join(lines), "rekomendasi": interventions}

    def _generate_case_summary(self, student_id, analysis):
        pred = analysis["prediction"]
        ctx = analysis["context"]
        factors = analysis["top_factors"]
        nama = ctx.get("nama", student_id)
        kelas = ctx.get("kelas", "")

        lines = [f"## Ringkasan Kasus: {nama} ({student_id}){(' - ' + kelas) if kelas else ''}\n"]
        lines.append(f"**Tanggal:** {datetime.now().strftime('%d %B %Y')}")
        lines.append(f"**Skor Risiko:** {pred['risk_score']}%")
        lines.append(f"**Status:** {pred['risk_label']}\n")

        lines.append("### Profil Siswa:")
        lines.append(f"- Kehadiran: {ctx.get('persentase_kehadiran', 'N/A')}%")
        lines.append(f"- Nilai Rata-rata: {ctx.get('rata_rata_nilai', 'N/A')}")
        lines.append(f"- Tren Nilai: {ctx.get('tren_nilai', 'N/A')}")
        lines.append(f"- Mapel di Bawah KKM: {ctx.get('jumlah_mapel_di_bawah_kkm', 0)}")
        lines.append(f"- Pelanggaran: {ctx.get('jumlah_pelanggaran', 0)}")
        lines.append(f"- Status KIP: {'Ya' if ctx.get('status_kip', 0) == 1 else 'Tidak'}")
        lines.append(f"- Jarak ke Sekolah: {ctx.get('jarak_rumah_km', 'N/A')} km\n")

        lines.append("### Faktor Risiko Utama:")
        for i, f in enumerate(factors, 1):
            lines.append(f"{i}. {f['feature']}: {f['impact']}% {f['direction']}")

        lines.append("\n### Kesimpulan:")
        if pred["is_at_risk"]:
            lines.append(f"Siswa {student_id} berada dalam kategori **BERISIKO TINGGI** dengan skor {pred['risk_score']}%. "
                        f"Tindakan intervensi segera diperlukan.")
        else:
            lines.append(f"Siswa {student_id} dalam kondisi **AMAN** (skor {pred['risk_score']}%). "
                        f"Lanjutkan pemantauan rutin.")

        return {"jawaban": "\n".join(lines), "ringkasan": True}

    def _parent_report(self, student_id, analysis):
        pred = analysis["prediction"]
        ctx = analysis["context"]
        factors = analysis["top_factors"]
        nama = ctx.get("nama", student_id)
        kelas = ctx.get("kelas", "")

        lines = [f"Laporan untuk Orang Tua/Wali - {nama} ({student_id}){(' - ' + kelas) if kelas else ''}\n"]
        lines.append(f"Yth. Orang Tua/Wali dari {nama} ({student_id}),\n")
        lines.append("Bersama surat ini, kami ingin menyampaikan hasil evaluasi akademik "
                     f"siswa {student_id} di sekolah kami.\n")
        lines.append(f"Berdasarkan analisis kami, siswa saat ini memiliki skor pemantauan "
                     f"akademik sebesar **{pred['risk_score']}%**.\n")

        lines.append("Area yang perlu perhatian:")
        for f in factors[:3]:
            lines.append(f"- {f['feature']}")

        lines.append("\nKami mengundang Bapak/Ibu untuk datang ke sekolah guna "
                     "berdiskusi lebih lanjut mengenai langkah dukungan yang dapat "
                     "dilakukan bersama demi kemajuan akademik siswa.")
        lines.append("\nHormat kami,\nTim BK Sekolah")

        return {"jawaban": "\n".join(lines), "for_parent": True}

    def _compare_with_classmates(self, student_id, analysis):
        pred = analysis["prediction"]
        ctx = analysis["context"]
        nama = ctx.get("nama", student_id)
        kelas = ctx.get("kelas", "")

        lines = [f"## Perbandingan: {nama} ({student_id}){(' - ' + kelas) if kelas else ''} vs Rata-rata Kelas\n"]

        metrics = [
            ("persentase_kehadiran", "Kehadiran", "%"),
            ("rata_rata_nilai", "Nilai Rata-rata", ""),
        ]

        for key, label, unit in metrics:
            if key in ctx and key in self.data.columns:
                siswa_val = ctx[key]
                avg_val = self.data[key].mean()
                diff = siswa_val - avg_val
                bar_siswa = "█" * int(siswa_val / 5)
                bar_avg = "█" * int(avg_val / 5)
                emoji = "⚠️" if diff < -10 else "✅" if diff > 0 else "➡️"
                lines.append(f"### {label} {emoji}")
                lines.append(f"Siswa:  {bar_siswa} {siswa_val}{unit}")
                lines.append(f"Kelas:  {bar_avg} {avg_val:.1f}{unit}")
                lines.append(f"Selisih: {diff:+.1f}{unit}\n")

        lines.append(f"**Skor Risiko Siswa:** {pred['risk_score']}%")
        if "putus_sekolah" in self.data.columns:
            avg_risk = self.data["putus_sekolah"].mean() * 100
            lines.append(f"**Risiko Rata-rata Kelas:** {avg_risk:.1f}%")

        return {"jawaban": "\n".join(lines)}

    def _general_analysis(self, student_id, analysis):
        pred = analysis["prediction"]
        return {
            "jawaban": f"## Status {student_id}\n"
                       f"Skor Risiko: **{pred['risk_score']}%** ({pred['risk_label']})\n\n"
                       f"Saya bisa membantu Anda dengan:\n"
                       f"- **Mengapa** siswa berisiko? Ketik: 'Mengapa siswa berisiko?'\n"
                       f"- **Rekomendasi** intervensi? Ketik: 'Rekomendasi intervensi'\n"
                       f"- **Ringkasan** kasus? Ketik: 'Ringkasan kasus'\n"
                       f"- **Laporan** untuk orang tua? Ketik: 'Laporan untuk orang tua'\n"
                       f"- **Perbandingan** dengan teman? Ketik: 'Bandingkan dengan teman'",
        }
