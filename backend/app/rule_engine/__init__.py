"""
SIGAP Sekolah - Rule Engine
Perhitungan risiko putus sekolah berdasarkan aturan SMK.
"""

MAX_BOBOT = 2.33


def hitung_risiko(persen_mapel_tuntas, nilai_produktif, kehadiran, alpha_hari,
                  prestasi_lomba, pelanggaran_berat, nilai_sikap, lulus_ukk):
    """
    Menghitung skor risiko putus sekolah berdasarkan 6 kriteria.

    Returns:
        risk_score: float (0-100)
        aturan_dilanggar: list of tuples (nama_aturan, bobot)
    """
    aturan_dilanggar = []

    if persen_mapel_tuntas < 70:
        aturan_dilanggar.append(('Ketuntasan Mapel <70%', 1.0))
    else:
        aturan_dilanggar.append(('Ketuntasan Mapel ≥70%', 0.0))

    if nilai_produktif < 65:
        aturan_dilanggar.append(('Nilai Produktif <65', 1.0))
    else:
        aturan_dilanggar.append(('Nilai Produktif ≥65', 0.0))

    if pelanggaran_berat == 1:
        aturan_dilanggar.append(('Pelanggaran Berat', 1.0))
    else:
        aturan_dilanggar.append(('Pelanggaran Berat (Nihil)', 0.0))

    if prestasi_lomba == 0:
        if kehadiran < 90 or alpha_hari > 11:
            aturan_dilanggar.append(('Absensi (Hadir<90% atau Alpa>11)', 0.8))
        else:
            aturan_dilanggar.append(('Absensi (Hadir≥90%, Alpa≤11)', 0.0))
    else:
        if kehadiran < 85:
            aturan_dilanggar.append(('Absensi (Hadir<85% walau Berprestasi)', 0.8))
        else:
            aturan_dilanggar.append(('Absensi (Hadir≥85%, Keringanan Prestasi)', 0.0))

    if nilai_sikap < 75:
        aturan_dilanggar.append(('Sikap Harian <75', 0.7))
    else:
        aturan_dilanggar.append(('Sikap Harian ≥75', 0.0))

    if lulus_ukk == 0:
        aturan_dilanggar.append(('UKK Tidak Lulus', 1.0))
    else:
        aturan_dilanggar.append(('UKK Lulus', 0.0))

    total_bobot = sum(bobot for _, bobot in aturan_dilanggar)
    risk_score = min((total_bobot / MAX_BOBOT) * 100, 100)
    return risk_score, aturan_dilanggar


def get_risk_category(score):
    """Mengategorikan skor risiko."""
    if score < 30:
        return "aman"
    elif score < 65:
        return "waspada"
    else:
        return "berisiko"


def get_risk_label(category):
    """Label manusiawi untuk kategori risiko."""
    return {
        "aman": "Aman",
        "waspada": "Waspada",
        "berisiko": "Berisiko Tinggi"
    }.get(category, "Unknown")


def get_recommendations(category, violations):
    """Menghasilkan rekomendasi intervensi berdasarkan pelanggaran."""
    recommendations = []

    for rule_name, bobot in violations:
        if bobot > 0:
            if "Mapel" in rule_name:
                recommendations.append({
                    "area": "Akademik",
                    "action": "Lakukan remedial dan bimbingan belajar intensif untuk mata pelajaran yang belum tuntas.",
                    "priority": "tinggi" if bobot >= 1.0 else "sedang"
                })
            elif "Produktif" in rule_name:
                recommendations.append({
                    "area": "Kompetensi",
                    "action": "Berikan praktik tambahan dan pendampingan keterampilan produktif.",
                    "priority": "tinggi" if bobot >= 1.0 else "sedang"
                })
            elif "Pelanggaran" in rule_name:
                recommendations.append({
                    "area": "Kedisiplinan",
                    "action": "Lakukan konseling mendalam dan perjanjian perilaku dengan orang tua.",
                    "priority": "tinggi"
                })
            elif "Absensi" in rule_name:
                recommendations.append({
                    "area": "Kehadiran",
                    "action": "Hubungi orang tua untuk memahami penyebab ketidakhadiran.",
                    "priority": "tinggi" if bobot >= 0.8 else "sedang"
                })
            elif "Sikap" in rule_name:
                recommendations.append({
                    "area": "Sikap",
                    "action": "Pantau perilaku dan berikan pembinaan karakter.",
                    "priority": "sedang"
                })
            elif "UKK" in rule_name:
                recommendations.append({
                    "area": "Praktik",
                    "action": "Siapkan pembinaan UKK tambahan dan praktik mandiri.",
                    "priority": "tinggi"
                })

    if not recommendations:
        recommendations.append({
            "area": "Umum",
            "action": "Siswa dalam kondisi aman. Lanjutkan pemantauan rutin.",
            "priority": "rendah"
        })

    return recommendations
