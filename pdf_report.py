"""
SIGAP Sekolah - PDF Report Generator
Generate laporan untuk guru, orang tua, dan kepala sekolah.
"""

import os
from datetime import datetime
from fpdf import FPDF

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
ARIAL_PATH = os.path.join(FONT_DIR, "arial.ttf")
ARIAL_BOLD_PATH = os.path.join(FONT_DIR, "arialbd.ttf")

if not os.path.exists(ARIAL_PATH):
    ARIAL_PATH = "C:\\Windows\\Fonts\\arial.ttf"
    ARIAL_BOLD_PATH = "C:\\Windows\\Fonts\\arialbd.ttf"


class SIGAPReport(FPDF):
    def __init__(self):
        super().__init__()
        if os.path.exists(ARIAL_PATH):
            self.add_font("Arial", "", ARIAL_PATH, uni=True)
        if os.path.exists(ARIAL_BOLD_PATH):
            self.add_font("Arial", "B", ARIAL_BOLD_PATH, uni=True)

    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(30, 58, 95)
        self.cell(0, 10, "SIGAP Sekolah", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Arial", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "Sistem Identifikasi Gejala Anak Berisiko Putus Sekolah",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 58, 95)
        self.set_line_width(0.5)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"SIGAP Sekolah - LKS AI 2026 | Halaman {self.page_no()}/{{nb}}",
                  align="C")

    def section_title(self, title):
        self.set_font("Arial", "B", 13)
        self.set_text_color(30, 58, 95)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 58, 95)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def key_value(self, key, value):
        self.set_font("Arial", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(60, 7, key, new_x="RIGHT")
        self.set_font("Arial", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")

    def risk_badge(self, score):
        if score > 60:
            self.set_fill_color(231, 76, 60)
            self.set_text_color(255, 255, 255)
        elif score > 30:
            self.set_fill_color(243, 156, 18)
            self.set_text_color(255, 255, 255)
        else:
            self.set_fill_color(39, 174, 96)
            self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 10)
        self.cell(40, 8, f"  {score:.1f}%", fill=True, align="C")
        self.set_text_color(30, 30, 30)
        self.ln(10)


def generate_student_report(student_id, student_data, prediction, top_factors, recommendations, output_dir="reports"):
    pdf = SIGAPReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.section_title("LAPORAN ANALISIS RISIKO SISWA")
    pdf.key_value("Nama Siswa", student_data.get("nama", "-"))
    pdf.key_value("ID Siswa", student_id)
    pdf.key_value("Kelas", student_data.get("kelas", "-"))
    pdf.key_value("Tanggal", datetime.now().strftime("%d %B %Y"))
    pdf.key_value("Skor Risiko", f"{prediction['risk_score']:.1f}%")
    pdf.key_value("Status", prediction["risk_label"])
    pdf.ln(5)

    pdf.section_title("SKOR RISIKO")
    pdf.risk_badge(prediction["risk_score"])

    pdf.section_title("DATA SISWA")
    field_map = {
        "persentase_kehadiran": ("Kehadiran", "%"),
        "rata_rata_nilai": ("Rata-rata Nilai", ""),
        "tren_nilai": ("Tren Nilai", ""),
        "jumlah_mapel_di_bawah_kkm": ("Mapel Bawah KKM", ""),
        "status_kip": ("Penerima KIP", ""),
        "jarak_rumah_km": ("Jarak ke Sekolah", " km"),
        "jumlah_pelanggaran": ("Pelanggaran", ""),
    }
    for key, (label, unit) in field_map.items():
        if key in student_data:
            val = student_data[key]
            if key == "status_kip":
                val = "Ya" if val == 1 else "Tidak"
            pdf.key_value(label, f"{val}{unit}")
    pdf.ln(3)

    pdf.section_title("FAKTOR RISIKO UTAMA")
    for i, f in enumerate(top_factors, 1):
        direction = "MENINGKATKAN" if f["direction"] == "meningkatkan" else "MENURUNKAN"
        pdf.set_font("Arial", "B", 10)
        pdf.cell(8, 7, f"{i}.", new_x="RIGHT")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 7, f"{f['feature']} - {f['impact']}% {direction} risiko",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.section_title("REKOMENDASI INTERVENSI")
    for i, rec in enumerate(recommendations, 1):
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, f"{i}. {rec}")
        pdf.ln(1)

    pdf.ln(5)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5,
        "Catatan: Laporan ini dihasilkan oleh sistem AI (SIGAP Sekolah). "
        "Keputusan akhir tetap di tangan guru BK dan pihak sekolah. "
        "Data siswa dianonimkan untuk menjaga privasi.")

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"laporan_{student_id}_{datetime.now().strftime('%Y%m%d')}.pdf")
    pdf.output(filepath)
    return filepath


def generate_parent_letter(student_id, student_data, prediction, top_factors, output_dir="reports"):
    pdf = SIGAPReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    nama = student_data.get("nama", student_id)
    kelas = student_data.get("kelas", "")
    pdf.section_title("LAPORAN UNTUK ORANG TUA/WALI")
    pdf.ln(3)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Yth. Orang Tua/Wali dari: {nama} ({student_id}){(' - ' + kelas) if kelas else ''}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6,
        "Bersama surat ini, kami dari pihak sekolah ingin menyampaikan hasil "
        "evaluasi pemantauan akademik putra/putri Bapak/Ibu.")
    pdf.ln(3)
    pdf.key_value("Skor Pemantauan", f"{prediction['risk_score']:.1f}%")
    pdf.key_value("Status", prediction["risk_label"])
    pdf.ln(3)
    pdf.section_title("Area yang Perlu Perhatian")
    for f in top_factors[:3]:
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"- {f['feature']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6,
        "Kami mengundang Bapak/Ibu untuk datang ke sekolah guna berdiskusi "
        "lebih lanjut mengenai langkah dukungan yang dapat dilakukan bersama "
        "demi kemajuan akademik putra/putri Bapak/Ibu.")
    pdf.ln(5)
    pdf.cell(0, 6, f"Hormat kami,", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Tim Bimbingan Konseling", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, datetime.now().strftime("%d %B %Y"), new_x="LMARGIN", new_y="NEXT")

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"orangtua_{student_id}_{datetime.now().strftime('%Y%m%d')}.pdf")
    pdf.output(filepath)
    return filepath
