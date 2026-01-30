from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import csv
from datetime import datetime

# =========================
# Inisialisasi Flask
# =========================
app = Flask(__name__)

# =========================
# Load Model Naive Bayes
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nb_beasiswa_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[INFO] Model Naive Bayes berhasil dimuat.")
except FileNotFoundError:
    print(f"[ERROR] Model tidak ditemukan di: {MODEL_PATH}")
    model = None

# =========================
# Fungsi Pendukung
# =========================
def simpan_ke_csv(data):
    """Menyimpan riwayat prediksi ke file CSV"""
    file_path = os.path.join(BASE_DIR, 'hasil_prediksi.csv')
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Jika file baru, buat header
        if not file_exists:
            writer.writerow(['Tanggal', 'IPK', 'Semester', 'Penghasilan', 'Prestasi', 'Organisasi', 'Hasil'])
        writer.writerow(data)

# =========================
# Route Beranda
# =========================
@app.route("/")
def beranda():
    return render_template("beranda.html")

# =========================
# Route Prediksi
# =========================
@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    hasil = None

    if request.method == "POST":
        try:
            # 1. Ambil input dari form
            # 'penghasilan' diambil dari hidden input yang murni angka (dari JavaScript)
            ipk = float(request.form["ipk"])
            semester = int(request.form["semester"])
            penghasilan = float(request.form["penghasilan"])
            prestasi = int(request.form["prestasi"])
            aktif_organisasi = int(request.form["aktif_organisasi"])

            # 2. Susun data untuk model [IPK, Semester, Penghasilan, Prestasi, Organisasi]
            data_input = np.array([[ipk, semester, penghasilan, prestasi, aktif_organisasi]])

            # 3. Prediksi menggunakan model
            if model:
                pred = model.predict(data_input)
                status_label = "Diterima" if pred[0] == 1 else "Tidak Diterima"
                hasil = "🎉 DITERIMA BEASISWA" if pred[0] == 1 else "❌ TIDAK DITERIMA BEASISWA"
                
                # 4. Simpan riwayat ke CSV
                waktu = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                simpan_ke_csv([waktu, ipk, semester, penghasilan, prestasi, aktif_organisasi, status_label])
            else:
                hasil = "Kesalahan: Model ML belum dimuat."

        except Exception as e:
            hasil = f"Terjadi Kesalahan: {str(e)}"

    return render_template("prediksi.html", hasil=hasil)

# =========================
# Route Rekapan (Riwayat)
# =========================
@app.route("/rekapan")
def rekapan():
    data_rekapan = []
    file_path = os.path.join(BASE_DIR, 'hasil_prediksi.csv')
    
    if os.path.isfile(file_path):
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data_rekapan.append(row)
    
    # Menampilkan dari data terbaru (reverse list)
    return render_template("rekapan.html", rekapan=data_rekapan[::-1])

# =========================
# Jalankan Aplikasi
# =========================
if __name__ == "__main__":
    app.run(debug=True)