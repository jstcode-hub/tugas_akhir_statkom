## **Dokumentasi Menjalankan Streamlit**

### **1. Persiapan Awal**

Pastikan Anda telah menginstal **Python** (versi 3.7 atau lebih baru) di komputer Anda. Anda dapat mengunduh Python dari [python.org](https://www.python.org/downloads/).

### **2. Clone Repository**

Salin kode dari repository Git yang berisi aplikasi Streamlit.

1. Buka terminal atau command prompt.
2. Jalankan perintah berikut untuk meng-clone repository:

   ```bash
   git clone https://github.com/jstcode-hub/tugas_akhir_statkom.git
   ```

3. Masuk ke direktori project:
   ```bash
   cd tugas_akhir_statkom
   ```

---

### **3. Membuat Virtual Environment**

Sangat disarankan untuk menggunakan virtual environment agar dependensi aplikasi terisolasi.

1. Buat virtual environment:

   ```bash
   python -m venv venv
   ```

2. Aktifkan virtual environment:

   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. Setelah virtual environment aktif, prompt terminal akan menunjukkan `(venv)` di awal.

---

### **4. Install Dependencies**

Pastikan Anda berada di dalam virtual environment, kemudian jalankan perintah berikut untuk menginstal modul yang diperlukan:

1. Instal modul dari file `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

2. Jika tidak ada file `requirements.txt`, instal modul secara manual. Contoh modul utama:

   ```bash
   pip install streamlit pandas matplotlib openpyxl
   ```

3. Verifikasi bahwa modul telah terinstal:
   ```bash
   pip list
   ```

---

### **5. Menjalankan Aplikasi Streamlit**

Setelah semua dependensi terinstal, Anda dapat menjalankan aplikasi Streamlit.

1. Jalankan aplikasi menggunakan perintah berikut:

   ```bash
   streamlit run streamlit_app.py
   ```

2. Aplikasi Streamlit akan terbuka di browser secara otomatis. Jika tidak, Anda dapat membuka URL yang muncul di terminal (biasanya `http://localhost:8501`).

---

### **6. Troubleshooting**

Jika terjadi masalah, coba langkah berikut:

- **Dependency Error:** Pastikan Anda telah menginstal semua modul yang dibutuhkan.
- **Streamlit Tidak Ditemukan:** Pastikan virtual environment aktif saat menjalankan `streamlit run`.
- **Port Sudah Digunakan:** Jalankan Streamlit pada port yang berbeda:
  ```bash
  streamlit run nama_file.py --server.port=8502
  ```

---

### **7. Menonaktifkan Virtual Environment**

Setelah selesai, Anda dapat menonaktifkan virtual environment dengan perintah:

```bash
deactivate
```

---

### **8. Menyimpan Perubahan (Opsional)**

Jika Anda ingin menyimpan perubahan ke repository:

1. Tambahkan file yang diubah:

   ```bash
   git add .
   ```

2. Commit perubahan:

   ```bash
   git commit -m "Pesan commit Anda"
   ```

3. Push perubahan ke repository:
   ```bash
   git push origin main
   ```
