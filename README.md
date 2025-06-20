# Classification of Obesity Levels using Random Forest Model

## Deskripsi Proyek
Proyek ini merupakan bagian dari tugas akhir Mata Kuliah **Kecerdasan Artifisial** di Universitas Negeri Surabaya. Tujuan utama proyek adalah mengembangkan sistem prediksi tingkat obesitas berdasarkan data antropometri, gaya hidup, dan kebiasaan makan menggunakan algoritma **Random Forest**.

Dataset yang digunakan berasal dari UCI Machine Learning Repository dan mencakup data individu dari Meksiko, Peru, dan Kolombia.

---

## Tujuan
- Memprediksi tingkat obesitas seseorang berdasarkan data input seperti tinggi badan, berat badan, umur, dan pola makan.
- Menerapkan algoritma Random Forest dan membandingkan kinerjanya dengan K-Nearest Neighbor (KNN).
- Mengembangkan aplikasi web interaktif berbasis **Streamlit** untuk klasifikasi obesitas.

---

## Dataset
- 📌 **Sumber**: [Obesity Dataset - UCI Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- 🧾 **Jumlah Data**: 2111 entri
- 🔢 **Fitur**: 17 atribut, termasuk umur, berat, tinggi, pola makan, transportasi, dan lainnya
- 🏷️ **Label**: `NObeyesdad` (7 kelas: Normal, Overweight Level I/II, Obesity Type I/II/III, Insufficient Weight)

---

## Algoritma yang Digunakan
### 1. Random Forest
- Ensemble method berbasis Decision Tree
- Parameter: 
  - `n_trees = 100`
  - `max_depth = 20`
  - `min_samples_split = 2`
- Akurasi akhir: **98%**
- Tahan terhadap outlier dan tidak memerlukan normalisasi

### 2. K-Nearest Neighbors (KNN)
- Parameter: `k=2`
- Digunakan 2 jenis metrik:
  - Euclidean Distance
  - Manhattan Distance
- Akurasi akhir: **94%**
- Membutuhkan **normalisasi** agar performa optimal

---

## Implementasi Aplikasi
Aplikasi berbasis web dikembangkan menggunakan **Streamlit**:
- Input data user berupa kuisioner
- Mapping dan encoding input
- Prediksi dilakukan dengan model Random Forest
- Output: Prediksi kategori obesitas

🔗 [🧪 Coba Aplikasinya](https://obesity-app-random-forest-2023b-kelompok-10.streamlit.app/)  

---

## Hasil Evaluasi
| Model         | Akurasi | Catatan |
|---------------|---------|---------|
| Random Forest | 98%     | Akurasi tinggi, stabil |
| KNN (Euclidean/Manhattan) | 94% | Tergantung pada normalisasi data |

---

## Kontributor
| Nama | NIM | 
|------|-----|
| Putri Manika Rukmamaya | 23031554091 | 
| Maysahayu Artika Maharani | 23031554214 | 
| Gesang Nur Zamroji | 23031554145 | 
