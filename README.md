# Tugas Take-Home Sesi 05: Advanced CNNs and Transfer Learning
**Klasifikasi Citra dengan Data Terbatas (Small-Data Regime)**

* **Nama:** Jefferson Gautama Swanto
* **NIM:** 2802474476 
* **Mata Kuliah:** Deep Learning

---

## 📌 Deskripsi Proyek
Repositori ini berisi implementasi komparatif klasifikasi citra berbasis **PyTorch** pada dataset **Oxford-IIIT Pet** (subset 10 ras hewan: 5 kucing dan 5 anjing, ~1000 citra) dengan skema pembagian data terstratifikasi (*Stratified Split* 70% Train, 15% Val, 15% Test).

Proyek ini mengevaluasi 5 kondisi eksperimen:
1. **B1 - Baseline from Scratch:** Pelatihan ResNet-18 dari inisialisasi bobot acak (`weights=None`).
2. **B2 - Feature Extraction:** ResNet-18 pretrained ImageNet-697 dengan seluruh lapisan *backbone* dibekukan, melatih hanya *classifier head* (5.130 parameter).
3. **B3 - Progressive Fine-Tuning:** Membuka kuncian blok residual `layer4` dan `fc` menggunakan *Differential Learning Rate*.
4. **B4 - Uji Kontrol Data Augmentation:** Pelatihan arsitektur B3 tanpa menerapkan pipeline augmentasi data (hanya deterministik *Resize*/*Crop*).
5. **B5 - Arsitektur Pembanding:** Implementasi transfer learning pada arsitektur efisien modern **MobileNetV3-Small**.

---