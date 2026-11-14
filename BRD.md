# Business Requirements Document (BRD)

**Sistem:** FaceCheck AI, Klasifikasi Citra Wajah Asli dan Citra Wajah Hasil Generative AI
**Topik TA:** Analisis Perbandingan Kinerja CNN dan Vision Transformer (ViT) untuk Klasifikasi Citra Wajah Asli dan Citra Wajah Hasil Generative AI
**Institusi:** Fakultas Ilmu Komputer, Universitas Esa Unggul
**Versi / Status:** 1.0 / Draf
**Tanggal:** 14 November 2026
**Penyusun:** Teguh Tegar Aulia


## Daftar Isi

1. Executive Summary
2. Business Objectives & Success Metrics
3. Scope
4. Stakeholder Analysis
5. Functional Requirements
6. Non-Functional Requirements
7. Business Process As-Is vs To-Be
8. Assumptions & Constraints
9. Risk Register
10. Acceptance Criteria

---

## 1. Executive Summary

### 1.1 Latar Belakang Bisnis

Perkembangan *Generative Artificial Intelligence* (Generative AI) memungkinkan pembuatan citra wajah sintetis yang tampak realistis. Citra semacam ini bermanfaat untuk kebutuhan kreatif, tetapi menimbulkan persoalan ketika asal-usulnya tidak diketahui atau tidak dijelaskan. Dalam lingkungan akademik, kebutuhan untuk memahami, mendemonstrasikan, dan membandingkan metode pemeriksaan keaslian citra wajah secara komputasional semakin relevan, baik untuk pembelajaran, penelitian, maupun literasi digital.

Dua keluarga arsitektur pembelajaran mendalam yang umum digunakan untuk klasifikasi citra adalah *Convolutional Neural Network* (CNN) dan *Vision Transformer* (ViT). Penelitian ini membandingkan ResNet-50 (CNN) dan ViT-Base/16 (ViT) pada tugas klasifikasi biner citra wajah *Real* dan *AI-Generated*, serta membangun prototipe web **FaceCheck AI** sebagai media demonstrasi.

### 1.2 Masalah Nyata

| No | Masalah | Dampak |
|---|---|---|
| M-01 | Kemiripan visual wajah asli dan wajah sintetis menyulitkan pemeriksaan keaslian hanya melalui pengamatan manusia. | Pemeriksaan bergantung pada penilaian subjektif dan tidak terukur. |
| M-02 | Literatur yang ada belum secara langsung membandingkan ResNet-50 dan ViT-Base/16 pada satu dataset wajah, satu pembagian data, dan satu konfigurasi pelatihan yang sama (celah penelitian sementara). | Pemilihan arsitektur untuk domain wajah sintetis belum didukung bukti komparatif yang terkontrol. |
| M-03 | Perbandingan model yang hanya berbasis metrik klasifikasi tidak memperlihatkan perbedaan efisiensi komputasi. | Keputusan pemilihan model tidak mempertimbangkan waktu inferensi. |
| M-04 | Tidak tersedia media sederhana bagi mahasiswa dan peneliti untuk mencoba kedua model pada citra baru. | Proses pembelajaran dan demonstrasi terhambat. |

### 1.3 Tujuan Sistem

Sistem FaceCheck AI bertujuan:

1. menyediakan dua model klasifikasi (ResNet-50 dan ViT-Base/16) yang dilatih dan dievaluasi dengan dataset serta protokol yang sama;
2. menghasilkan perbandingan kinerja klasifikasi dan efisiensi inferensi yang terdokumentasi;
3. menyediakan prototipe web untuk mengunggah citra wajah dan menampilkan hasil klasifikasi kedua model sebagai media demonstrasi.

> **Penegasan batas penggunaan.** Sistem tidak dimaksudkan sebagai alat verifikasi identitas, alat bukti forensik, atau penentu keaslian citra dengan kepastian mutlak. Hasil klasifikasi bersifat probabilistik dan terbatas pada domain dataset pelatihan.

---

## 2. Business Objectives & Success Metrics

### 2.1 Tujuan Bisnis (SMART)

| ID | Tujuan | Specific | Measurable | Achievable | Relevant | Time-bound |
|---|---|---|---|---|---|---|
| BO-01 | Menerapkan ResNet-50 dan ViT-Base/16 untuk klasifikasi biner wajah *Real* vs *AI-Generated* | Kedua model dilatih dengan *transfer learning* (*backbone* dibekukan, *classifier head* dilatih) pada dataset dan konfigurasi yang sama | Dua model terlatih tersimpan dan dapat dimuat ulang; konfigurasi terdokumentasi | Memanfaatkan bobot pralatih ImageNet | Menjawab rumusan masalah 1 | [ISI DATA] |
| BO-02 | Membandingkan kinerja klasifikasi kedua model | Evaluasi pada data uji dengan protokol yang sama | Seluruh metrik pada Tabel 2.2 dilaporkan untuk kedua model | Metrik dihitung dengan pustaka evaluasi standar | Menjawab rumusan masalah 2 | [ISI DATA] |
| BO-03 | Membandingkan efisiensi inferensi kedua model | Pengukuran waktu inferensi pada perangkat dan prosedur yang sama | Waktu inferensi rata-rata per citra dilaporkan untuk kedua model | Pengukuran dilakukan pada satu perangkat uji | Menjawab rumusan masalah 3 | [ISI DATA] |
| BO-04 | Menyediakan prototipe web demonstrasi | Prototipe menerima unggahan citra dan menampilkan hasil klasifikasi kedua model | Seluruh skenario pengujian fungsional (*black-box*) berstatus lulus | Dibangun di atas model yang sudah terlatih | Mendukung manfaat praktis penelitian | [ISI DATA] |
| BO-05 | Melaporkan validitas dan keterbatasan dataset secara transparan | Audit duplikasi, potensi kebocoran data, dan cakupan generator | Hasil audit dan daftar keterbatasan tercantum pada laporan | Audit dilakukan pada dataset yang sama | Mencegah kesimpulan yang berlebihan | [ISI DATA] |

### 2.2 Success Metrics

| ID | Metrik | Definisi | Target | Sumber Data |
|---|---|---|---|---|
| SM-01 | *Accuracy* | Proporsi prediksi benar terhadap seluruh data uji | [ISI DATA] | Data uji |
| SM-02 | *Precision* | TP / (TP + FP) | [ISI DATA] | Data uji |
| SM-03 | *Recall* | TP / (TP + FN) | [ISI DATA] | Data uji |
| SM-04 | *Specificity* | TN / (TN + FP) | [ISI DATA] | Data uji |
| SM-05 | *F1-score* dan *Macro-F1* | Rerata harmonik *precision* dan *recall*; rerata F1 antarkelas | [ISI DATA] | Data uji |
| SM-06 | *ROC-AUC* | Luas area di bawah kurva ROC | [ISI DATA] | Data uji |
| SM-07 | *Confusion matrix* | Rincian TP, FP, TN, FN untuk tiap model | Tersaji untuk kedua model | Data uji |
| SM-08 | Waktu inferensi | Waktu rata-rata per citra pada perangkat uji yang sama | [ISI DATA] | Skrip pengukuran |
| SM-09 | Keberhasilan pengujian fungsional | Persentase skenario *black-box* berstatus lulus | [ISI DATA] | Lembar pengujian |
| SM-10 | Kelengkapan dokumentasi eksperimen | Seluruh konfigurasi, *seed*, dan versi pustaka tercatat | 100% butir tercatat | Laporan TA |

> Target numerik tidak ditetapkan di awal karena penelitian ini tidak mengasumsikan salah satu model pasti lebih unggul. Target ditetapkan oleh penulis bersama pembimbing sebelum pengujian.

---

## 3. Scope

### 3.1 In-Scope

| No | Cakupan |
|---|---|
| IS-01 | Klasifikasi biner citra wajah manusia: *Real* dan *AI-Generated*. |
| IS-02 | Penggunaan dataset Hugging Face *shravya11/human-faces-dataset* (5.000 citra *Real* dari CelebA; 4.630 citra *AI-Generated* dari StyleGAN/GAN; total 9.630 citra). |
| IS-03 | Pembagian data terstratifikasi 70% pelatihan, 15% validasi, 15% pengujian. |
| IS-04 | Prapemrosesan citra yang seragam untuk kedua model. |
| IS-05 | Penerapan ResNet-50 dan ViT-Base/16 dengan *transfer learning* (bobot pralatih ImageNet, *backbone* dibekukan, *classifier head* dilatih). |
| IS-06 | Evaluasi: *accuracy*, *precision*, *recall*, *specificity*, *F1-score*, *Macro-F1*, *ROC-AUC*, *confusion matrix*. |
| IS-07 | Pengukuran waktu inferensi pada perangkat dan prosedur yang sama. |
| IS-08 | Prototipe web FaceCheck AI dengan deteksi wajah Haar Cascade. |
| IS-09 | Pengujian fungsional *black-box* prototipe. |
| IS-10 | Audit validitas dataset dan pembahasan ancaman terhadap validitas. |

### 3.2 Out-of-Scope

| No | Di Luar Cakupan | Alasan |
|---|---|---|
| OS-01 | Verifikasi atau pengenalan identitas seseorang | Bukan tujuan penelitian; menimbulkan risiko privasi. |
| OS-02 | Bukti forensik digital atau penentuan keaslian dengan kepastian mutlak | Hasil model bersifat probabilistik. |
| OS-03 | Deteksi citra hasil model difusi dan generator selain StyleGAN/GAN | Dataset tidak mencakup jenis generator tersebut. |
| OS-04 | Deteksi video, audio, atau konten *deepfake* bergerak | Penelitian terbatas pada citra statis. |
| OS-05 | Klasifikasi citra non-wajah | Di luar domain penelitian. |
| OS-06 | *Fine-tuning* penuh seluruh parameter *backbone* dan pencarian arsitektur baru | Rancangan eksperimen membekukan *backbone*. |
| OS-07 | Penerapan produksi skala besar (*high availability*, *load balancing*) | Prototipe bersifat demonstrasi. |
| OS-08 | Aplikasi seluler *native* | Prototipe berbasis web. |
| OS-09 | Penyimpanan permanen citra unggahan pengguna untuk pelatihan ulang | Pertimbangan privasi; lihat NFR-SEC. |

---

## 4. Stakeholder Analysis

| Stakeholder | Peran | Kepentingan | Influence |
|---|---|---|---|
| Mahasiswa peneliti (penulis TA) | Perancang, pengembang, penguji, dan penulis laporan | Menyelesaikan TA yang valid, dapat direproduksi, dan memenuhi syarat kelulusan | Tinggi |
| Dosen pembimbing | Pengarah metodologi dan penilai kemajuan | Kesesuaian metodologi, kualitas ilmiah, dan ketepatan waktu | Tinggi |
| Dosen penguji / tim seminar | Penilai kelayakan proposal dan hasil | Kejelasan masalah, kebaruan, validitas eksperimen | Tinggi |
| Program Studi Teknik Informatika | Penyelenggara akademik | Kepatuhan pada pedoman TA dan luaran penelitian | Sedang |
| Fakultas Ilmu Komputer | Pemangku kebijakan fakultas dan penyedia sumber daya | Luaran yang berkualitas dan sejalan dengan bidang keilmuan | Sedang |
| Mahasiswa dan peneliti lain (calon pengguna) | Pengguna prototipe dan pembaca laporan | Media demonstrasi, bahan referensi, dan data pembanding | Rendah |
| Penyedia dataset dan model pralatih (Hugging Face, sumber ImageNet) | Penyedia sumber data dan bobot | Kepatuhan pada lisensi dan ketentuan penggunaan | Rendah (berdampak pada ketersediaan) |
| Subjek citra pada dataset | Pihak yang citranya digunakan | Perlindungan privasi dan penggunaan sesuai lisensi dataset | Rendah (berdampak pada etika) |
| Unit penyedia infrastruktur komputasi [ISI DATA] | Penyedia GPU/server pelatihan dan hosting | Penggunaan sumber daya sesuai kebijakan | Sedang |

---

## 5. Functional Requirements

**Prioritas:** M = *Must have*, S = *Should have*, C = *Could have*.

### 5.1 Pengelolaan Data dan Eksperimen

| ID | Kebutuhan | Deskripsi | Prioritas | Terkait |
|---|---|---|---|---|
| FR-01 | Pengambilan dataset | Sistem dapat memuat dataset *shravya11/human-faces-dataset* dari repositori Hugging Face dan memverifikasi jumlah citra per kelas terhadap komposisi yang tercatat. | M | BO-01, IS-02 |
| FR-02 | Audit validitas dataset | Sistem dapat memeriksa duplikasi citra, label, dan potensi kebocoran data antar-subset, serta mencatat hasilnya. | M | BO-05, IS-10 |
| FR-03 | Pembagian data terstratifikasi | Sistem dapat membagi data menjadi pelatihan, validasi, dan pengujian dengan rasio 70:15:15 secara terstratifikasi dan dapat direproduksi (*seed* tercatat). | M | IS-03 |
| FR-04 | Prapemrosesan citra | Sistem menerapkan penyesuaian ukuran, konversi ke tensor, dan normalisasi yang sama untuk kedua model sesuai konfigurasi yang terdokumentasi. | M | IS-04 |

### 5.2 Pelatihan dan Evaluasi Model

| ID | Kebutuhan | Deskripsi | Prioritas | Terkait |
|---|---|---|---|---|
| FR-05 | Konfigurasi ResNet-50 | Sistem memuat ResNet-50 berbobot pralatih ImageNet, membekukan *backbone*, dan mengganti *classifier head* untuk dua kelas. | M | BO-01 |
| FR-06 | Konfigurasi ViT-Base/16 | Sistem memuat ViT-Base/16 berbobot pralatih ImageNet, membekukan *backbone*, dan mengganti *classifier head* untuk dua kelas. | M | BO-01 |
| FR-07 | Pelatihan seragam | Sistem melatih kedua model dengan data, hiperparameter yang dapat dibandingkan, dan kriteria penghentian yang sama, serta mencatat kurva *loss* dan akurasi. | M | BO-01 |
| FR-08 | Penyimpanan dan pemuatan model | Sistem menyimpan model terlatih beserta konfigurasinya dan dapat memuatnya kembali untuk inferensi. | M | BO-01 |
| FR-09 | Evaluasi klasifikasi | Sistem menghitung *accuracy*, *precision*, *recall*, *specificity*, *F1-score*, *Macro-F1*, dan *ROC-AUC* pada data uji untuk kedua model. | M | BO-02 |
| FR-10 | *Confusion matrix* | Sistem menghasilkan dan menampilkan *confusion matrix* untuk tiap model. | M | BO-02 |
| FR-11 | Pengukuran waktu inferensi | Sistem mengukur waktu inferensi per citra dengan prosedur yang sama (termasuk perlakuan pemanasan dan pengulangan sesuai konfigurasi) dan melaporkan statistik ringkasnya. | M | BO-03 |
| FR-12 | Laporan perbandingan | Sistem menyajikan tabel perbandingan kedua model untuk seluruh metrik dan waktu inferensi dalam satu keluaran terstruktur (tabel dan grafik). | S | BO-02, BO-03 |

### 5.3 Prototipe Web FaceCheck AI

| ID | Kebutuhan | Deskripsi | Prioritas | Terkait |
|---|---|---|---|---|
| FR-13 | Unggah citra | Pengguna dapat mengunggah citra wajah dengan format yang didukung ([ISI DATA]) dan batas ukuran berkas ([ISI DATA]). | M | BO-04 |
| FR-14 | Validasi masukan | Sistem menolak berkas dengan format, ukuran, atau isi yang tidak valid, dan menampilkan pesan kesalahan yang jelas. | M | BO-04 |
| FR-15 | Deteksi dan pemotongan wajah | Sistem mendeteksi wajah dengan Haar Cascade dan memotong area wajah sebelum klasifikasi; bila tidak ada wajah terdeteksi, sistem memberi tahu pengguna. | M | IS-08 |
| FR-16 | Klasifikasi oleh dua model | Sistem menjalankan ResNet-50 dan ViT-Base/16 pada citra yang sama dan menampilkan label prediksi serta skor kepercayaan dari masing-masing model. | M | BO-04 |
| FR-17 | Pemilihan model | Pengguna dapat memilih menjalankan salah satu model atau keduanya. | S | BO-04 |
| FR-18 | Penayangan waktu inferensi | Sistem menampilkan waktu inferensi tiap model pada setiap permintaan. | S | BO-03 |
| FR-19 | Penjelasan batas penggunaan | Sistem menampilkan pernyataan bahwa hasil bersifat probabilistik, terbatas pada domain dataset, dan bukan bukti forensik atau verifikasi identitas. | M | OS-01, OS-02 |
| FR-20 | Penanganan kegagalan | Sistem menampilkan pesan yang informatif dan tidak menghentikan aplikasi ketika terjadi kegagalan pemuatan model atau inferensi. | M | NFR-REL |
| FR-21 | Pembersihan citra unggahan | Sistem tidak menyimpan citra unggahan secara permanen dan menghapus berkas sementara setelah pemrosesan selesai. | M | NFR-SEC |
| FR-22 | Berkas log eksperimen | Sistem mencatat log proses pelatihan, evaluasi, dan inferensi untuk keperluan penelusuran. | C | SM-10 |

---

## 6. Non-Functional Requirements

### 6.1 Performance

| ID | Kebutuhan | Ukuran | Target |
|---|---|---|---|
| NFR-PERF-01 | Waktu inferensi model per citra pada perangkat uji | Rata-rata dan simpangan baku | [ISI DATA] |
| NFR-PERF-02 | Waktu respons prototipe dari unggah hingga hasil tampil | Waktu total per permintaan | [ISI DATA] |
| NFR-PERF-03 | Penggunaan memori saat inferensi kedua model | Memori puncak | [ISI DATA] |
| NFR-PERF-04 | Waktu pelatihan tiap model | Durasi total pada perangkat pelatihan | [ISI DATA] (dicatat, tanpa target keras) |

### 6.2 Security & Privacy

| ID | Kebutuhan | Keterangan |
|---|---|---|
| NFR-SEC-01 | Citra unggahan tidak disimpan permanen dan tidak digunakan untuk pelatihan ulang. | Mengacu FR-21 dan OS-09. |
| NFR-SEC-02 | Validasi tipe dan ukuran berkas unggahan untuk mencegah unggahan berbahaya. | Mengacu FR-14. |
| NFR-SEC-03 | Komunikasi prototipe memakai HTTPS bila di-*hosting* di luar lingkungan lokal. | Kebijakan *hosting*: [ISI DATA]. |
| NFR-SEC-04 | Kepatuhan pada lisensi dataset dan bobot pralatih. | Lisensi diverifikasi dan dicantumkan pada laporan. |
| NFR-SEC-05 | Tidak ada fitur pengenalan identitas atau penyimpanan data biometrik. | Mengacu OS-01. |

### 6.3 Usability

| ID | Kebutuhan | Ukuran | Target |
|---|---|---|---|
| NFR-USA-01 | Antarmuka sederhana: pengguna dapat mengunggah citra dan membaca hasil tanpa instruksi tambahan. | Penyelesaian tugas utama oleh penguji | [ISI DATA] |
| NFR-USA-02 | Hasil klasifikasi dan skor kepercayaan mudah dipahami, disertai penjelasan keterbatasan. | Penilaian kejelasan oleh penguji | [ISI DATA] |
| NFR-USA-03 | Antarmuka dapat diakses pada peramban modern dan responsif pada layar desktop. | Daftar peramban yang diuji | [ISI DATA] |
| NFR-USA-04 | Bahasa antarmuka konsisten (Bahasa Indonesia). | Pemeriksaan konsistensi teks | 100% teks utama |

### 6.4 Reliability & Reproducibility

| ID | Kebutuhan | Keterangan |
|---|---|---|
| NFR-REL-01 | Hasil eksperimen dapat direproduksi: *seed*, versi pustaka, dan konfigurasi tercatat. | Mengacu SM-10. |
| NFR-REL-02 | Model yang sama menghasilkan keluaran yang konsisten untuk citra yang sama pada konfigurasi yang sama. | Uji konsistensi: [ISI DATA]. |
| NFR-REL-03 | Prototipe menangani masukan tak valid tanpa *crash*. | Mengacu FR-14 dan FR-20. |
| NFR-REL-04 | Cadangan model terlatih dan kode sumber disimpan pada media terpisah. | Lokasi cadangan: [ISI DATA]. |

---

## 7. Business Process As-Is vs To-Be

### 7.1 As-Is

Pemeriksaan keaslian citra wajah di lingkungan akademik umumnya dilakukan dengan pengamatan visual atau dengan menjalankan model secara terpisah tanpa protokol perbandingan yang seragam.

| Langkah | Aktivitas | Pelaku | Kelemahan |
|---|---|---|---|
| 1 | Pengguna menerima citra wajah yang keasliannya diragukan | Mahasiswa / peneliti | Tidak ada acuan objektif |
| 2 | Pengamatan visual manual | Mahasiswa / peneliti | Subjektif; wajah sintetis sulit dibedakan |
| 3 | (Opsional) Menjalankan satu model secara mandiri | Mahasiswa / peneliti | Perbandingan antarmodel tidak seragam dan tanpa data waktu inferensi |
| 4 | Keputusan berdasarkan kesan atau satu model | Mahasiswa / peneliti | Tidak terukur, tidak terdokumentasi |

### 7.2 To-Be

| Langkah | Aktivitas | Pelaku / Sistem | Perbaikan |
|---|---|---|---|
| 1 | Pengguna mengunggah citra wajah ke FaceCheck AI | Pengguna | Antarmuka tunggal |
| 2 | Validasi berkas (format dan ukuran) | Sistem | Mencegah masukan tidak valid |
| 3 | Deteksi dan pemotongan wajah (Haar Cascade) | Sistem | Masukan model seragam |
| 4 | Prapemrosesan identik untuk kedua model | Sistem | Perbandingan yang sebanding |
| 5 | Inferensi ResNet-50 dan ViT-Base/16 | Sistem | Dua model pada citra yang sama |
| 6 | Penayangan label, skor kepercayaan, dan waktu inferensi | Sistem | Hasil terukur dan terbandingkan |
| 7 | Penayangan pernyataan batas penggunaan | Sistem | Mencegah penafsiran berlebihan |
| 8 | Pengguna menafsirkan hasil sebagai indikasi awal | Pengguna | Keputusan berbasis data dengan kesadaran keterbatasan |

### 7.3 Proses Penelitian (Pendukung)

```
Pengambilan dataset → Audit validitas → Pembagian 70:15:15 → Prapemrosesan
→ Pelatihan ResNet-50 dan ViT-Base/16 → Evaluasi metrik → Pengukuran waktu inferensi
→ Integrasi ke prototipe web → Pengujian black-box → Analisis dan pelaporan
```

---

## 8. Assumptions & Constraints

### 8.1 Assumptions

| ID | Asumsi |
|---|---|
| AS-01 | Dataset Hugging Face tetap tersedia dan label kelas pada dataset sesuai dengan deskripsinya. |
| AS-02 | Bobot pralatih ImageNet untuk ResNet-50 dan ViT-Base/16 tersedia dan dapat digunakan sesuai lisensi. |
| AS-03 | Perangkat komputasi ([ISI DATA]) cukup untuk pelatihan dan inferensi kedua model. |
| AS-04 | Pembimbing menyetujui target metrik dan tenggat sebelum pengujian. |
| AS-05 | Pengguna prototipe mengunggah citra wajah frontal yang cukup jelas agar terdeteksi Haar Cascade. |
| AS-06 | Hasil pada dataset ini dapat memberi gambaran awal, tetapi tidak otomatis berlaku untuk generator lain. |

### 8.2 Constraints

| ID | Batasan | Dampak |
|---|---|---|
| CN-01 | Dataset hanya mencakup *Real* dari CelebA dan *AI-Generated* dari StyleGAN/GAN. | Hasil tidak mewakili model difusi atau generator lain. |
| CN-02 | *Backbone* dibekukan; hanya *classifier head* yang dilatih. | Kinerja dapat berbeda bila dilakukan *fine-tuning* penuh. |
| CN-03 | Waktu penelitian dibatasi oleh jadwal TA ([ISI DATA]). | Memengaruhi kedalaman eksperimen. |
| CN-04 | Sumber daya komputasi terbatas ([ISI DATA]). | Memengaruhi ukuran *batch* dan jumlah *epoch*. |
| CN-05 | Pengukuran waktu inferensi berlaku pada satu perangkat uji. | Hasil tidak dapat digeneralisasi ke perangkat lain. |
| CN-06 | Telaah literatur belum berupa SLR final. | Celah penelitian bersifat sementara. |
| CN-07 | Haar Cascade sensitif terhadap pose dan pencahayaan. | Sebagian citra mungkin tidak terdeteksi wajahnya. |
| CN-08 | Ketentuan etika, privasi, dan lisensi dataset. | Membatasi penggunaan dan penyimpanan citra. |

---

## 9. Risk Register

Skala: **Probabilitas** dan **Dampak** = Rendah / Sedang / Tinggi (penilaian awal oleh penulis, perlu divalidasi bersama pembimbing).

| ID | Risiko | Probabilitas | Dampak | Mitigasi |
|---|---|---|---|---|
| R-01 | Kebocoran data atau duplikasi antar-subset membuat metrik terlalu optimistis | Sedang | Tinggi | Audit duplikasi (FR-02); pembagian terstratifikasi dengan *seed* tercatat; laporkan hasil audit |
| R-02 | Dataset hanya mencakup StyleGAN/GAN sehingga generalisasi terbatas | Tinggi | Tinggi | Nyatakan keterbatasan secara eksplisit; hindari klaim umum; usulkan dataset tambahan pada saran |
| R-03 | Model mempelajari artefak dataset (misalnya perbedaan sumber, resolusi, atau pemrosesan) dan bukan ciri wajah sintetis | Sedang | Tinggi | Prapemrosesan seragam; analisis kesalahan; pembahasan ancaman validitas |
| R-04 | Ketidakseimbangan kelas (5.000 vs 4.630) memengaruhi metrik | Rendah | Sedang | Laporkan *Macro-F1*, *specificity*, dan *confusion matrix*; pembagian terstratifikasi |
| R-05 | Keterbatasan GPU atau memori menghambat pelatihan ViT-Base/16 | Sedang | Sedang | Atur ukuran *batch*; gunakan *backbone* beku; siapkan alternatif komputasi ([ISI DATA]) |
| R-06 | Pelatihan tidak sebanding antarmodel (hiperparameter berbeda) sehingga perbandingan bias | Sedang | Tinggi | Konfigurasi seragam terdokumentasi (FR-07); catat setiap penyimpangan beserta alasannya |
| R-07 | Wajah tidak terdeteksi Haar Cascade pada citra unggahan | Tinggi | Sedang | Pesan jelas ke pengguna (FR-15); petunjuk kualitas citra; opsi detektor alternatif pada saran |
| R-08 | Pengguna menafsirkan hasil sebagai bukti mutlak atau alat verifikasi identitas | Sedang | Tinggi | Pernyataan batas penggunaan (FR-19); skor kepercayaan ditampilkan |
| R-09 | Pelanggaran privasi akibat penyimpanan citra unggahan | Rendah | Tinggi | Tidak menyimpan citra permanen (FR-21); tidak ada pengenalan identitas |
| R-10 | Ketidaksesuaian lisensi dataset atau bobot pralatih | Rendah | Sedang | Verifikasi dan cantumkan lisensi (NFR-SEC-04) |
| R-11 | Dataset atau repositori model tidak lagi tersedia | Rendah | Sedang | Simpan salinan lokal dan catat versi dataset |
| R-12 | Hasil tidak dapat direproduksi | Sedang | Sedang | Catat *seed*, versi pustaka, dan konfigurasi (NFR-REL-01) |
| R-13 | Jadwal TA tidak terpenuhi | Sedang | Tinggi | Prioritaskan kebutuhan *Must have*; susun jadwal dengan pembimbing ([ISI DATA]) |
| R-14 | Hasil model tidak berbeda bermakna atau perbedaan kecil sehingga kesimpulan lemah | Sedang | Sedang | Laporkan apa adanya; lakukan analisis kesalahan; hindari asumsi keunggulan awal |

---

## 10. Acceptance Criteria

### 10.1 Kriteria Penerimaan Penelitian dan Model

| ID | Kriteria | Cara Verifikasi | Status |
|---|---|---|---|
| AC-01 | Dataset termuat dan komposisi per kelas sesuai catatan (5.000 *Real*; 4.630 *AI-Generated*) | Skrip verifikasi jumlah | [ISI DATA] |
| AC-02 | Pembagian data 70:15:15 terstratifikasi tercatat beserta *seed* | Tinjauan konfigurasi | [ISI DATA] |
| AC-03 | Hasil audit duplikasi dan potensi kebocoran data terdokumentasi | Tinjauan laporan audit | [ISI DATA] |
| AC-04 | ResNet-50 dan ViT-Base/16 terlatih dengan konfigurasi seragam dan dapat dimuat ulang | Uji pemuatan model | [ISI DATA] |
| AC-05 | Seluruh metrik pada SM-01 sampai SM-07 dilaporkan untuk kedua model pada data uji | Tinjauan tabel hasil | [ISI DATA] |
| AC-06 | Waktu inferensi kedua model diukur dengan prosedur dan perangkat yang sama | Tinjauan prosedur dan hasil | [ISI DATA] |
| AC-07 | Kesimpulan perbandingan didukung data hasil pengujian, disertai keterbatasan dataset | Tinjauan pembimbing | [ISI DATA] |
| AC-08 | Seluruh konfigurasi eksperimen dapat ditelusuri dan direproduksi | Percobaan ulang pada sampel | [ISI DATA] |

### 10.2 Kriteria Penerimaan Prototipe

| ID | Kriteria | FR Terkait | Skenario Uji | Status |
|---|---|---|---|---|
| AC-09 | Unggah citra valid menghasilkan label dan skor dari kedua model | FR-13, FR-16 | Unggah citra *Real* dan *AI-Generated* | [ISI DATA] |
| AC-10 | Berkas tidak valid (format, ukuran) ditolak dengan pesan jelas | FR-14 | Unggah berkas non-citra dan berkas melebihi batas | [ISI DATA] |
| AC-11 | Citra tanpa wajah menghasilkan pemberitahuan, tanpa kegagalan aplikasi | FR-15, FR-20 | Unggah citra tanpa wajah | [ISI DATA] |
| AC-12 | Pernyataan batas penggunaan tampil pada halaman hasil | FR-19 | Inspeksi antarmuka | [ISI DATA] |
| AC-13 | Waktu inferensi tampil untuk tiap model | FR-18 | Inspeksi hasil | [ISI DATA] |
| AC-14 | Citra unggahan tidak tersimpan permanen setelah pemrosesan | FR-21 | Pemeriksaan direktori penyimpanan | [ISI DATA] |
| AC-15 | Seluruh kebutuhan berprioritas *Must have* berstatus lulus pada pengujian *black-box* | FR-01 s.d. FR-16, FR-19 s.d. FR-21 | Lembar pengujian | [ISI DATA] |

### 10.3 Persetujuan

| Peran | Nama | Tanda Tangan | Tanggal |
|---|---|---|---|
| Penyusun / Mahasiswa | [ISI DATA] | | |
| Dosen Pembimbing | [ISI DATA] | | |
| Ketua Program Studi (bila diperlukan) | [ISI DATA] | | |

---

## Lampiran A: Matriks Keterlacakan Ringkas

| Tujuan Bisnis | Kebutuhan Fungsional | Kebutuhan Non-Fungsional | Kriteria Penerimaan |
|---|---|---|---|
| BO-01 | FR-01 s.d. FR-08 | NFR-REL-01, NFR-PERF-04 | AC-01 s.d. AC-04 |
| BO-02 | FR-09, FR-10, FR-12 | NFR-REL-01 | AC-05, AC-07 |
| BO-03 | FR-11, FR-12, FR-18 | NFR-PERF-01 s.d. NFR-PERF-03 | AC-06, AC-13 |
| BO-04 | FR-13 s.d. FR-21 | NFR-SEC, NFR-USA, NFR-REL-03 | AC-09 s.d. AC-15 |
| BO-05 | FR-02 | NFR-SEC-04 | AC-03, AC-07 |

## Lampiran B: Daftar Istilah

| Istilah | Penjelasan |
|---|---|
| Generative AI | Model yang menghasilkan data baru, termasuk citra wajah sintetis |
| CNN | *Convolutional Neural Network*, jaringan yang mempelajari pola visual melalui operasi konvolusi |
| ViT | *Vision Transformer*, model yang membagi citra menjadi *patch* dan memakai mekanisme *attention* |
| Transfer learning | Pemanfaatan bobot pralatih untuk tugas baru |
| Haar Cascade | Metode deteksi objek (wajah) berbasis fitur Haar |
| *Black-box testing* | Pengujian fungsi tanpa melihat struktur internal kode |
