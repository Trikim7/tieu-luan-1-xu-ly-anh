# Tiểu luận 1: Xử lý ảnh dựa trên giá trị điểm ảnh

Ứng dụng web xử lý ảnh được xây dựng bằng Streamlit cho môn Xử lý ảnh - Đại học Công nghiệp TP.HCM (IUH).

## 🎯 Mục tiêu

Thực hiện các yêu cầu của tiểu luận 1:

1. **Biến đổi cường độ** - Intensity Transformations
2. **Cân bằng lược đồ mức xám** - Histogram Equalization
3. **Ứng dụng thực tế** - Real-world Applications

## ✨ Tính năng

### 1. Biến đổi cường độ (Intensity Transformations)

- **Negative**: Âm bản của ảnh
- **Log Transform**: Biến đổi logarithm để tăng cường vùng tối
- **Gamma Correction**: Điều chỉnh gamma để cải thiện độ sáng
- **Piecewise Linear**: Biến đổi tuyến tính từng đoạn

### 2. Cân bằng lược đồ mức xám (Histogram Processing)

- **Histogram Equalization**: Cân bằng toàn cục
- **CLAHE**: Cân bằng cục bộ có giới hạn để giảm nhiễu

### 3. Ứng dụng thực tế (Applications)

- **Cải thiện biển số xe**: Tăng cường độ rõ nét cho OCR
- **Xử lý ảnh vệ tinh**: Tăng cường chi tiết địa hình
- **Cải thiện ảnh thiếu sáng**: Tăng độ sáng vùng tối
- **Khôi phục tài liệu**: Làm rõ văn bản bị mờ/ố vàng

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- pip

### Các bước cài đặt

1. **Clone repository**:

```bash
git clone https://github.com/YOUR_USERNAME/tieu-luan-1-xu-ly-anh.git
cd tieu-luan-1-xu-ly-anh
```

2. **Tạo môi trường ảo**:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Cài đặt dependencies**:

```bash
pip install -r requirements.txt
```

4. **Chạy ứng dụng**:

```bash
streamlit run app.py
```

## 📦 Dependencies

- **streamlit**: Giao diện web
- **opencv-python**: Xử lý ảnh
- **numpy**: Tính toán ma trận
- **pillow**: Thao tác với ảnh
- **matplotlib**: Vẽ biểu đồ histogram

## 📁 Cấu trúc project

```
tieu_luan_1/
├── app.py                 # Giao diện chính Streamlit
├── requirements.txt       # Dependencies
├── processing/           # Thuật toán xử lý ảnh
│   ├── intensity.py      # Biến đổi cường độ
│   ├── histogram.py      # Xử lý histogram
│   └── applications.py   # Ứng dụng thực tế
└── utils/               # Utilities
    ├── image_io.py      # I/O ảnh
    └── plot.py          # Vẽ biểu đồ
```

## 🎨 Giao diện

Ứng dụng có 3 tab chính tương ứng với 3 yêu cầu:

1. **Biến đổi cường độ**: Các phép biến đổi pixel-wise
2. **Cân bằng lược đồ**: Xử lý histogram và so sánh
3. **Ứng dụng thực tế**: Các thuật toán cho bài toán cụ thể

## 🧠 Thuật toán

### Custom Implementations

- **Histogram Equalization**: Tự implement CDF mapping
- **CLAHE**: Adaptive histogram với clip limit
- **Log Transform**: Với xử lý edge cases
- **Gamma Correction**: Công thức s = c * r^γ

### Optimizations

- Sử dụng OpenCV cho một số thuật toán để tránh artifacts
- Bilateral filtering cho khử nhiễu thông minh
- Adaptive thresholding cho nhị phân hóa
