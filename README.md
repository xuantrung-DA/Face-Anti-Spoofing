# 🛡️ Face Anti-Spoofing (FAS) Module
> **Sub-module cốt lõi thuộc dự án Secure Login System.**
> Phát hiện và ngăn chặn các cuộc tấn công giả mạo khuôn mặt (Presentation Attacks) bằng kiến trúc CDCN tích hợp Attention Mechanism.

<p align="center">
  <img src="[https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)" />
  <img src="[https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)" />
  <img src="[https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)" />
  <img src="[https://img.shields.io/badge/Accuracy-96.9%25-brightgreen?style=for-the-badge](https://img.shields.io/badge/Accuracy-96.9%25-brightgreen?style=for-the-badge)" />
  <img src="[https://img.shields.io/badge/ACER-0.032-blue?style=for-the-badge](https://img.shields.io/badge/ACER-0.032-blue?style=for-the-badge)" />
</p>

---

## 📖 Giới thiệu (Overview)
Đây là module Face Anti-Spoofing (FAS) độc lập, đóng vai trò như lớp khiên bảo vệ đầu tiên trước khi đưa dữ liệu vào hệ thống Face Recognition của **Secure Login System**. 

Hệ thống được thiết kế để phân biệt giữa khuôn mặt thật (Live) và các nỗ lực giả mạo (Spoofing) như ảnh in, video phát lại (replay attack), hoặc mặt nạ 3D. 

**Phương pháp tiếp cận:** Sử dụng kiến trúc **CDCN** (Central Difference Convolutional Networks) để trích xuất các đặc điểm vi mô (micro-features). Nhằm tối ưu hóa hiệu năng, dự án tiến hành thử nghiệm nghiêm ngặt giữa hai cơ chế Attention:
1. **SA (Spatial Attention):** Tập trung vào vùng không gian quan trọng trên khuôn mặt.
2. **CBAM (Convolutional Block Attention Module):** Kết hợp cả sự chú ý về kênh (Channel) và không gian (Spatial).

---

## 📂 Cấu trúc Repository (Repo Structure)
Repository này lưu trữ các module kiến trúc, script thu thập dữ liệu và weights đã được huấn luyện.

```text
📦 Face-Anti-Spoofing
 ┣ 📜 cdcnCBAM.py          # Kiến trúc mô hình CDCN tích hợp CBAM Attention
 ┣ 📜 cdcnSA.py            # Kiến trúc mô hình CDCN tích hợp Spatial Attention
 ┣ 📜 collect.py           # Script thu thập dữ liệu/test nội bộ
 ┣ 📜 cdcnpp80k.pt         # PyTorch Checkpoint (Weights) của mô hình tốt nhất
 ┗ 📜 cdcnpp_cbam.onnx     # Mô hình đã được export sang ONNX để tối ưu inference

```

---

## ⚡ Hiệu năng Mô hình (Performance Metrics)

Sau quá trình test và benchmark giữa SA và CBAM, mô hình được chọn cuối cùng đã đạt được hiệu suất vượt trội trên tập dữ liệu thử nghiệm, đáp ứng tiêu chuẩn của một hệ thống định danh thực tế:

* **Accuracy:** `96.9%`
* **ACER (Average Classification Error Rate):** `0.032`

Mô hình tốt nhất (tích hợp CBAM) đã được convert sang định dạng `ONNX` (`cdcnpp_cbam.onnx`) để đảm bảo tốc độ suy luận (inference speed) ở mức mili-giây khi deploy vào luồng Login thực tế.

---

## 🚀 Hướng dẫn sử dụng (Quick Start)

### 1. Yêu cầu môi trường (Prerequisites)

Đảm bảo bạn đã cài đặt các thư viện cần thiết:

```bash
pip install torch torchvision onnx onnxruntime opencv-python

```

### 2. Sử dụng mô hình ONNX để Inference

Do mô hình được dùng làm module tích hợp, phương pháp tối ưu nhất là load qua `onnxruntime`:

```python
import onnxruntime as ort

# Load the optimized CBAM model
ort_session = ort.InferenceSession("cdcnpp_cbam.onnx")

# Run inference (Example input shape requirement depends on the CDCN config)
# outputs = ort_session.run(None, {"input": processed_face_tensor})

```

---
