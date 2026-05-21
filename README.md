# HairFit AI Face Shape Analyzer Service (FastAPI)

Dịch vụ AI phân tích khuôn mặt để xác định dáng khuôn mặt (`ROUND`, `OVAL`, `SQUARE`, `HEART`) viết bằng Python FastAPI.

## Hướng dẫn cài đặt và khởi chạy

### 1. Yêu cầu hệ thống
* Đã cài đặt **Python 3.8** trở lên.

### 2. Cài đặt thư viện
Di chuyển vào thư mục `ai-service` và cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 3. Chạy dịch vụ
Chạy FastAPI server bằng lệnh:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Dịch vụ sẽ hoạt động tại địa chỉ: `http://localhost:8000`

### 4. Kiểm tra API
Bạn có thể truy cập trang tài liệu Swagger UI tự động tích hợp tại:
`http://localhost:8000/docs`
để test trực tiếp endpoint `/analyze-face`.
