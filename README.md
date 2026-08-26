# Recruit Restaurant Visitor Forecasting

Dự án phân tích và dự đoán số lượng khách ghé thăm nhà hàng theo ngày, dựa trên bộ dữ liệu của cuộc thi Kaggle [**Recruit Restaurant Visitor Forecasting**](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting).

## Bài toán

Recruit Holdings sở hữu Hot Pepper Gourmet (dịch vụ đánh giá nhà hàng), AirREGI (hệ thống POS cho nhà hàng) và Restaurant Board (phần mềm quản lý đặt chỗ). Bài toán đặt ra là dùng dữ liệu đặt chỗ và lịch sử ghé thăm để **dự đoán tổng số khách ghé thăm một nhà hàng vào các ngày trong tương lai**, giúp nhà hàng chủ động hơn trong việc chuẩn bị nguyên liệu và bố trí nhân sự.

Dữ liệu đến từ hai hệ thống:
- **AirREGI (air)**: hệ thống POS/đặt chỗ, tương tự Square.
- **HOT PEPPER Gourmet (hpg)**: nền tảng tìm kiếm & đặt chỗ nhà hàng, tương tự Yelp.

## Cấu trúc repo

| File | Mô tả |
|---|---|
| `report.ipynb` | Notebook chính: khám phá dữ liệu (EDA), xử lý đặc trưng, huấn luyện và đánh giá mô hình dự đoán số khách |
| `kaggle.zip` | Dữ liệu gốc của cuộc thi (tải từ Kaggle) |
| `weather.zip` | Dữ liệu thời tiết bổ sung, dùng làm đặc trưng ngoại sinh (thời tiết ảnh hưởng đến lượng khách ghé nhà hàng) |
| `leaf-wise.webp` | Hình minh hoạ chiến lược tăng trưởng cây "leaf-wise" của LightGBM — mô hình chính được dùng trong notebook |

## Cài đặt

Giải nén dữ liệu trước khi chạy notebook:

```bash
unzip kaggle.zip -d data/
unzip weather.zip -d data/weather/
```

Cài các thư viện cần thiết (điều chỉnh theo import thực tế trong `report.ipynb`):

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn jupyter
```

## Cách chạy

```bash
jupyter notebook report.ipynb
```

Chạy tuần tự các cell trong notebook để tái hiện lại pipeline: nạp dữ liệu → xử lý/feature engineering (bao gồm đặc trưng thời tiết) → huấn luyện mô hình (LightGBM) → đánh giá kết quả.

## Phương pháp (tóm tắt)

- **Đặc trưng**: thông tin nhà hàng (thể loại, khu vực, vị trí), lịch sử ghé thăm, dữ liệu đặt chỗ, ngày lễ, và **dữ liệu thời tiết** theo khu vực/ngày.
- **Mô hình**: LightGBM (dựa trên hình `leaf-wise.webp` minh hoạ trong repo).
- **Đánh giá**: theo chỉ số của cuộc thi gốc — RMSLE (Root Mean Squared Logarithmic Error).

## Nguồn dữ liệu

- [Recruit Restaurant Visitor Forecasting – Kaggle](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting)
