# Obfuscated Web Attack Detection

Dự án phát hiện tấn công web SQL Injection và XSS ở mức ký tự
(`char-level`). Mô hình chính là Hybrid 1D-CNN + LSTM, kèm các baseline
CNN-only và LSTM-only để so sánh.

## Cấu Trúc Dự Án

```text
.
+-- dataset/
|   +-- SQLInjection_XSS_MixDataset.1.0.0.csv
|   +-- csic_database.csv
|   +-- obfuscated_http_dataset.csv
|   +-- obfuscated_grouped.csv
|   +-- xss_payloads_with_obfuscated.csv
|   +-- obfuscation_dataset_full.xlsx          # dataset cũ, chỉ giữ tham khảo
+-- preprocessing/
|   +-- preprocess_data.py
+-- cnn_lstm/
|   +-- CNN_LSTM.py
|   +-- CNN_LSTM.ipynb
|   +-- artifacts_cnn_lstm_by_dataset/
|   +-- artifacts_cnn_lstm_tuning/
+-- cnn_only/
|   +-- train_cnn_only.py
|   +-- cnn_only_by_dataset.ipynb
|   +-- artifacts_cnn_only_by_dataset/
+-- lstm_only/
|   +-- artifacts_lstm_only_by_dataset/
+-- analysis/
|   +-- analyze_cnn_lstm.py
|   +-- evaluate_external_obfu.py
|   +-- probe_evasion.py
|   +-- obfu_eval_outputs/
+-- experiments/
+-- webapp/
```

## Vai Trò Dataset

Hai tập benchmark chính:

```text
dataset/SQLInjection_XSS_MixDataset.1.0.0.csv
dataset/csic_database.csv
```

Tập obfuscation do nhóm tự xây dựng:

```text
dataset/obfuscated_http_dataset.csv
```

Hai tập obfuscation bên ngoài dùng để kiểm thử thêm:

```text
dataset/obfuscated_grouped.csv
dataset/xss_payloads_with_obfuscated.csv
```

File `obfuscation_dataset_full.xlsx` là dataset cũ, không còn là tập
obfuscation chính.

## Tiền Xử Lý

Toàn bộ pipeline tiền xử lý dùng chung nằm ở:

```text
preprocessing/preprocess_data.py
```

Nguyên tắc tiền xử lý:

```text
không URL decode
không HTML unescape
không chuyển lowercase
chỉ chuẩn hóa whitespace
token hóa theo từng ký tự
```

Với dữ liệu dạng HTTP, input được gom về một envelope thống nhất:

```text
[METHOD] ... [PATH] ... [QUERY] ... [BODY] ... [COOKIE] ... [CONTENT_TYPE] ... [USER_AGENT] ...
```

Với Kaggle và CSIC, dữ liệu được chia train/val/test có stratify theo nhãn.
Với `obfuscated_http_dataset.csv`, pipeline dùng trực tiếp cột `split` đã có:

```text
train
val
test
test_unseen_technique
test_unseen_seed
test_unseen_both
```

## Train CNN-LSTM

Chạy từ thư mục gốc project:

```bash
python cnn_lstm/CNN_LSTM.py
```

Chạy nhanh để kiểm thử:

```bash
python cnn_lstm/CNN_LSTM.py --sample-size 3000 --obfu-sample-size 1000 --epochs 3
```

Artifact của CNN-LSTM theo từng dataset được lưu tại:

```text
cnn_lstm/artifacts_cnn_lstm_by_dataset/
```

Model tuned 768 đang dùng cho WebApp và detector DVWA nằm tại:

```text
cnn_lstm/artifacts_cnn_lstm_tuning_768/obfu_http/final/
```

Các file cần có:

```text
best_tuned_hybrid_cnn_lstm.keras
tokenizer.pkl
metadata_and_results.json
```

## Train CNN-Only Baseline

Chạy từ thư mục gốc project:

```bash
python cnn_only/train_cnn_only.py
```

Artifact của CNN-only được lưu tại:

```text
cnn_only/artifacts_cnn_only_by_dataset/
```

CNN-only dùng cùng pipeline tiền xử lý với CNN-LSTM và đánh giá toàn bộ các
split dạng `test*`, gồm cả các split unseen obfuscation.

## Đánh Giá Tập Obfuscation Gộp

Dùng artifact đã train sẵn để kiểm thử, không train lại:

```bash
python analysis/evaluate_external_obfu.py
```

Script này tạo tập kiểm thử gộp từ:

```text
dataset/obfuscated_http_dataset.csv        # chỉ lấy các split test*
dataset/obfuscated_grouped.csv
dataset/xss_payloads_with_obfuscated.csv
```

Output được lưu tại:

```text
analysis/obfu_eval_outputs/
```

Bảng kết quả chính:

```text
analysis/obfu_eval_outputs/external_obfu_eval_results.csv
```

Tên tập kiểm thử chính:

```text
combined_obfu_all_sources
```

File sau chỉ dùng cho thí nghiệm train lại trong tương lai, không dùng làm
tập test chính cho artifact đã từng train trên `obfuscated_http_dataset.csv`:

```text
analysis/obfu_eval_outputs/combined_obfu_all_sources_full.csv
```

## Chạy Web App

Cài thư viện:

```bash
python -m pip install -r webapp/requirements.txt
```

Khởi động app:

```bash
cd webapp
python app.py
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

Web app đang load model tuned 768 ở:

```text
cnn_lstm/artifacts_cnn_lstm_tuning_768/obfu_http/final/best_tuned_hybrid_cnn_lstm.keras
cnn_lstm/artifacts_cnn_lstm_tuning_768/obfu_http/final/tokenizer.pkl
cnn_lstm/artifacts_cnn_lstm_tuning_768/obfu_http/final/metadata_and_results.json
```

## Tích hợp detector vào DVWA

DVWA nằm tại `DVWA/DVWA/DVWA`. PHP hook trong `detector/hook.php` được nạp
trước mọi trang bài tập qua `auto_prepend_file`, quét tham số GET, POST và
COOKIE, sau đó gọi API detector tại `http://host.docker.internal:8000/api/predict`.
Hook chỉ hiển thị cảnh báo để giữ nguyên mục tiêu thực nghiệm của DVWA; request
không bị chặn hoặc sửa nội dung.

Khởi động detector để container có thể truy cập từ Docker Desktop:

```powershell
$env:DETECTOR_HOST = "0.0.0.0"
.\.venv-webapp\Scripts\python.exe webapp\app.py
```

Trong terminal khác, khởi động DVWA:

```powershell
cd DVWA\DVWA\DVWA
docker compose up -d
```

DVWA được mở tại `http://127.0.0.1:4280`. Endpoint `/api/predict` trả về trường
`max_len=768`, cho phép kiểm tra trực tiếp rằng hook đang gọi đúng checkpoint.

## Ghi Chú

- Artifact sinh ra khi train được ignore bằng rule `**/artifacts*/`.
- Dataset raw và các file CSV/XLSX sinh ra được ignore bằng rule `*.csv` và
  `*.xlsx`.
- Các file CSV cũ của bước đánh giá obfuscation đã được chuyển vào:

```text
analysis/obfu_eval_outputs/legacy/
```
