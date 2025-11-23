# Dự án Phân loại Văn bản - Bài tập lớn Môn học học máy

Dự án này xây dựng và đánh giá một chuỗi các mô hình Machine Learning để giải quyết bài toán phân loại văn bản. Quy trình bao gồm từ tiền xử lý dữ liệu, so sánh các phương pháp trích xuất đặc trưng, tối ưu siêu tham số cho đến việc lựa chọn và lưu trữ mô hình tốt nhất.

## Giới thiệu bài toán

Bài toán đặt ra là phân loại các đoạn văn bản vào các danh mục (labels) đã được định sẵn.

- **Input**: Một đoạn văn bản thô.
- **Output**: Nhãn (danh mục) tương ứng của đoạn văn bản đó.

Đây là một bài toán phân loại đa lớp (multi-class classification) phổ biến trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP), có nhiều ứng dụng thực tế như phân loại email (spam/không spam), phân tích cảm xúc (tích cực/tiêu cực/trung tính), phân loại tin tức...

## Cấu trúc thư mục

```
.
├── data/
│   └── data.csv          # File chứa dữ liệu huấn luyện (cần được người dùng cung cấp)
├── models/
│   └── final_nb_model.joblib # Mô hình tốt nhất sau khi huấn luyện, sẵn sàng để sử dụng
├── training_results/
│   ├── results_baseline_count.csv
│   ├── results_tfidf_uni.csv
│   ├── results_gridsearch_nb.csv
│   ├── results_balanced_prior.csv
│   └── results_svm.csv   # Các file CSV chứa kết quả chi tiết của từng thử nghiệm
├── data_loader.py        # Module để tải và chuẩn bị dữ liệu
├── experiments.py        # Module chính chứa các hàm thực nghiệm, đánh giá mô hình
├── models.py             # Module định nghĩa các lớp mô hình (NaiveBayesModel, SVMModel)
├── run.py                # File thực thi chính của dự án
├── utils.py              # Chứa các hàm tiện ích, ví dụ: hàm tiền xử lý văn bản
└── README.md             # File này
```

## Yêu cầu cài đặt

Dự án yêu cầu Python 3.x và các thư viện sau. Bạn có thể cài đặt chúng bằng pip:

```bash
pandas 
numpy 
scikit-learn 
joblib 
nltk
```

## Cách sử dụng

1.  **Chuẩn bị dữ liệu**: Đặt file dữ liệu của bạn (ví dụ: `data.csv`) vào thư mục `data/`. File này cần có ít nhất 2 cột, một cột chứa văn bản và một cột chứa nhãn.

2.  **Chạy quy trình**: Mở terminal và chạy file `run.py`.

    ```bash
    python run.py
    ```

3.  **Kết quả**: Sau khi chạy xong:
    -   Tất cả các kết quả đánh giá mô hình sẽ được lưu dưới dạng file `.csv` trong thư mục `training_results/`.
    -   Mô hình Naive Bayes cuối cùng với các tham số tốt nhất sẽ được huấn luyện trên toàn bộ dữ liệu và lưu tại `models/final_nb_model.joblib`.

## Quy trình thực nghiệm

Dự án thực hiện một chuỗi các thử nghiệm được định nghĩa trong `experiments.py` để tìm ra mô hình hiệu quả nhất. Tất cả các thử nghiệm đều sử dụng phương pháp đánh giá chéo `StratifiedKFold` (5-fold) để đảm bảo kết quả đáng tin cậy.

1.  **Thử nghiệm 1: Baseline**
    -   **Mô hình**: Multinomial Naive Bayes.
    -   **Trích xuất đặc trưng**: `CountVectorizer` (đếm số lần xuất hiện của từ).
    -   **Mục đích**: Xây dựng một mô hình cơ sở để làm mốc so sánh.

2.  **Thử nghiệm 2: Cải tiến với TF-IDF**
    -   **Mô hình**: Multinomial Naive Bayes.
    -   **Trích xuất đặc trưng**: `TfidfVectorizer` (sử dụng unigram).
    -   **Mục đích**: Đánh giá hiệu quả của TF-IDF so với CountVectorizer.

3.  **Thử nghiệm 3: Tối ưu siêu tham số (GridSearch)**
    -   Sử dụng `GridSearchCV` để tự động tìm ra bộ tham số tốt nhất cho mô hình Naive Bayes + TF-IDF.
    -   **Các tham số được tối ưu**:
        -   `ngram_range`: (1,1), (1,2), (1,3)
        -   `alpha` (tham số làm mịn của Naive Bayes): 0.01, 0.1, 0.5, 1.0
    -   **Mục đích**: Tìm ra cấu hình mô hình cho performance cao nhất.

4.  **Thử nghiệm 4: Cân bằng lớp (Balanced Prior)**
    -   **Mô hình**: Naive Bayes với các tham số tốt nhất từ GridSearch.
    -   **Tham số**: `class_prior` được thiết lập để cân bằng (giả định xác suất tiên nghiệm của các lớp là bằng nhau).
    -   **Mục đích**: Kiểm tra xem việc xử lý mất cân bằng dữ liệu (nếu có) có cải thiện kết quả hay không.

5.  **Thử nghiệm 5: So sánh với SVM**
    -   **Mô hình**: Support Vector Machine với kernel tuyến tính (`LinearSVC`).
    -   **Trích xuất đặc trưng**: `TfidfVectorizer` với `ngram_range` tốt nhất tìm được từ GridSearch.
    -   **Mục đích**: So sánh hiệu năng của Naive Bayes với một mô hình mạnh mẽ khác là SVM.

## Kết luận

Dựa vào các file kết quả trong thư mục `training_results/` và kết quả được in ra trong màn hình console, chúng ta có thể so sánh performance và những thông tin chi tiết hơn của các mô hình để đưa ra kết luận cuối cùng về phương pháp nào là hiệu quả nhất cho bộ dữ liệu này.

Mô hình tốt nhất sau đó được huấn luyện lại trên toàn bộ tập dữ liệu và lưu lại để có thể tái sử dụng cho các tác vụ dự đoán trong tương lai.

---