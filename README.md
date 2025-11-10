1. Giải thích Các Bước Thực Hiện
1.1 Cài đặt và Nhập Thư viện
Bước đầu tiên là tải dữ liệu và cài đặt các thư viện cần thiết:

datasets: Tải dataset từ Hugging Face Hub

scikit-learn: Machine learning với các model Logistic Regression, TF-IDF

pyspark: Xử lý dữ liệu lớn với Spark ML

numpy: Xử lý ma trận và vector

re: Tokenization với biểu thức chính quy

1.2 Xây dựng lớp TextClassifier
Lớp chính cho phân loại văn bản với các phương thức:

Phương thức __init__: Khởi tạo vectorizer (TF-IDF hoặc CountVectorizer) và model Logistic Regression

Phương thức fit: Huấn luyện model trên dữ liệu văn bản và nhãn

Phương thức predict: Dự đoán nhãn cho văn bản mới

Phương thức evaluate: Đánh giá model với các metrics accuracy, precision, recall, f1-score

1.3 Xây dựng lớp RegexTokenizer
Tokenizer đơn giản sử dụng biểu thức chính quy:

Chuyển văn bản thành chữ thường

Tách từ bằng pattern \b\w+\b

Trả về danh sách tokens

1.4 Thực hiện với Train/Test Split
Chia dữ liệu 80% train, 20% test

Đánh giá hiệu suất trên cả tập train và test

Kiểm tra khả năng tổng quát hóa của model

1.5 Triển khai với PySpark
Xây dựng pipeline xử lý dữ liệu lớn:

Tokenizer: Tách văn bản thành từ

StopWordsRemover: Loại bỏ stop words

HashingTF: Chuyển đổi thành vector đặc trưng

IDF: Tính trọng số nghịch đảo tần số văn bản

LogisticRegression: Model phân loại

1.6 Cải tiến và So sánh Model
AdvancedTextPreprocessor: Tiền xử lý nâng cao (xóa URL, ký tự đặc biệt)

ImprovedTextClassifier: Hỗ trợ nhiều model (Naive Bayes, Random Forest, Gradient Boosting)

SimpleWord2VecClassifier: Sử dụng word embeddings

2. Hướng Dẫn Chạy Code
Cách 1: Chạy trên Google Colab
Mở Google Colab (colab.research.google.com)

Tạo notebook mới

Copy toàn bộ code vào các ô

Chạy lần lượt các ô bằng Shift+Enter

Đợi tải dataset từ Hugging Face

Xem kết quả output

Cách 2: Chạy Trên Máy Cục bộ
bash
# 1. Tạo virtual environment
python -m venv sentiment_env
source sentiment_env/bin/activate  # Trên Windows: sentiment_env\Scripts\activate

# 2. Cài đặt dependencies
pip install datasets scikit-learn pyspark numpy

# 3. Chạy toàn bộ notebook
jupyter notebook lab4_TranVanLam_22001268.ipynb

# Hoặc chạy từng phần trong file Python
Cách 3: Chạy Từng Phần
Nếu muốn test từng phần riêng biệt:

python
# Test TextClassifier cơ bản
from sklearn.feature_extraction.text import TfidfVectorizer
classifier = TextClassifier(TfidfVectorizer(max_features=100))
classifier.fit(texts_train, labels_train)
predictions = classifier.predict(texts_test)

# Test với PySpark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
# ... tiếp tục với Spark pipeline
Lưu Ý Quan Trọng:

Cần kết nối internet để tải dataset từ Hugging Face

PySpark yêu cầu Java 8 hoặc 11

Dataset có kích thước vừa phải (~1MB)

3. Phân Tích Kết Quả
3.1 Kết quả với TextClassifier Cơ bản
Trên tập train:

Accuracy: 1.0000 - Model học hoàn hảo tập train

Precision: 1.0000, Recall: 1.0000, F1: 1.0000

Trên tập test:

Accuracy: 0.7500 - Model tổng quát khá tốt

Precision: 0.6667, Recall: 1.0000, F1: 0.8000

Nhận xét:

Model bị overfitting nhẹ trên tập train

Recall cao cho thấy model phát hiện tốt positive class

Precision thấp hơn cho thấy có false positives

3.2 Kết quả với PySpark
Trên tập test:

Accuracy: 0.7143

F1 Score: 0.7143

Weighted Precision: 0.7143

Weighted Recall: 0.7143

Confusion Matrix:

text
TN=2, FP=1, FN=1, TP=3
Phân tích:

Hiệu suất tương đương với scikit-learn

Có thể xử lý được dataset lớn hơn

Pipeline hoàn chỉnh từ tiền xử lý đến dự đoán

3.3 So sánh các Model Architecture
Kết quả so sánh:

Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	0.5000	0.5000	0.6667	0.5714
Naive Bayes	0.5000	0.0000	0.0000	0.0000
Random Forest	0.5000	0.5000	0.6667	0.5714
Gradient Boosting	0.5000	0.5000	0.6667	0.5714
Nhận xét:

Logistic Regression cho kết quả ổn định nhất

Naive Bayes không phù hợp với dataset này

Ensemble methods không cải thiện đáng kể

3.4 Word2Vec đơn giản
Accuracy: 0.6667

F1 Score: 0.6667

Precision: 0.6667, Recall: 0.6667

Đánh giá:

Hiệu suất khá tốt với embeddings ngẫu nhiên

Có tiềm năng cải thiện với pre-trained embeddings

4. So sánh Các Phương pháp
4.1 Traditional ML vs Deep Learning
Traditional ML (TF-IDF + Logistic Regression):

Ưu điểm: Nhanh, ít dữ liệu, dễ giải thích

Nhược điểm: Không nắm bắt ngữ nghĩa sâu

Word Embeddings:

Ưu điểm: Hiểu ngữ nghĩa từ, xử lý từ đồng nghĩa

Nhược điểm: Cần nhiều dữ liệu, tính toán phức tạp

4.2 Single Machine vs Distributed Computing
Scikit-learn (Single Machine):

Phù hợp dataset nhỏ và vừa

Deployment đơn giản

Hiệu suất tốt với dữ liệu trong memory

PySpark (Distributed):

Xử lý dataset cực lớn

Khả năng scale horizontal

Phức tạp hơn trong deployment

4.3 Khi Nào Dùng Phương pháp Nào?
Dataset nhỏ (<100K samples): Scikit-learn
Dataset lớn (>1M samples): PySpark
Yêu cầu real-time: Traditional ML
Cần độ chính xác cao: Word Embeddings + Deep Learning
Cần giải thích model: Logistic Regression với TF-IDF

5. Khó Khăn và Giải Pháp
5.1 Vấn đề: Overfitting
Biểu hiện: Accuracy train = 1.0000, test = 0.7500

Giải pháp:

Tăng regularization parameter

Giảm số features trong TF-IDF

Thêm nhiều dữ liệu training

Sử dụng cross-validation

python
# Giải pháp: Tăng regularization
LogisticRegression(C=0.1, solver='liblinear')  # Giảm C để tăng regularization
5.2 Vấn đề: Dữ liệu không cân bằng
Giải pháp:

Class weights trong Logistic Regression

Oversampling/Undersampling

Sử dụng F1-score thay vì accuracy

python
# Sử dụng class weights
LogisticRegression(class_weight='balanced')
5.3 Vấn đề: Hiệu suất thấp trên test set
Nguyên nhân: Dataset quá nhỏ (chỉ 30 samples)

Giải pháp:

Thu thập thêm dữ liệu

Sử dụng data augmentation

Transfer learning với pre-trained models

5.4 Vấn đề: Xử lý từ OOV trong Word2Vec
Giải pháp:

Sử dụng pre-trained Word2Vec/GloVe

Áp dụng FastText cho subword information

Character-level embeddings

6. Tài liệu Tham Khảo
6.1 Các Tài liệu Chính
Scikit-learn Documentation - https://scikit-learn.org/
Tài liệu chính thức về các algorithms ML

PySpark ML Documentation - https://spark.apache.org/docs/latest/ml-guide.html
Hướng dẫn sử dụng ML library trong Spark

Hugging Face Datasets - https://huggingface.co/docs/datasets/
Documentation về datasets library

6.2 Bài báo Khoa học
"A Comparative Study of Text Classification Algorithms" - Sun et al.
So sánh các algorithms phân loại văn bản

"Distributed Representations of Words and Phrases" - Mikolov et al.
Giới thiệu Word2Vec và các cải tiến

"TF-IDF and its Applications" - Manning et al.
Lý thuyết và ứng dụng của TF-IDF

6.3 Sách và Tutorials
"Speech and Language Processing" - Jurafsky & Martin
Sách toàn diện về xử lý ngôn ngữ tự nhiên

"Applied Text Analysis with Python" - Bengfort et al.
Hướng dẫn thực hành phân tích văn bản

Spark MLlib Tutorials - https://spark.apache.org/docs/latest/ml-guide.html
Tutorials chính thức từ Apache Spark

6.4 Dataset References
Twitter Financial News Sentiment Dataset - https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
Dataset chính được sử dụng trong project

IMDB Movie Reviews - Alternative dataset cho sentiment analysis
Dataset lớn hơn để testing scalability

