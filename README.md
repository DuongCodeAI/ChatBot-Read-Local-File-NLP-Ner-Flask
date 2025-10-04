🤖 Dự án Chatbot đọc file nội bộ công ty (NLP + NER)

Chatbot được xây dựng để **trả lời tự động các câu hỏi FAQ** từ dữ liệu nội bộ của công ty.
**Lưu ý** 
1. File dữ liệu  duLieuCongTy.json được tạo **tự động hoàn toàn** bằng AI và không liên quan đến bất kì cá nhân, tổ chức nào
2. Các file data_chuDe.json đã xin phép ViettelPost trước khi scrap dữ liệu câu hỏi FAQ
Dự án ứng dụng các kỹ thuật **NLP, NER, semantic search và rule-based matching** để tối ưu hóa khả năng trả lời chính xác.

---

## 🚀 Tính năng chính
- Dùng NLP và các công nghệ package hỗ trợ như **underthesea, difflib** để hiểu input câu hỏi + render câu trả lời 
- Tích hợp API **Flask**
- Đọc dữ liệu từ các file **JSON nội bộ**.
- Hỗ trợ **tìm kiếm chính xác** và **tìm kiếm ngữ nghĩa**.
- Tích hợp **Named Entity Recognition (NER)** để hiểu ngữ cảnh.
- Có cơ chế **fallback** khi không tìm thấy câu trả lời phù hợp.
- Ghi **log** toàn bộ **pipeline** để dễ dàng theo dõi và debug.


---

## 🏗️ Kiến trúc hệ thống
Pipeline xử lý câu hỏi trong chatbot:

+---------------------+
| User query (input) |
+---------------------+
│
▼
+---------------------+
| Exact question check|
| - So sánh query với |
| tất cả question |
| - Nếu có duy nhất |
| match → trả answer|
| - Nếu không → tiếp |
| Data Loading |
+---------------------+
│
▼
+---------------------+
| Data Loading |
| - Load JSON files |
| - Index documents |
| - Error handling |
| - Logging |
| - Caching vectors |
+---------------------+
│
▼
+---------------------+
| Preprocessing |
| - Tokenize, lowercase|
| - Remove stop words |
| - Lemmatize / Stem |
| - Logging |
+---------------------+
│
▼
| NER extraction |
| - Detect entity |
| - Logging |
+---------------------+
│
▼
| Semantic search |
| - TF-IDF / Embedding|
| - Cosine similarity |
| - Entity-boosted |
| - Logging |
+---------------------+
│
▼
| Rule-based matching |
| - Entity-aware |
| - Threshold check |
| - Fallback response |
| - Logging |
+---------------------+
│
▼
| Reranking / Fusion |
| - Kết hợp kết quả |
| - Chọn câu trả lời |
| - Logging |
+---------------------+
│
▼
| Postprocessing |
| - Format output |
| - Handle no-match |
+---------------------+
│
▼
+---------------------+
| Return answer |
+---------------------+

### Giao diện khởi đầu của hệ thống 
![giao diện ban đầu ](images/img1.jpg)

###  Khi hỏi câu hỏi giống nhau, chatbot vẫn trả lời được 
![](images/img2.jpg)


###  Hỏi những câu hỏi tổng quát
![Ảnh 1](images/img3.jpg)
![Ảnh 2](images/img4.jpg)

###  Cơ chế fellback khi không phải chủ đề liên quan
![Fellback](images/img5.jpg)

---

## ⚙️ Cài đặt & chạy thử

1. Clone repo
```bash
git clone https://github.com/<username>/<repo>.git
cd du_an_chatBot_noi_bo
 2. Tạo môi trường ảo & cài dependency
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
3. Chạy ứng dụng Flask
python app.py web


