# KẾ HOẠCH KHÓA LUẬN TỐT NGHIỆP — PHIÊN BẢN MASTER 6.0

# Lan truyền rủi ro từ thị trường tài chính toàn cầu đến thị trường chứng khoán Việt Nam:

# Tiếp cận bằng Học máy Nhân quả đa tần số (Wavelet-based Neural Causal Discovery)

---

## Thông tin tổng quan

| Thành phần          | Nội dung                                                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Tên đề tài          | Lan truyền rủi ro từ thị trường tài chính toàn cầu đến thị trường chứng khoán Việt Nam: Tiếp cận bằng Học máy Nhân quả đa tần số |
| Hướng nghiên cứu    | Financial Econometrics + Wavelet Analysis + Causal Machine Learning                                                              |
| Phương pháp lõi     | MODWT + Wavelet Coherence + Neural Granger + NOTEARS + Graph Analytics                                                           |
| Giai đoạn dữ liệu   | 01/01/2015 → 31/12/2025                                                                                                          |
| Tần suất dữ liệu    | Daily                                                                                                                            |
| Ngôn ngữ triển khai | Python                                                                                                                           |
| Framework AI        | PyTorch                                                                                                                          |
| Mục tiêu ứng dụng   | Dashboard trực quan + Telegram Early Warning Bot                                                                                 |
| Đầu ra cuối cùng    | Khóa luận hoàn chỉnh + Source code + Dashboard + Slide bảo vệ                                                                    |

---

# MỤC LỤC

1. Tổng quan nghiên cứu
2. Động cơ và tính cấp thiết
3. Mục tiêu nghiên cứu
4. Câu hỏi nghiên cứu
5. Đối tượng & phạm vi nghiên cứu
6. Đóng góp học thuật và thực tiễn
7. Cơ sở lý thuyết
8. Tổng quan nghiên cứu thực nghiệm
9. Khoảng trống nghiên cứu
10. Kiến trúc dữ liệu và hệ thống
11. Pipeline triển khai
12. Thiết kế mô hình nghiên cứu
13. Thiết kế thực nghiệm
14. Event Window Analysis
15. Hệ thống đánh giá rủi ro
16. Dashboard & Telegram Bot
17. Cấu trúc bài khóa luận
18. Checklist triển khai chi tiết
19. Risk Management cho dự án
20. Timeline hoàn thành
21. Tài liệu tham khảo cốt lõi
22. Định hướng mở rộng sau khóa luận

---

# 1. Tổng quan nghiên cứu

Khóa luận tập trung nghiên cứu cơ chế lan truyền rủi ro (Risk Spillover / Financial Contagion) từ các thị trường tài chính toàn cầu đến thị trường chứng khoán Việt Nam thông qua góc nhìn đa tần số (multi-frequency) và học máy nhân quả (causal machine learning).

Nghiên cứu xem VN-Index như một node trung tâm trong mạng lưới tài chính toàn cầu và phân tích:

* Tài sản nào ảnh hưởng mạnh nhất đến VN-Index
* Mức độ ảnh hưởng thay đổi như thế nào theo thời gian
* Ảnh hưởng khác nhau ra sao giữa ngắn hạn và dài hạn
* Cấu trúc nhân quả có thay đổi trong khủng hoảng hay không
* Có thể xây dựng hệ thống cảnh báo sớm bằng AI hay không

Nghiên cứu kết hợp:

| Thành phần        | Vai trò                                        |
| ----------------- | ---------------------------------------------- |
| MODWT             | Phân rã tín hiệu đa tần số                     |
| Wavelet Coherence | Phân tích đồng biến động theo thời gian–tần số |
| Neural Granger    | Học nhân quả phi tuyến                         |
| NOTEARS           | Xây dựng DAG không chu trình                   |
| Graph Theory      | Đo mức độ lan truyền rủi ro                    |
| Event Window      | Stress-test trong khủng hoảng                  |

---

# 2. Động cơ và tính cấp thiết

## 2.1. Bối cảnh toàn cầu

Giai đoạn 2015–2025 chứng kiến nhiều cú sốc hệ thống:

| Sự kiện            | Tác động                         |
| ------------------ | -------------------------------- |
| Bong bóng SSE 2015 | Shock từ Trung Quốc              |
| COVID-19           | Đồng pha hóa thị trường toàn cầu |
| BTC Crash 2021     | Risk-on/Risk-off dynamics        |
| Fed Hike 2022      | Tightening shock                 |
| Nga–Ukraine        | Energy & inflation crisis        |

Những cú sốc này cho thấy:

* Thị trường tài chính ngày càng liên kết mạnh
* Các cú sốc lan truyền nhanh hơn trước
* Thị trường mới nổi như Việt Nam dễ bị tác động gián tiếp

## 2.2. Hạn chế của phương pháp truyền thống

| Phương pháp         | Hạn chế                      |
| ------------------- | ---------------------------- |
| Pearson Correlation | Chỉ đo tương quan trung bình |
| VAR                 | Tuyến tính                   |
| GARCH               | Không có cấu trúc tần số     |
| TVP-VAR             | Khó mô hình hóa phi tuyến    |
| DCC-GARCH           | Không khám phá causal graph  |

Các mô hình trên không:

* Mô hình hóa được phi tuyến
* Khám phá được DAG động
* Tách được short-term và long-term spillover
* Xây dựng được hệ thống cảnh báo sớm trực quan

## 2.3. Lý do chọn hướng nghiên cứu này

Nghiên cứu này kết hợp:

* Wavelet → xử lý multi-scale dynamics
* Neural Granger → causal learning phi tuyến
* NOTEARS → causal graph discovery
* Graph analytics → đo contagion hubs

Đây là hướng tiếp cận hiện đại và rất mới trong tài chính định lượng tại Việt Nam.

---

# 3. Mục tiêu nghiên cứu

## 3.1. Mục tiêu tổng quát

Khám phá cấu trúc nhân quả đa tần số giữa các thị trường tài chính toàn cầu và VN-Index nhằm:

* Đo lường mức độ lan truyền rủi ro
* Xác định các nguồn phát sinh rủi ro chính
* Xây dựng hệ thống cảnh báo sớm bằng AI

## 3.2. Mục tiêu cụ thể

1. Phân tích đồng biến động đa tần số bằng Wavelet Coherence
2. Phân rã phương sai bằng MODWT
3. Xây dựng DAG bằng Neural Granger + NOTEARS
4. Xác định transmitter / receiver risk nodes
5. Phân tích contagion effect trong khủng hoảng
6. Thiết kế AI Risk Score System
7. Xây dựng Dashboard trực quan hóa DAG
8. Thiết kế Telegram Bot cảnh báo rủi ro

---

# 4. Câu hỏi nghiên cứu

1. VN-Index chịu ảnh hưởng mạnh nhất từ tài sản nào?
2. Cấu trúc nhân quả thay đổi như thế nào giữa ngắn hạn và dài hạn?
3. SP500 hay SSE ảnh hưởng mạnh hơn đến VN-Index?
4. VIX có đóng vai trò “Fear Transmitter” hay không?
5. Các cú sốc toàn cầu có làm DAG thay đổi mạnh không?
6. Neural causal discovery có vượt trội hơn VAR truyền thống không?
7. Có thể dùng AI để xây dựng hệ thống cảnh báo sớm không?

---

# 5. Đối tượng & phạm vi nghiên cứu

## 5.1. Biến nghiên cứu

| Nhóm                   | Biến          | Ticker    |
| ---------------------- | ------------- | --------- |
| Chứng khoán Việt Nam   | VN-Index      | VNINDEX   |
| Chứng khoán Mỹ         | S&P 500       | ^GSPC     |
| Chứng khoán Trung Quốc | SSE Composite | 000001.SS |
| Chỉ số sợ hãi          | VIX           | ^VIX      |
| Tiền điện tử           | Bitcoin       | BTC-USD   |
| Hàng hóa               | Dầu WTI       | CL=F      |
| Hàng hóa               | Vàng          | GC=F      |

## 5.2. Giai đoạn nghiên cứu

| Giai đoạn  | Thời gian | Đặc điểm                   |
| ---------- | --------- | -------------------------- |
| Pre-COVID  | 2015–2019 | Tương đối ổn định          |
| COVID      | 2020–2021 | Shock hệ thống             |
| Post-COVID | 2022–2025 | Tightening + địa chính trị |

## 5.3. Event Windows

| Event      | Window            |
| ---------- | ----------------- |
| SSE Bubble | 2015-06 → 2015-09 |
| BTC Crash  | 2021-04 → 2021-07 |
| Fed Hike   | 2022-03 → 2022-12 |

---

# 6. Đóng góp học thuật và thực tiễn

## 6.1. Đóng góp học thuật

| Thành phần                | Đóng góp                             |
| ------------------------- | ------------------------------------ |
| Wavelet + AI              | Kết hợp hiếm tại Việt Nam            |
| Neural DAG                | Phát hiện causal structure phi tuyến |
| Multi-frequency causality | Phân tích theo horizon               |
| Dynamic contagion         | Phân tích stress windows             |
| Risk Graph                | Trực quan hóa hệ thống tài chính     |

## 6.2. Đóng góp thực tiễn

### Đối với nhà đầu tư

* Theo dõi leading indicators
* Quản trị danh mục theo horizon
* Nhận cảnh báo sớm risk regime

### Đối với cơ quan quản lý

* Giám sát contagion channels
* Theo dõi systemic risk
* Cảnh báo áp lực dòng vốn

---

# 7. Cơ sở lý thuyết

## 7.1. Financial Contagion Theory

Khái niệm:

* Shock từ một thị trường lan sang thị trường khác
* Gia tăng mạnh trong khủng hoảng
* Mang tính network effect

## 7.2. Modern Portfolio Theory

* Diversification
* Correlation structure
* Dynamic hedging

## 7.3. Safe Haven Theory

| Khái niệm  | Ý nghĩa                         |
| ---------- | ------------------------------- |
| Hedge      | Tương quan âm trung bình        |
| Safe Haven | Tương quan âm trong khủng hoảng |

## 7.4. Structural Causal Model

DAG gồm:

* Node = tài sản
* Edge = causal effect

Mục tiêu:

* Tìm causal transmitter
* Tìm contagion hub
* Mô hình hóa directed spillover

## 7.5. Wavelet Theory

Wavelet cho phép:

* Time-frequency analysis
* Multi-scale decomposition
* Localized dynamics

## 7.6. Neural Granger Causality

Sử dụng cLSTM để:

* Học quan hệ nhân quả phi tuyến
* Học temporal dependencies
* Kết hợp group lasso để sparse selection

## 7.7. NOTEARS

NOTEARS:

* Học DAG bằng continuous optimization
* Ép đồ thị không chu trình
* Thay thế brute-force DAG search

---

# 8. Tổng quan nghiên cứu thực nghiệm

## 8.1. Nhóm nghiên cứu Wavelet

| Tác giả                  | Phương pháp       |
| ------------------------ | ----------------- |
| Torrence & Compo (1998)  | Wavelet analysis  |
| Grinsted et al. (2004)   | Wavelet coherence |
| Percival & Walden (2000) | MODWT             |

## 8.2. Nhóm Spillover

| Tác giả          | Nội dung               |
| ---------------- | ---------------------- |
| Diebold & Yilmaz | Volatility spillover   |
| Forbes & Rigobon | Contagion              |
| Bekaert et al.   | VIX & emerging markets |

## 8.3. Nhóm Causal ML

| Tác giả      | Nội dung       |
| ------------ | -------------- |
| Pearl        | Causality      |
| Zheng et al. | NOTEARS        |
| Tank et al.  | Neural Granger |

---

# 9. Khoảng trống nghiên cứu

## 9.1. Khoảng trống hiện tại

Các nghiên cứu tại Việt Nam:

* Chủ yếu dùng Wavelet hoặc VAR
* Chưa dùng causal machine learning
* Chưa xây dựng DAG đa tần số
* Chưa có AI-based risk monitoring system

## 9.2. Điểm mới của khóa luận

| Thành phần                | Điểm mới          |
| ------------------------- | ----------------- |
| Neural DAG                | Rất mới tại VN    |
| Multi-frequency causality | Hiếm              |
| Risk graph                | Hiếm              |
| AI Warning System         | Mới               |
| Dashboard                 | Tính ứng dụng cao |

---

# 10. Kiến trúc dữ liệu và hệ thống

## 10.1. Data Flow

```text
API Data Sources
      ↓
Data Cleaning
      ↓
Log-return + Volatility
      ↓
MODWT Decomposition
      ↓
Wavelet Coherence
      ↓
Neural Causal Discovery
      ↓
Graph Analytics
      ↓
Risk Score Engine
      ↓
Dashboard + Telegram Bot
```

## 10.2. Data Sources

| Data          | Source   |
| ------------- | -------- |
| VNINDEX       | vnstock  |
| Global assets | yfinance |

## 10.3. Data Processing Policy

| Trường hợp          | Xử lý          |
| ------------------- | -------------- |
| Mỹ nghỉ lễ          | ffill(limit=3) |
| Trung Quốc nghỉ dài | interpolate    |
| BTC missing đầu kỳ  | Không fill     |
| Missing bất thường  | Drop           |

---

# 11. Pipeline triển khai

## 11.1. Main Pipeline

```text
01_data_ingestion.py
02_preprocessing.py
03_wavelet_analysis.py
04_modwt_decomposition.py
05_neural_causal_discovery.py
06_graph_metrics.py
07_event_analysis.py
08_dashboard.py
09_telegram_bot.py
```

## 11.2. Project Structure

```text
thesis_project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── models/
│   ├── utils/
│   ├── visualization/
│   └── risk_engine/
├── results/
│   ├── tables/
│   └── figures/
├── app/
├── thesis/
└── requirements.txt
```

---

# 12. Thiết kế mô hình nghiên cứu

# 12.1. MODWT Variance Decomposition

Mục tiêu:

* Tách tín hiệu theo frequency band
* Đo volatility contribution

Frequency bands:

| Scale | Horizon   |
| ----- | --------- |
| D1    | 2–4 ngày  |
| D2    | 4–8 ngày  |
| D3    | 8–16 ngày |
| A3    | Dài hạn   |

## 12.2. Wavelet Coherence

Mục tiêu:

* Khám phá co-movement
* Xác định lead-lag
* Quan sát contagion dynamics

## 12.3. Neural Granger

Architecture:

```text
Input Time Series
        ↓
LSTM Layers
        ↓
Group Lasso Penalty
        ↓
Sparse Causal Matrix
```

## 12.4. NOTEARS

Constraint:

* DAG không chu trình
* Continuous optimization

Output:

* Weighted adjacency matrix
* Directed causal graph

## 12.5. Graph Analytics

Metrics:

| Metric         | Ý nghĩa            |
| -------------- | ------------------ |
| In-degree      | Receiver risk      |
| Out-degree     | Transmitter risk   |
| Net Risk Score | Risk balance       |
| PageRank       | Central importance |
| Betweenness    | Contagion hub      |

---

# 13. Thiết kế thực nghiệm

## 13.1. Baseline Models

| Model           | Vai trò              |
| --------------- | -------------------- |
| Pearson         | Baseline correlation |
| VAR             | Baseline causality   |
| Wavelet-Granger | Frequency causality  |
| Neural DAG      | Final model          |

## 13.2. Evaluation Criteria

| Metric           | Ý nghĩa             |
| ---------------- | ------------------- |
| Stability        | Độ ổn định DAG      |
| Interpretability | Khả năng giải thích |
| Forecast quality | Chất lượng cảnh báo |
| Graph sparsity   | Mức độ tối giản DAG |

## 13.3. Robustness Checks

* SSE vs CSI300
* Different wavelet basis
* Different lag length
* Different regularization strength

---

# 14. Event Window Analysis

## 14.1. SSE Bubble 2015

Mục tiêu:

* Kiểm tra contagion từ Trung Quốc
* Quan sát SSE → VNI

## 14.2. BTC Crash 2021

Mục tiêu:

* Kiểm tra crypto spillover
* Quan sát BTC & VIX dynamics

## 14.3. Fed Hike 2022

Mục tiêu:

* Quan sát tightening shock
* Theo dõi SP500 & VIX transmission

## 14.4. Event Outputs

Mỗi event sẽ có:

* Wavelet plots
* DAG comparison
* Risk score comparison
* Contagion premium analysis

---

# 15. Hệ thống đánh giá rủi ro

## 15.1. Risk Score Engine

Risk score được xây dựng từ:

* Out-degree
* Weighted centrality
* Incoming spillover
* Graph density

## 15.2. Risk Regimes

| Regime       | Ý nghĩa     |
| ------------ | ----------- |
| Low Risk     | Bình thường |
| Medium Risk  | Cảnh báo    |
| High Risk    | Stress      |
| Extreme Risk | Contagion   |

## 15.3. Early Warning Logic

Ví dụ:

```text
Nếu:
- VIX → VNI tăng mạnh
- Net Risk Score > 95 percentile
- Graph density tăng nhanh

→ Trigger ALERT
```

---

# 16. Dashboard & Telegram Bot

## 16.1. Dashboard

Công nghệ:

* Streamlit
* Plotly
* NetworkX

Features:

* Interactive DAG
* Risk heatmap
* Dynamic centrality chart
* Event replay
* Frequency selector

## 16.2. Telegram Bot

Ví dụ message:

```text
⚠️ RISK ALERT

Net Risk Score của VNINDEX vượt ngưỡng 95%
Nguồn rủi ro chính:
- VIX
- SP500

Khuyến nghị:
Theo dõi volatility regime.
```

---

# 17. Cấu trúc bài khóa luận

## Chương 1 — Giới thiệu

* Bối cảnh
* Tính cấp thiết
* Mục tiêu
* Câu hỏi nghiên cứu

## Chương 2 — Cơ sở lý thuyết

* Spillover
* Wavelet
* SCM
* Neural causality

## Chương 3 — Dữ liệu & Phương pháp

* Data pipeline
* MODWT
* WTC
* cLSTM
* NOTEARS

## Chương 4 — Thực nghiệm

* Descriptive statistics
* WTC
* DAG
* Risk metrics
* Event windows

## Chương 5 — Kết luận

* Findings
* Implications
* Limitations
* Future work

---

# 18. Checklist triển khai chi tiết

## Giai đoạn 1 — Data Engineering

* [x] Crawl dữ liệu
* [x] Merge datasets
* [x] Fill missing values
* [x] Log-return
* [x] Rolling volatility
* [x] ADF/KPSS

## Giai đoạn 2 — Wavelet Analysis

* [x] Wavelet Coherence
* [ ] MODWT decomposition
* [ ] Frequency exports

## Giai đoạn 3 — Neural DAG

* [ ] VAR baseline
* [ ] cLSTM implementation
* [ ] Group Lasso
* [ ] NOTEARS integration
* [ ] DAG rendering

## Giai đoạn 4 — Risk System

* [ ] Risk score engine
* [ ] Centrality metrics
* [ ] Event analysis
* [ ] Stress testing

## Giai đoạn 5 — Application

* [ ] Streamlit dashboard
* [ ] Telegram bot
* [ ] Final report
* [ ] Slide defense

---

# 19. Risk Management cho dự án

| Rủi ro            | Giải pháp               |
| ----------------- | ----------------------- |
| Dữ liệu thiếu     | Multi-source fallback   |
| DAG quá dense     | Stronger regularization |
| Training unstable | Early stopping          |
| Overfitting       | Cross-validation        |
| Dashboard lag     | Cache & optimization    |

---

# 20. Timeline hoàn thành

| Giai đoạn        | Thời gian |
| ---------------- | --------- |
| Data pipeline    | 1–2 tuần  |
| Wavelet analysis | 1 tuần    |
| Neural DAG       | 2–3 tuần  |
| Risk engine      | 1 tuần    |
| Dashboard        | 1 tuần    |
| Viết khóa luận   | 2–3 tuần  |
| Slide & defense  | 1 tuần    |

---

# 21. Tài liệu tham khảo cốt lõi

## Wavelet & Econometrics

* Torrence & Compo (1998)
* Percival & Walden (2000)
* Grinsted et al. (2004)
* Diebold & Yilmaz (2012)

## Causal Machine Learning

* Pearl (2009)
* Zheng et al. (2018)
* Tank et al. (2021)

## Financial Spillover

* Forbes & Rigobon (2002)
* Baur & Lucey (2010)
* Whaley (2000)
* Bekaert et al. (2013)

---

# 22. Định hướng mở rộng sau khóa luận

## Hướng 1 — Dynamic Graph Neural Networks

* Temporal Graph Networks
* Dynamic adjacency matrix

## Hướng 2 — Macro-financial Integration

Thêm:

* CPI
* FED rate
* USD/VND
* Bond yields

## Hướng 3 — ASEAN Spillover Network

Mở rộng:

* VN
* Thailand
* Indonesia
* Malaysia
* Singapore

## Hướng 4 — Real-time Monitoring System

* Kafka streaming
* Real-time DAG updates
* Intraday spillover detection

---

# Kết luận định hướng

Đây là một đề tài có:

* Độ mới học thuật cao
* Kết hợp AI + Econometrics + Network Science
* Tính ứng dụng mạnh
* Có khả năng phát triển thành nghiên cứu khoa học hoặc paper quốc tế

Nếu triển khai tốt:

* Có thể publish conference/journal
* Có thể mở rộng thành hệ thống quản trị rủi ro thực tế
* Có giá trị cao khi apply Data Science / Quant / AI Research
