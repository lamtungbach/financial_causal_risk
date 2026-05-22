# KẾ HOẠCH KHÓA LUẬN TỐT NGHIỆP — PHIÊN BẢN AI-FIRST 1.0

# Xây dựng framework học máy nhân quả đa tần số cho phát hiện lan truyền rủi ro trên chuỗi thời gian tài chính

# Ứng dụng trên thị trường chứng khoán Việt Nam và các chỉ báo tài chính toàn cầu

---

## Thông tin tổng quan

| Thành phần          | Nội dung                                                                 |
| ------------------- | ------------------------------------------------------------------------ |
| Tên đề tài đề xuất  | Xây dựng framework học máy nhân quả đa tần số cho phát hiện lan truyền rủi ro trên chuỗi thời gian tài chính |
| Định hướng ngành    | Trí tuệ nhân tạo, học máy chuỗi thời gian, học máy nhân quả, học sâu      |
| Miền ứng dụng       | Chuỗi thời gian tài chính, lan truyền rủi ro, thị trường chứng khoán Việt Nam |
| Trọng tâm nghiên cứu | Multi-scale representation + Neural Granger Causality + Graph-based Risk Analysis |
| Phương pháp lõi     | MODWT, Wavelet Coherence, Neural Granger, Sparse Learning, Graph Analytics |
| Phương pháp mở rộng | NOTEARS, PCMCI, Dynamic Graph Neural Networks                             |
| Giai đoạn dữ liệu   | 01/01/2015 → 31/12/2025                                                   |
| Tần suất dữ liệu    | Daily                                                                     |
| Ngôn ngữ triển khai | Python                                                                    |
| Framework AI        | PyTorch                                                                   |
| Công cụ trực quan   | Streamlit, Plotly, NetworkX                                               |
| Đầu ra cuối cùng    | Khóa luận hoàn chỉnh + source code + kết quả thực nghiệm + dashboard minh họa + slide bảo vệ |

---

# MỤC LỤC

1. Tổng quan định hướng khóa luận
2. Động cơ nghiên cứu và tính cấp thiết
3. Phát biểu bài toán AI
4. Mục tiêu nghiên cứu
5. Câu hỏi nghiên cứu
6. Đối tượng, dữ liệu và phạm vi nghiên cứu
7. Đóng góp dự kiến
8. Cơ sở lý thuyết
9. Tổng quan nghiên cứu liên quan
10. Khoảng trống nghiên cứu
11. Framework đề xuất
12. Thiết kế dữ liệu và tiền xử lý
13. Thiết kế mô hình
14. Thiết kế thực nghiệm
15. Đánh giá mô hình
16. Ablation Study và Robustness Check
17. Phân tích kết quả theo giai đoạn sự kiện
18. Graph-based Risk Index
19. Dashboard minh họa kết quả
20. Cấu trúc khóa luận
21. Checklist triển khai
22. Quản trị rủi ro dự án
23. Timeline hoàn thành
24. Tài liệu tham khảo cốt lõi
25. Định hướng mở rộng sau khóa luận

---

# 1. Tổng quan định hướng khóa luận

Khóa luận tập trung xây dựng một framework học máy nhân quả đa tần số cho dữ liệu chuỗi thời gian tài chính. Trong đó, dữ liệu tài chính được xem là miền ứng dụng để kiểm chứng năng lực của mô hình AI trong việc phát hiện các quan hệ phụ thuộc có hướng giữa nhiều chuỗi thời gian.

Thay vì đặt trọng tâm vào phân tích kinh tế lượng truyền thống, khóa luận nhấn mạnh vào các vấn đề cốt lõi của Trí tuệ nhân tạo:

* Biểu diễn chuỗi thời gian đa tỉ lệ bằng wavelet decomposition
* Học quan hệ định hướng phi tuyến bằng Neural Granger Causality
* Học cấu trúc đồ thị thưa từ dữ liệu chuỗi thời gian đa biến
* Đánh giá mô hình bằng baseline thống kê, học máy và học sâu
* Phân tích khả năng diễn giải của mô hình thông qua đồ thị quan hệ
* Xây dựng prototype trực quan hóa kết quả mô hình

Miền ứng dụng cụ thể là bài toán phát hiện lan truyền rủi ro từ các chỉ báo tài chính toàn cầu đến thị trường chứng khoán Việt Nam.

---

# 2. Động cơ nghiên cứu và tính cấp thiết

## 2.1. Bối cảnh dữ liệu chuỗi thời gian tài chính

Chuỗi thời gian tài chính là một dạng dữ liệu phức tạp với nhiều đặc điểm gây khó khăn cho mô hình học máy:

| Đặc điểm | Ý nghĩa đối với mô hình AI |
| -------- | --------------------------- |
| Nhiễu cao | Mô hình dễ học quan hệ giả |
| Phi tuyến | Các mô hình tuyến tính khó nắm bắt đầy đủ quan hệ |
| Không dừng | Phân phối dữ liệu thay đổi theo thời gian |
| Đa tần số | Quan hệ ngắn hạn và dài hạn có thể khác nhau |
| Phụ thuộc chéo | Nhiều chuỗi có thể ảnh hưởng lẫn nhau |
| Chịu tác động sự kiện | Cấu trúc quan hệ thay đổi trong khủng hoảng |

Do đó, đây là một miền dữ liệu phù hợp để kiểm chứng các phương pháp học máy nhân quả và học sâu cho chuỗi thời gian đa biến.

## 2.2. Hạn chế của cách tiếp cận truyền thống

| Nhóm phương pháp | Hạn chế chính |
| ---------------- | ------------- |
| Pearson Correlation | Chỉ đo tương quan trung bình, không có hướng |
| VAR Granger | Tuyến tính, khó học quan hệ phi tuyến |
| GARCH / DCC-GARCH | Tập trung vào volatility, không trực tiếp học causal graph |
| Wavelet truyền thống | Phân tích tốt theo tần số nhưng chưa tự động học cấu trúc nhân quả |
| Mô hình deep learning dự báo | Có thể dự báo tốt nhưng thường khó diễn giải quan hệ giữa biến |

Khóa luận đề xuất kết hợp biểu diễn đa tần số với neural causal discovery để vừa học được quan hệ phi tuyến, vừa giữ được khả năng diễn giải thông qua đồ thị.

## 2.3. Lý do chọn hướng AI-first

Đề tài thuộc ngành Trí tuệ nhân tạo nên trọng tâm không phải là khẳng định một kết luận kinh tế tuyệt đối, mà là xây dựng và đánh giá một framework AI có khả năng:

* Học biểu diễn đa tần số cho chuỗi thời gian
* Phát hiện quan hệ phụ thuộc định hướng theo nghĩa Granger
* Biểu diễn kết quả dưới dạng graph có thể diễn giải
* So sánh với các baseline truyền thống và học sâu
* Ứng dụng vào một bài toán thực tế có ý nghĩa

---

# 3. Phát biểu bài toán AI

## 3.1. Dữ liệu đầu vào

Cho tập chuỗi thời gian đa biến:

```text
X = {x_1, x_2, ..., x_N}
```

Trong đó:

* `N` là số lượng biến tài chính
* `x_i` là chuỗi thời gian của biến thứ `i`
* Mỗi quan sát được lấy theo tần suất ngày
* Các chuỗi được đồng bộ theo cùng một trục thời gian

Tại thời điểm `t`, vector quan sát được ký hiệu:

```text
X_t = [x_{1,t}, x_{2,t}, ..., x_{N,t}]
```

## 3.2. Bài toán học quan hệ định hướng

Mục tiêu là học một ma trận quan hệ có hướng:

```text
A ∈ R^{N × N}
```

Trong đó:

```text
A_ij > 0
```

biểu thị rằng chuỗi `x_i` có đóng góp dự báo đối với chuỗi `x_j` theo nghĩa Granger.

Nói cách khác, nếu quá khứ của `x_i` giúp cải thiện khả năng dự báo `x_j`, mô hình có thể tạo một cạnh có hướng:

```text
x_i → x_j
```

## 3.3. Bài toán học quan hệ đa tần số

Sau khi phân rã wavelet, mỗi chuỗi thời gian được tách thành nhiều thành phần theo tần số:

```text
x_i = D_{i,1} + D_{i,2} + ... + D_{i,K} + A_{i,K}
```

Mục tiêu khi đó là học một tập các ma trận quan hệ:

```text
A^(1), A^(2), ..., A^(K)
```

Trong đó mỗi `A^(k)` biểu diễn graph quan hệ ở một dải tần số khác nhau:

| Thành phần | Ý nghĩa |
| ---------- | ------- |
| D1 | Quan hệ rất ngắn hạn |
| D2 | Quan hệ ngắn hạn |
| D3 | Quan hệ trung hạn |
| A3 | Quan hệ dài hạn |

## 3.4. Đầu ra mong muốn

Đầu ra của framework gồm:

* Ma trận quan hệ định hướng giữa các biến
* Đồ thị causal/spillover graph theo từng dải tần
* Các chỉ số graph như in-degree, out-degree, PageRank, betweenness
* Risk index dựa trên cấu trúc graph
* Kết quả so sánh với baseline
* Dashboard trực quan hóa kết quả

---

# 4. Mục tiêu nghiên cứu

## 4.1. Mục tiêu tổng quát

Xây dựng và đánh giá một framework học máy nhân quả đa tần số kết hợp wavelet decomposition và Neural Granger Causality nhằm phát hiện các quan hệ phụ thuộc định hướng trong dữ liệu chuỗi thời gian tài chính.

## 4.2. Mục tiêu cụ thể

1. Thu thập, đồng bộ và tiền xử lý dữ liệu chuỗi thời gian tài chính đa biến.
2. Xây dựng biểu diễn đa tần số cho chuỗi thời gian bằng MODWT.
3. Phân tích quan hệ thời gian–tần số bằng Wavelet Coherence.
4. Xây dựng mô hình Neural Granger để học quan hệ định hướng phi tuyến.
5. Sinh đồ thị quan hệ thưa từ trọng số học được của mô hình.
6. So sánh mô hình đề xuất với các baseline thống kê, học máy và học sâu.
7. Đánh giá mô hình theo khả năng dự báo, độ ổn định graph, độ thưa graph và khả năng diễn giải.
8. Phân tích sự thay đổi graph trong các giai đoạn thị trường khác nhau.
9. Xây dựng Graph-based Risk Index để minh họa khả năng ứng dụng.
10. Xây dựng dashboard đơn giản để trực quan hóa kết quả mô hình.

---

# 5. Câu hỏi nghiên cứu

1. Biểu diễn đa tần số bằng wavelet có cải thiện khả năng phát hiện quan hệ định hướng trong chuỗi thời gian tài chính không?
2. Neural Granger Causality có phát hiện được quan hệ phi tuyến tốt hơn VAR Granger truyền thống không?
3. Cấu trúc graph học được có thay đổi giữa ngắn hạn, trung hạn và dài hạn không?
4. Graph học được có ổn định qua các giai đoạn thị trường khác nhau không?
5. Cấu trúc graph có trở nên dày hơn hoặc tập trung hơn trong các giai đoạn rủi ro cao không?
6. Graph-based Risk Index có hỗ trợ phát hiện các giai đoạn biến động mạnh của thị trường không?
7. Framework đề xuất có khả năng diễn giải tốt hơn so với mô hình deep learning dự báo thuần túy không?

---

# 6. Đối tượng, dữ liệu và phạm vi nghiên cứu

## 6.1. Đối tượng nghiên cứu

Đối tượng nghiên cứu là dữ liệu chuỗi thời gian tài chính đa biến và bài toán học quan hệ định hướng giữa các chuỗi thời gian.

Miền ứng dụng là phát hiện lan truyền rủi ro từ các thị trường và chỉ báo tài chính toàn cầu đến thị trường chứng khoán Việt Nam.

## 6.2. Biến nghiên cứu chính

| Nhóm | Biến | Ticker gợi ý | Vai trò |
| ---- | ---- | ------------ | ------- |
| Việt Nam | VN-Index | VNINDEX | Biến trung tâm cần phân tích |
| Mỹ | S&P 500 | ^GSPC | Đại diện thị trường chứng khoán Mỹ |
| Trung Quốc | SSE Composite | 000001.SS | Đại diện thị trường Trung Quốc |
| Rủi ro toàn cầu | VIX | ^VIX | Chỉ báo fear/risk sentiment |
| Tiền điện tử | Bitcoin | BTC-USD | Đại diện tài sản rủi ro mới |
| Hàng hóa | Dầu WTI | CL=F | Đại diện năng lượng |
| Hàng hóa | Vàng | GC=F | Đại diện safe haven |
| Tỷ giá | USD/VND hoặc DXY | USDVND / DX-Y.NYB | Kênh tỷ giá và sức mạnh USD |
| Lãi suất | US 10Y Yield | ^TNX | Kênh lãi suất toàn cầu |

## 6.3. Biến mục tiêu và biến đầu vào

Tùy theo thực nghiệm, khóa luận có thể sử dụng hai dạng dữ liệu chính:

| Dạng dữ liệu | Công thức / mô tả | Mục đích |
| ------------ | ----------------- | -------- |
| Log-return | `r_t = log(P_t / P_{t-1})` | Học quan hệ biến động giá |
| Volatility proxy | rolling std hoặc absolute return | Học quan hệ lan truyền rủi ro |

## 6.4. Giai đoạn nghiên cứu

| Giai đoạn | Thời gian | Đặc điểm |
| --------- | --------- | -------- |
| Pre-COVID | 2015–2019 | Giai đoạn tương đối ổn định |
| COVID | 2020–2021 | Cú sốc hệ thống toàn cầu |
| Post-COVID | 2022–2025 | Lạm phát, Fed tightening, địa chính trị |

## 6.5. Event Windows dự kiến

| Event | Window | Mục đích phân tích |
| ----- | ------ | ------------------ |
| SSE Bubble | 2015-06 → 2015-09 | Kiểm tra ảnh hưởng từ Trung Quốc |
| COVID-19 Crash | 2020-02 → 2020-06 | Kiểm tra shock hệ thống toàn cầu |
| BTC Crash | 2021-04 → 2021-07 | Kiểm tra spillover từ crypto |
| Fed Hike Cycle | 2022-03 → 2022-12 | Kiểm tra ảnh hưởng tightening |

## 6.6. Phạm vi giới hạn

* Dữ liệu ở tần suất ngày, chưa xét dữ liệu intraday.
* Kết quả causal discovery được hiểu theo nghĩa Granger/directional predictability, không khẳng định nhân quả tuyệt đối theo nghĩa thực nghiệm can thiệp.
* Mô hình tập trung vào quan hệ giữa các biến tài chính, chưa đưa đầy đủ các biến vĩ mô nội địa.
* Dashboard chỉ đóng vai trò minh họa kết quả, không phải hệ thống real-time production.

---

# 7. Đóng góp dự kiến

## 7.1. Đóng góp về mặt AI

| Thành phần | Đóng góp |
| ---------- | -------- |
| Multi-scale representation | Dùng MODWT để biểu diễn chuỗi thời gian theo nhiều dải tần |
| Neural causal discovery | Áp dụng Neural Granger để học quan hệ định hướng phi tuyến |
| Sparse graph learning | Sinh graph thưa có khả năng diễn giải từ mô hình học sâu |
| Model evaluation | Đánh giá bằng baseline thống kê, học máy và học sâu |
| Ablation study | Kiểm tra vai trò của wavelet, lag length, regularization và biến đầu vào |
| Explainable AI | Diễn giải mô hình thông qua graph metrics và visualization |

## 7.2. Đóng góp về mặt ứng dụng

| Thành phần | Ý nghĩa |
| ---------- | ------- |
| Financial risk monitoring | Minh họa cách AI có thể hỗ trợ theo dõi lan truyền rủi ro |
| Graph-based risk index | Biến kết quả mô hình thành chỉ số dễ hiểu hơn |
| Event analysis | Cho thấy mô hình phản ứng như thế nào trong các giai đoạn thị trường căng thẳng |
| Dashboard | Hỗ trợ trực quan hóa graph và risk regime |

---

# 8. Cơ sở lý thuyết

## 8.1. Chuỗi thời gian đa biến

Chuỗi thời gian đa biến là tập nhiều chuỗi quan sát theo thời gian. Trong bài toán này, mỗi biến tài chính là một chuỗi và các chuỗi có thể có quan hệ phụ thuộc chéo.

Các vấn đề chính:

* Tính không dừng
* Tự tương quan
* Phụ thuộc theo độ trễ
* Phụ thuộc chéo giữa nhiều biến
* Thay đổi phân phối theo thời gian

## 8.2. Granger Causality

Một chuỗi `x_i` được xem là Granger-cause `x_j` nếu quá khứ của `x_i` giúp cải thiện dự báo `x_j` so với chỉ dùng quá khứ của `x_j`.

Đây không phải là nhân quả tuyệt đối theo nghĩa can thiệp, mà là quan hệ định hướng dựa trên khả năng dự báo.

## 8.3. Neural Granger Causality

Neural Granger mở rộng Granger truyền thống bằng cách dùng mạng neural để học quan hệ phi tuyến.

Ý tưởng chính:

* Sử dụng LSTM hoặc MLP để dự báo chuỗi mục tiêu
* Quan sát trọng số đầu vào hoặc nhóm tham số theo từng biến
* Dùng regularization để loại bỏ các biến không có đóng góp dự báo
* Sinh ma trận quan hệ có hướng từ cấu trúc trọng số học được

## 8.4. Sparse Learning và Group Lasso

Group Lasso được dùng để ép mô hình chọn hoặc loại bỏ toàn bộ nhóm đặc trưng tương ứng với một biến đầu vào.

Trong bài toán Neural Granger, group lasso giúp:

* Làm graph thưa hơn
* Giảm quan hệ giả
* Tăng khả năng diễn giải
* Hỗ trợ phát hiện biến nào ảnh hưởng đến biến nào

## 8.5. Wavelet Decomposition

Wavelet decomposition cho phép phân rã chuỗi thời gian thành nhiều thành phần ở các dải tần số khác nhau.

Trong khóa luận, wavelet được xem như một bước biểu diễn dữ liệu đa tỉ lệ:

```text
Raw time series → Multi-scale components → Neural causal discovery
```

## 8.6. MODWT

MODWT phù hợp với dữ liệu tài chính vì:

* Không yêu cầu độ dài chuỗi là lũy thừa của 2
* Giữ nguyên số lượng quan sát sau phân rã
* Phù hợp với phân tích biến động theo nhiều horizon
* Dễ liên kết với rolling window và event analysis

## 8.7. Wavelet Coherence

Wavelet Coherence dùng để đo mức độ đồng biến động giữa hai chuỗi theo cả thời gian và tần số.

Vai trò trong khóa luận:

* Phân tích mô tả trước khi học graph
* Quan sát lead-lag pattern
* Kiểm tra sự khác biệt giữa các giai đoạn thị trường
* Hỗ trợ diễn giải kết quả Neural Granger

## 8.8. Causal Discovery

Causal discovery là nhóm phương pháp học cấu trúc quan hệ giữa các biến từ dữ liệu.

Trong khóa luận, causal discovery được sử dụng theo hướng:

* Học quan hệ định hướng trong chuỗi thời gian
* Tạo graph có thể diễn giải
* So sánh giữa mô hình tuyến tính và phi tuyến

## 8.9. Graph Representation và Graph Analytics

Kết quả causal discovery được biểu diễn bằng directed graph:

* Node: biến tài chính
* Edge: quan hệ định hướng học được
* Edge weight: độ mạnh quan hệ

Các chỉ số graph giúp diễn giải vai trò của từng biến trong hệ thống.

## 8.10. Financial Spillover như miền ứng dụng

Financial spillover được dùng làm bối cảnh ứng dụng. Trong khóa luận này, spillover được hiểu là sự lan truyền thông tin hoặc rủi ro giữa các biến tài chính, được mô hình hóa bằng quan hệ định hướng trên graph.

---

# 9. Tổng quan nghiên cứu liên quan

## 9.1. Nhóm nghiên cứu về wavelet cho chuỗi thời gian

| Tác giả | Nội dung |
| ------- | -------- |
| Torrence & Compo (1998) | Wavelet analysis cho chuỗi thời gian |
| Percival & Walden (2000) | Wavelet methods và MODWT |
| Grinsted et al. (2004) | Wavelet coherence |

## 9.2. Nhóm nghiên cứu về Granger và Neural Granger

| Tác giả | Nội dung |
| ------- | -------- |
| Granger (1969) | Khái niệm Granger causality |
| Tank et al. (2021) | Neural Granger causality cho chuỗi thời gian |
| Khanna & Tan (2020) | Economy statistical recurrent units cho Granger causality |

## 9.3. Nhóm nghiên cứu về causal discovery

| Tác giả | Nội dung |
| ------- | -------- |
| Pearl (2009) | Causality và Structural Causal Models |
| Zheng et al. (2018) | NOTEARS và continuous optimization cho DAG learning |
| Runge et al. | PCMCI cho causal discovery trong time series |

## 9.4. Nhóm nghiên cứu về financial spillover

| Tác giả | Nội dung |
| ------- | -------- |
| Diebold & Yilmaz | Spillover index và connectedness |
| Forbes & Rigobon | Financial contagion |
| Bekaert et al. | Global risk, VIX và thị trường mới nổi |

---

# 10. Khoảng trống nghiên cứu

## 10.1. Khoảng trống về phương pháp

Các nghiên cứu hiện có thường gặp một số giới hạn:

* Nhiều mô hình chỉ xét quan hệ tuyến tính
* Một số nghiên cứu wavelet chỉ dừng ở phân tích đồng biến động, chưa học graph định hướng
* Các mô hình deep learning thường dự báo tốt nhưng khó giải thích quan hệ giữa biến
* Ít nghiên cứu kết hợp multi-scale representation với neural causal discovery
* Ít nghiên cứu đánh giá causal graph bằng ablation và robustness check

## 10.2. Khoảng trống về ứng dụng tại Việt Nam

Trong bối cảnh thị trường Việt Nam:

* Các nghiên cứu về spillover thường dùng VAR, GARCH hoặc wavelet truyền thống
* Chưa phổ biến hướng Neural Granger cho phát hiện quan hệ định hướng
* Chưa nhiều nghiên cứu xây dựng graph-based risk index từ mô hình học máy
* Chưa nhiều hệ thống trực quan hóa causal graph cho dữ liệu tài chính Việt Nam

## 10.3. Điểm mới của khóa luận

| Thành phần | Điểm mới |
| ---------- | -------- |
| Wavelet + Neural Granger | Kết hợp biểu diễn đa tần số với học nhân quả phi tuyến |
| Frequency-specific graph | Học graph riêng cho từng dải tần |
| Graph-based interpretation | Diễn giải mô hình thông qua graph metrics |
| AI evaluation | So sánh với baseline và thực hiện ablation study |
| Application prototype | Dashboard minh họa kết quả trên dữ liệu thực |

---

# 11. Framework đề xuất

## 11.1. Tổng quan framework

```text
Raw Financial Data
        ↓
Data Cleaning & Alignment
        ↓
Log-return / Volatility Construction
        ↓
Stationarity Check & Normalization
        ↓
MODWT Multi-scale Decomposition
        ↓
Wavelet Coherence Analysis
        ↓
Neural Granger Causal Discovery
        ↓
Sparse Directed Graph Construction
        ↓
Graph Analytics
        ↓
Graph-based Risk Index
        ↓
Visualization Dashboard
```

## 11.2. Vai trò của từng thành phần

| Thành phần | Vai trò |
| ---------- | ------- |
| Data preprocessing | Làm sạch, đồng bộ, chuẩn hóa dữ liệu |
| MODWT | Tạo biểu diễn đa tần số |
| Wavelet Coherence | Phân tích mô tả quan hệ thời gian–tần số |
| Neural Granger | Học quan hệ định hướng phi tuyến |
| Sparse regularization | Loại bỏ cạnh yếu và tăng diễn giải |
| Graph analytics | Đo vai trò của từng node trong graph |
| Risk index | Chuyển graph thành chỉ số ứng dụng |
| Dashboard | Trực quan hóa kết quả |

## 11.3. Đóng góp chính của framework

Đóng góp chính không nằm ở việc phát minh một mô hình hoàn toàn mới, mà ở việc thiết kế một framework tích hợp có hệ thống:

```text
Multi-scale representation + Neural causal discovery + Graph-based interpretation
```

Framework này phù hợp với định hướng khóa luận ngành Trí tuệ nhân tạo vì tập trung vào mô hình, biểu diễn dữ liệu, đánh giá thực nghiệm và khả năng giải thích.

---

# 12. Thiết kế dữ liệu và tiền xử lý

## 12.1. Data Sources

| Dữ liệu | Nguồn gợi ý |
| ------- | ----------- |
| VNINDEX | vnstock, FiinPro nếu có, Stooq nếu phù hợp |
| Global assets | yfinance |
| VIX | yfinance |
| DXY / US10Y | yfinance, FRED nếu cần |

## 12.2. Data Processing Policy

| Trường hợp | Cách xử lý đề xuất |
| ---------- | ------------------ |
| Ngày nghỉ khác nhau giữa thị trường | Align theo trading days, forward-fill có giới hạn |
| Missing ngắn hạn | ffill(limit=3) |
| Missing dài hạn | Loại bỏ hoặc thay nguồn dữ liệu |
| Outlier bất thường | Kiểm tra bằng thống kê và tin tức sự kiện |
| Chuỗi giá | Chuyển sang log-return |
| Chuỗi volatility | Tính rolling volatility hoặc absolute return |

## 12.3. Kiểm định và chuẩn hóa

Các bước cần thực hiện:

* Kiểm tra missing value
* Kiểm tra outlier
* Tính log-return
* Kiểm định tính dừng bằng ADF/KPSS
* Chuẩn hóa bằng z-score hoặc robust scaler
* Chia dữ liệu train/validation/test theo thời gian

## 12.4. Chia tập dữ liệu

| Tập dữ liệu | Giai đoạn gợi ý | Mục đích |
| ----------- | --------------- | -------- |
| Train | 2015–2021 | Huấn luyện mô hình |
| Validation | 2022–2023 | Chọn tham số |
| Test | 2024–2025 | Đánh giá cuối |

Có thể dùng thêm rolling window để đánh giá tính ổn định theo thời gian.

---

# 13. Thiết kế mô hình

## 13.1. MODWT Multi-scale Representation

Mục tiêu:

* Tách chuỗi thời gian thành các thành phần theo horizon
* Giữ thông tin ngắn hạn, trung hạn và dài hạn
* Cung cấp đầu vào đa tần số cho mô hình Neural Granger

Frequency bands dự kiến:

| Scale | Horizon gần đúng | Ý nghĩa |
| ----- | ---------------- | ------- |
| D1 | 2–4 ngày | Biến động rất ngắn hạn |
| D2 | 4–8 ngày | Biến động ngắn hạn |
| D3 | 8–16 ngày | Biến động trung hạn |
| A3 | Trên 16 ngày | Xu hướng dài hạn |

## 13.2. Wavelet Coherence

Mục tiêu:

* Quan sát đồng biến động giữa VN-Index và các biến toàn cầu
* Hỗ trợ giải thích kết quả graph
* Kiểm tra lead-lag pattern theo thời gian và tần số
* So sánh các giai đoạn Pre-COVID, COVID và Post-COVID

Output:

* Heatmap coherence theo thời gian–tần số
* Phase difference / lead-lag visualization
* So sánh coherence trong các event windows

## 13.3. VAR Granger Baseline

VAR Granger được dùng làm baseline tuyến tính.

Vai trò:

* Tạo graph tuyến tính để so sánh với Neural Granger
* Đánh giá xem mô hình phi tuyến có phát hiện thêm quan hệ có ý nghĩa không
* Là baseline dễ giải thích với hội đồng

## 13.4. LSTM Forecasting Baseline

LSTM được dùng làm baseline deep learning dự báo thuần túy.

Vai trò:

* Kiểm tra khả năng dự báo của mô hình học sâu thông thường
* So sánh với Neural Granger về khả năng diễn giải
* Làm rõ lợi ích của sparse causal structure

## 13.5. Neural Granger Model

Kiến trúc tổng quát:

```text
Lagged Multivariate Time Series
        ↓
LSTM / cLSTM Encoder
        ↓
Prediction Layer
        ↓
Group Lasso Regularization
        ↓
Sparse Causal Matrix
```

Ý tưởng:

* Mỗi biến mục tiêu có một mô hình dự báo riêng hoặc một kiến trúc multi-output
* Đầu vào là quá khứ của tất cả các biến
* Group lasso ép mô hình loại bỏ các biến không đóng góp dự báo
* Quan hệ còn lại được biểu diễn dưới dạng directed graph

## 13.6. Wavelet + Neural Granger

Đây là mô hình đề xuất chính.

Quy trình:

```text
Raw series
   ↓
MODWT decomposition
   ↓
Frequency-specific components
   ↓
Neural Granger for each frequency band
   ↓
Frequency-specific causal graphs
```

Mô hình cho phép trả lời:

* Quan hệ nào tồn tại ở ngắn hạn?
* Quan hệ nào chỉ xuất hiện ở dài hạn?
* Graph có thay đổi theo frequency band không?
* Biểu diễn wavelet có làm graph ổn định hơn không?

## 13.7. NOTEARS / PCMCI như mô hình mở rộng

NOTEARS hoặc PCMCI có thể được dùng như baseline mở rộng nếu còn thời gian.

Vai trò:

* So sánh với một phương pháp causal discovery khác
* Kiểm tra tính nhất quán của graph học được
* Không bắt buộc là phần lõi của khóa luận

---

# 14. Thiết kế thực nghiệm

## 14.1. Các mô hình so sánh

| Nhóm | Mô hình | Vai trò |
| ---- | ------ | ------- |
| Correlation baseline | Pearson / Spearman | Baseline không hướng |
| Statistical baseline | VAR Granger | Baseline tuyến tính có hướng |
| Deep learning baseline | LSTM | Baseline dự báo phi tuyến |
| Proposed model 1 | Neural Granger | Học quan hệ phi tuyến |
| Proposed model 2 | Wavelet + Neural Granger | Mô hình đề xuất chính |
| Optional | NOTEARS / PCMCI | Causal discovery baseline mở rộng |

## 14.2. Thiết kế train/validation/test

Nguyên tắc:

* Không shuffle dữ liệu chuỗi thời gian
* Chia theo thứ tự thời gian
* Chọn hyperparameter trên validation set
* Báo cáo kết quả cuối trên test set

## 14.3. Rolling Window Experiment

Để đánh giá tính động của graph, sử dụng rolling window:

```text
Window size: 252 trading days
Step size: 20 trading days
Output: sequence of adjacency matrices
```

Mục tiêu:

* Theo dõi graph thay đổi theo thời gian
* Đo graph stability
* So sánh graph trong giai đoạn ổn định và khủng hoảng

## 14.4. Frequency-specific Experiment

Chạy mô hình riêng trên từng thành phần wavelet:

| Experiment | Input | Output |
| ---------- | ----- | ------ |
| Raw Neural Granger | Log-return gốc | Một graph tổng hợp |
| D1 Neural Granger | Thành phần D1 | Graph ngắn hạn |
| D2 Neural Granger | Thành phần D2 | Graph ngắn hạn/trung hạn |
| D3 Neural Granger | Thành phần D3 | Graph trung hạn |
| A3 Neural Granger | Thành phần A3 | Graph dài hạn |

## 14.5. Event Window Experiment

So sánh graph trước, trong và sau các sự kiện lớn.

Output:

* Số cạnh trong graph
* Graph density
* Vai trò của từng node
* Edge weight đến VN-Index
* Graph-based risk index

---

# 15. Đánh giá mô hình

## 15.1. Forecasting Metrics

| Metric | Ý nghĩa |
| ------ | ------- |
| MAE | Sai số tuyệt đối trung bình |
| RMSE | Sai số bình phương trung bình |
| MAPE | Sai số phần trăm, dùng cẩn thận với return gần 0 |
| Directional Accuracy | Tỷ lệ dự đoán đúng chiều tăng/giảm |

## 15.2. Graph Metrics

| Metric | Ý nghĩa |
| ------ | ------- |
| Graph density | Mức độ dày của graph |
| Number of edges | Số quan hệ được phát hiện |
| In-degree | Mức độ nhận ảnh hưởng |
| Out-degree | Mức độ truyền ảnh hưởng |
| Weighted degree | Mức độ ảnh hưởng có xét trọng số |
| PageRank | Mức độ trung tâm |
| Betweenness | Vai trò trung gian lan truyền |

## 15.3. Stability Metrics

| Metric | Ý nghĩa |
| ------ | ------- |
| Jaccard similarity | Độ giống nhau giữa edge sets của hai graph |
| Edge persistence | Tần suất một edge xuất hiện qua rolling windows |
| Rank correlation | Độ ổn định thứ hạng centrality |

## 15.4. Interpretability Evaluation

Đánh giá khả năng diễn giải bằng:

* Graph có thưa và dễ hiểu không?
* Các cạnh quan trọng có phù hợp với bối cảnh sự kiện không?
* Mô hình có chỉ ra được transmitter và receiver không?
* Kết quả có nhất quán giữa wavelet coherence và causal graph không?

## 15.5. Risk Detection Metrics

Nếu xây dựng risk regime, có thể đánh giá bằng:

| Metric | Ý nghĩa |
| ------ | ------- |
| Precision | Tỷ lệ cảnh báo đúng trong các cảnh báo |
| Recall | Tỷ lệ phát hiện đúng giai đoạn rủi ro |
| F1-score | Cân bằng precision và recall |
| AUC | Khả năng phân biệt risk / non-risk |

Nhãn risk có thể được xây dựng dựa trên volatility của VN-Index vượt percentile 90 hoặc 95.

---

# 16. Ablation Study và Robustness Check

## 16.1. Ablation Study

| Thí nghiệm | Mục đích |
| ---------- | -------- |
| Neural Granger không wavelet | Kiểm tra mô hình gốc |
| Wavelet + Neural Granger | Kiểm tra lợi ích của multi-scale representation |
| Bỏ group lasso | Kiểm tra vai trò của sparse regularization |
| Thay lag length | Kiểm tra độ nhạy temporal dependency |
| Bỏ từng biến đầu vào | Kiểm tra vai trò của từng biến tài chính |
| Chạy trên return và volatility | Kiểm tra dạng dữ liệu nào phù hợp hơn |

## 16.2. Robustness Check

| Kiểm tra | Mục đích |
| -------- | -------- |
| Thay wavelet basis | Kiểm tra độ nhạy với loại wavelet |
| Thay window size | Kiểm tra độ ổn định rolling graph |
| Thay regularization strength | Kiểm tra độ thưa của graph |
| Thay biến đại diện Trung Quốc | SSE vs CSI300 |
| Thay biến đại diện thị trường Mỹ | S&P 500 vs Nasdaq |
| Thay ngưỡng risk label | Percentile 90 vs 95 |

---

# 17. Phân tích kết quả theo giai đoạn sự kiện

## 17.1. Pre-COVID, COVID và Post-COVID

So sánh graph ở ba giai đoạn:

| Giai đoạn | Mục đích |
| --------- | -------- |
| Pre-COVID | Graph trong trạng thái tương đối bình thường |
| COVID | Graph trong cú sốc hệ thống |
| Post-COVID | Graph sau cú sốc, có tightening và địa chính trị |

## 17.2. Event Window Analysis

Mỗi event sẽ được phân tích theo các khía cạnh:

* Wavelet coherence thay đổi như thế nào?
* Graph density có tăng không?
* Node nào trở thành transmitter chính?
* Edge nào đến VN-Index trở nên mạnh hơn?
* Risk index có tăng trước hoặc trong event không?

## 17.3. Event Outputs

Mỗi event nên có:

* Biểu đồ giá/return/volatility
* Wavelet coherence plot
* Graph trước và trong event
* Bảng graph metrics
* Biểu đồ risk index
* Nhận xét định tính ngắn gọn

---

# 18. Graph-based Risk Index

## 18.1. Mục tiêu

Graph-based Risk Index được xây dựng để chuyển kết quả mô hình thành một chỉ số dễ diễn giải hơn.

Chỉ số này không nên được trình bày như một hệ thống cảnh báo tài chính hoàn chỉnh, mà là một minh họa cho khả năng ứng dụng của graph học được.

## 18.2. Thành phần đề xuất

Risk Index có thể kết hợp:

| Thành phần | Ý nghĩa |
| ---------- | ------- |
| Incoming weight to VN-Index | Mức độ VN-Index nhận ảnh hưởng |
| Graph density | Mức độ kết nối toàn hệ thống |
| VIX out-degree / centrality | Vai trò của risk sentiment |
| Edge persistence | Độ bền của các quan hệ rủi ro |
| Volatility percentile | Mức độ biến động hiện tại |

## 18.3. Công thức minh họa

```text
RiskIndex_t = α * IncomingRisk_t
            + β * GraphDensity_t
            + γ * VIXCentrality_t
            + δ * VolatilityPercentile_t
```

Trong đó `α, β, γ, δ` có thể được chọn bằng:

* Quy tắc heuristic có giải thích
* Grid search trên validation set
* Logistic regression nếu có nhãn risk regime

## 18.4. Risk Regimes

| Regime | Ý nghĩa |
| ------ | ------- |
| Low Risk | Thị trường tương đối ổn định |
| Medium Risk | Có dấu hiệu tăng rủi ro |
| High Risk | Biến động mạnh |
| Extreme Risk | Rủi ro lan truyền rõ rệt |

## 18.5. Lưu ý diễn giải

Risk Index là sản phẩm phụ trợ để minh họa khả năng ứng dụng của framework, không phải khuyến nghị đầu tư.

---

# 19. Dashboard minh họa kết quả

## 19.1. Mục tiêu

Dashboard giúp trực quan hóa kết quả của framework, hỗ trợ người đọc hiểu graph, risk index và sự thay đổi quan hệ theo thời gian.

## 19.2. Công nghệ

| Thành phần | Công nghệ |
| ---------- | --------- |
| Web app | Streamlit |
| Interactive plots | Plotly |
| Graph visualization | NetworkX / PyVis |
| Data processing | Pandas, NumPy |
| Model output | PyTorch checkpoints, CSV/JSON graph files |

## 19.3. Features tối thiểu

* Chọn giai đoạn thời gian
* Chọn frequency band
* Hiển thị directed graph
* Hiển thị bảng graph metrics
* Hiển thị risk index theo thời gian
* So sánh graph giữa hai giai đoạn

## 19.4. Features mở rộng

* Event replay
* Edge filtering theo threshold
* Xuất hình graph
* Telegram bot gửi cảnh báo mẫu

Telegram bot chỉ nên để ở mức mở rộng, không phải mục tiêu chính.

---

# 20. Cấu trúc khóa luận

## Chương 1 — Giới thiệu

* Bối cảnh bài toán
* Động cơ nghiên cứu
* Định hướng AI-first
* Mục tiêu nghiên cứu
* Câu hỏi nghiên cứu
* Đóng góp dự kiến
* Phạm vi và giới hạn

## Chương 2 — Cơ sở lý thuyết và nghiên cứu liên quan

* Chuỗi thời gian đa biến
* Granger Causality
* Neural Granger Causality
* Wavelet decomposition và MODWT
* Causal discovery
* Sparse learning và group lasso
* Graph representation
* Financial spillover như miền ứng dụng
* Tổng quan nghiên cứu liên quan

## Chương 3 — Framework đề xuất

* Phát biểu bài toán
* Kiến trúc tổng thể framework
* Tiền xử lý dữ liệu
* Multi-scale representation bằng MODWT
* Neural Granger model
* Sparse graph construction
* Graph-based risk index
* Dashboard visualization

## Chương 4 — Thực nghiệm và đánh giá

* Mô tả dataset
* Thiết lập thực nghiệm
* Baseline models
* Evaluation metrics
* Forecasting results
* Graph results
* Frequency-specific analysis
* Rolling window analysis
* Event window analysis
* Ablation study
* Robustness check

## Chương 5 — Kết luận và hướng phát triển

* Tổng kết kết quả chính
* Đóng góp của khóa luận
* Hạn chế
* Hướng phát triển tiếp theo

---

# 21. Checklist triển khai

## Giai đoạn 1 — Data Engineering

* [ ] Xác định danh sách biến cuối cùng
* [ ] Crawl/tải dữ liệu từ nguồn phù hợp
* [ ] Đồng bộ ngày giao dịch
* [ ] Xử lý missing values
* [ ] Tính log-return
* [ ] Tính volatility proxy
* [ ] Kiểm định ADF/KPSS
* [ ] Chuẩn hóa dữ liệu
* [ ] Chia train/validation/test

## Giai đoạn 2 — Wavelet Representation

* [ ] Cài đặt MODWT decomposition
* [ ] Phân rã các chuỗi thành D1, D2, D3, A3
* [ ] Lưu dữ liệu theo từng frequency band
* [ ] Vẽ wavelet decomposition plots
* [ ] Chạy wavelet coherence cho các cặp chính
* [ ] Phân tích lead-lag pattern

## Giai đoạn 3 — Baseline Models

* [ ] Pearson/Spearman correlation baseline
* [ ] VAR Granger baseline
* [ ] LSTM forecasting baseline
* [ ] Lưu kết quả baseline
* [ ] Xây dựng bảng so sánh ban đầu

## Giai đoạn 4 — Neural Granger Model

* [ ] Xây dựng dataset dạng lagged window
* [ ] Cài đặt LSTM/cLSTM model
* [ ] Cài đặt group lasso regularization
* [ ] Huấn luyện Neural Granger trên raw series
* [ ] Huấn luyện Neural Granger trên từng frequency band
* [ ] Trích xuất adjacency matrix
* [ ] Render graph

## Giai đoạn 5 — Evaluation

* [ ] Tính MAE/RMSE/Directional Accuracy
* [ ] Tính graph density, in-degree, out-degree
* [ ] Tính PageRank, betweenness
* [ ] Tính graph stability bằng Jaccard similarity
* [ ] So sánh baseline và mô hình đề xuất
* [ ] Thực hiện ablation study
* [ ] Thực hiện robustness check

## Giai đoạn 6 — Event Analysis và Risk Index

* [ ] Xác định event windows
* [ ] So sánh graph trước/trong/sau event
* [ ] Xây dựng Graph-based Risk Index
* [ ] Tạo risk regime labels nếu cần
* [ ] Đánh giá risk detection metrics
* [ ] Viết nhận xét kết quả

## Giai đoạn 7 — Dashboard và báo cáo

* [ ] Xây dựng Streamlit dashboard
* [ ] Thêm graph visualization
* [ ] Thêm risk index chart
* [ ] Thêm frequency selector
* [ ] Viết Chương 1–5
* [ ] Tạo bảng kết quả và hình minh họa
* [ ] Chuẩn bị slide bảo vệ

---

# 22. Quản trị rủi ro dự án

| Rủi ro | Tác động | Giải pháp |
| ------ | -------- | --------- |
| Dữ liệu thiếu hoặc lệch ngày | Kết quả sai lệch | Dùng nhiều nguồn, align cẩn thận, ghi rõ policy |
| Mô hình Neural Granger khó train | Kết quả không ổn định | Bắt đầu từ kiến trúc đơn giản, early stopping, seed cố định |
| Graph quá dày | Khó diễn giải | Tăng regularization, threshold edge weight |
| Graph quá thưa | Mất quan hệ quan trọng | Giảm regularization, kiểm tra lag length |
| Wavelet khó diễn giải | Kết quả mơ hồ | Gắn frequency band với horizon cụ thể |
| Evaluation thiếu thuyết phục | Khóa luận yếu | Chuẩn bị baseline, ablation và robustness check |
| Dashboard mất nhiều thời gian | Chậm tiến độ | Chỉ làm dashboard tối thiểu |
| NOTEARS không tích hợp tốt | Gây loãng đề tài | Đưa NOTEARS vào optional extension |

---

# 23. Timeline hoàn thành

| Giai đoạn | Thời gian dự kiến | Kết quả cần có |
| --------- | ----------------- | -------------- |
| Data pipeline | 1–2 tuần | Dataset sạch, log-return, volatility |
| Wavelet analysis | 1 tuần | MODWT components, WTC plots |
| Baseline models | 1 tuần | Pearson, VAR, LSTM baseline |
| Neural Granger | 2–3 tuần | Causal graphs, adjacency matrices |
| Evaluation & ablation | 1–2 tuần | Bảng kết quả, robustness check |
| Event analysis & risk index | 1 tuần | Event comparison, risk index plots |
| Dashboard | 1 tuần | Streamlit prototype |
| Viết khóa luận | 2–3 tuần | Bản thảo Chương 1–5 |
| Slide & defense | 1 tuần | Slide, script thuyết trình, câu hỏi dự kiến |

---

# 24. Tài liệu tham khảo cốt lõi

## 24.1. Time Series và Granger Causality

* Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods.
* Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.

## 24.2. Neural Granger và Deep Learning cho chuỗi thời gian

* Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. (2021). Neural Granger Causality.
* Khanna, S., & Tan, V. Y. F. (2020). Economy Statistical Recurrent Units for Inferring Nonlinear Granger Causality.
* Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017). A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction.

## 24.3. Wavelet Analysis

* Torrence, C., & Compo, G. P. (1998). A Practical Guide to Wavelet Analysis.
* Percival, D. B., & Walden, A. T. (2000). Wavelet Methods for Time Series Analysis.
* Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of the cross wavelet transform and wavelet coherence.

## 24.4. Causal Discovery

* Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
* Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS.
* Runge, J. et al. PCMCI and causal discovery for time series.

## 24.5. Financial Spillover và Connectedness

* Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers.
* Forbes, K. J., & Rigobon, R. (2002). No contagion, only interdependence.
* Bekaert, G., Hoerova, M., & Lo Duca, M. (2013). Risk, uncertainty and monetary policy.
* Baur, D. G., & Lucey, B. M. (2010). Is gold a hedge or a safe haven?

---

# 25. Định hướng mở rộng sau khóa luận

## 25.1. Dynamic Graph Neural Networks

Có thể mở rộng framework sang:

* Temporal Graph Networks
* Dynamic Graph Neural Networks
* Graph Attention Networks cho time series
* EvolveGCN hoặc TGAT

## 25.2. Causal Representation Learning

Thay vì dùng wavelet cố định, có thể học biểu diễn đa tần số bằng:

* Temporal convolution
* Attention-based encoder
* Contrastive learning cho chuỗi thời gian
* Self-supervised time series representation learning

## 25.3. Macro-financial Integration

Mở rộng thêm các biến vĩ mô:

* CPI
* Lãi suất điều hành
* USD/VND
* Bond yields
* Dòng vốn ngoại
* Chỉ số sản xuất công nghiệp

## 25.4. ASEAN Financial Network

Mở rộng graph sang các thị trường Đông Nam Á:

* Việt Nam
* Thái Lan
* Indonesia
* Malaysia
* Singapore
* Philippines

## 25.5. Real-time Risk Monitoring

Phát triển thành hệ thống gần thời gian thực:

* Streaming data pipeline
* Online model updating
* Real-time graph update
* Telegram/Email alert
* Model monitoring dashboard

---

# Kết luận định hướng

Khóa luận nên được định vị là một đề tài Trí tuệ nhân tạo ứng dụng trên dữ liệu tài chính, trong đó trọng tâm là xây dựng framework học máy nhân quả đa tần số cho chuỗi thời gian đa biến.

Cách framing phù hợp nhất là:

```text
Financial time series
    → Multi-scale representation by MODWT
    → Neural Granger causal discovery
    → Sparse directed graph
    → Graph-based risk interpretation
    → Evaluation against statistical and deep learning baselines
```

Với định hướng này, tài chính đóng vai trò là miền ứng dụng thực tế, còn đóng góp chính của khóa luận nằm ở mô hình AI, thiết kế thực nghiệm, đánh giá mô hình và khả năng diễn giải kết quả.

Nếu triển khai đúng trọng tâm, đề tài có thể đáp ứng tốt yêu cầu của ngành Trí tuệ nhân tạo, đồng thời vẫn có giá trị ứng dụng trong phân tích rủi ro tài chính.
