# AI-first Financial Time Series Causal Discovery Thesis Project

Bộ code này được viết lại theo hướng **khóa luận ngành Trí tuệ nhân tạo**: tài chính là miền ứng dụng, còn trọng tâm là mô hình học máy cho chuỗi thời gian.

## Cấu trúc chính

```text
finance_ai_project/
├── config.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── common.py
│   ├── metrics.py
│   ├── neural_granger.py
│   └── wavelet_utils.py
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_validate_data.py
│   ├── 02_build_features.py
│   ├── 03_eda_stationarity.py
│   ├── 04_modwt_multiscale.py
│   ├── 05_wavelet_coherence.py
│   ├── 06_var_granger_baseline.py
│   ├── 07_lstm_forecasting_baseline.py
│   ├── 08_neural_granger.py
│   ├── 09_graph_analysis.py
│   ├── 10_event_window_analysis.py
│   ├── 11_graph_risk_index.py
│   └── 12_ablation_robustness.py
├── app/
│   └── 13_dashboard_streamlit.py
└── run_pipeline.py
```

## Thứ tự chạy

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy từng phần:

```bash
python scripts/00_check_environment.py
python scripts/01_validate_data.py
python scripts/02_build_features.py
python scripts/03_eda_stationarity.py
python scripts/04_modwt_multiscale.py
python scripts/05_wavelet_coherence.py
python scripts/06_var_granger_baseline.py
python scripts/07_lstm_forecasting_baseline.py
python scripts/08_neural_granger.py
python scripts/09_graph_analysis.py
python scripts/10_event_window_analysis.py
python scripts/11_graph_risk_index.py
python scripts/12_ablation_robustness.py
```

Hoặc chạy toàn bộ:

```bash
python run_pipeline.py
```

Mở dashboard:

```bash
streamlit run app/13_dashboard_streamlit.py
```

## Ghi chú học thuật

- `04_modwt_multiscale.py` dùng SWT của PyWavelets như một xấp xỉ thực dụng cho MODWT.
- `05_wavelet_coherence.py` hiện có fallback bằng rolling multi-scale correlation để tránh phụ thuộc nặng. Nếu cần wavelet coherence đúng nghĩa, có thể tích hợp `pycwt.wct` từ script cũ.
- `08_neural_granger.py` dùng Lagged MLP + group lasso để học quan hệ Granger phi tuyến dạng sparse graph. Đây là phiên bản nhẹ, dễ chạy và dễ giải thích cho khóa luận tốt nghiệp.
- `11_graph_risk_index.py` đặt tên là **Graph-based Risk Index**, tránh khẳng định quá mạnh là hệ thống cảnh báo đầu tư.

## Đầu ra quan trọng

- `results/tables/06_var_granger_adjacency.csv`
- `results/tables/07_lstm_metrics.csv`
- `results/tables/08_neural_granger_adjacency_thresholded.csv`
- `results/tables/08_neural_granger_forecast_metrics.csv`
- `results/tables/09_graph_node_metrics.csv`
- `results/tables/10_event_window_summary.csv`
- `results/tables/11_graph_based_risk_index.csv`
- `results/figures/09_neural_granger_graph.png`
- `results/figures/11_graph_based_risk_index.png`
=======
# financial_causal_risk
>>>>>>> f014b6186a834a8b2900c7d952f1f67c8f6c3210
