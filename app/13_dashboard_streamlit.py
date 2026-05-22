from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import TABLE_DIR, FIGURE_DIR

st.set_page_config(page_title="AI-first Financial Risk Dashboard", layout="wide")
st.title("Wavelet-based Neural Causal Discovery Dashboard")
st.caption("Prototype minh họa kết quả khóa luận AI-first: graph, risk index, event windows.")

risk_path = TABLE_DIR / "11_graph_based_risk_index.csv"
if risk_path.exists():
    risk = pd.read_csv(risk_path, parse_dates=["date"])
    st.subheader("Graph-based Risk Index")
    fig = px.line(risk, x="date", y="graph_based_risk_index", color="risk_regime")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chưa có risk index. Hãy chạy scripts/11_graph_risk_index.py")

node_path = TABLE_DIR / "09_graph_node_metrics.csv"
if node_path.exists():
    nodes = pd.read_csv(node_path)
    st.subheader("Graph node metrics")
    st.dataframe(nodes, use_container_width=True)
    fig = px.bar(nodes, x="node", y="net_transmitter_score", title="Net transmitter score")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chưa có graph metrics. Hãy chạy scripts/09_graph_analysis.py")

event_path = TABLE_DIR / "10_event_window_summary.csv"
if event_path.exists():
    ev = pd.read_csv(event_path)
    st.subheader("Event window comparison")
    st.dataframe(ev, use_container_width=True)
    fig = px.bar(ev, x="event", y=["graph_density", "incoming_to_vni", "outgoing_from_vni"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chưa có event analysis. Hãy chạy scripts/10_event_window_analysis.py")

img = FIGURE_DIR / "09_neural_granger_graph.png"
if img.exists():
    st.subheader("Neural Granger Graph")
    st.image(str(img))
