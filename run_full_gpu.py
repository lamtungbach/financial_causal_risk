from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(cmd):
    print("\n" + "="*100)
    print("RUN", " ".join(cmd))
    print("="*100)
    subprocess.run(cmd, cwd=ROOT, check=True)

# Các script CPU/statistical trước.
for script in [
    "scripts/00_check_environment.py",
    "scripts/01_validate_data.py",
    "scripts/02_build_features.py",
    "scripts/03_eda_stationarity.py",
    "scripts/04_modwt_multiscale.py",
    "scripts/05_wavelet_coherence.py",
    "scripts/06_var_granger_baseline.py",
]:
    run([sys.executable, script])

# Full hơn cho deep learning. Có thể tăng epochs/hidden nếu GPU mạnh.
run([sys.executable, "scripts/07_lstm_forecasting_baseline.py", "--device", "cuda", "--lags", "20", "--hidden", "64", "--epochs", "300", "--patience", "30"])
run([sys.executable, "scripts/08_neural_granger.py", "--device", "cuda", "--lags", "20", "--hidden", "64", "--epochs", "300", "--patience", "30", "--lam", "0.0005"])
run([sys.executable, "scripts/08b_neural_granger_multiband.py", "--device", "cuda", "--lags", "20", "--hidden", "64", "--epochs", "300", "--patience", "30", "--lam", "0.0005"])

for script in [
    "scripts/09_graph_analysis.py",
    "scripts/10_event_window_analysis.py",
    "scripts/11_graph_risk_index.py",
    "scripts/12_ablation_robustness.py",
]:
    run([sys.executable, script])

print("\n✓ Full GPU pipeline completed")
