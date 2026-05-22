from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "scripts/00_check_environment.py",
    "scripts/01_validate_data.py",
    "scripts/02_build_features.py",
    "scripts/03_eda_stationarity.py",
    "scripts/04_modwt_multiscale.py",
    "scripts/05_wavelet_coherence.py",
    "scripts/06_var_granger_baseline.py",
    "scripts/07_lstm_forecasting_baseline.py",
    "scripts/08_neural_granger.py",
    "scripts/09_graph_analysis.py",
    "scripts/10_event_window_analysis.py",
    "scripts/11_graph_risk_index.py",
    "scripts/12_ablation_robustness.py",
]

for s in SCRIPTS:
    print("\n" + "="*80)
    print("RUN", s)
    print("="*80)
    subprocess.run([sys.executable, str(ROOT / s)], check=True, cwd=ROOT)
print("\n✓ Pipeline completed")
