import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = ["numpy", "pandas", "matplotlib", "scipy", "statsmodels", "sklearn", "torch", "networkx", "pywt", "plotly", "streamlit"]
OPTIONAL = ["pycwt", "yfinance", "vnstock"]

print("Project root:", ROOT)
print("\n[Core packages]")
for pkg in REQUIRED:
    name = "PyWavelets" if pkg == "pywt" else pkg
    try:
        m = importlib.import_module(pkg)
        print(f"✓ {name}: {getattr(m, '__version__', 'installed')}")
    except Exception:
        print(f"✗ {name}: missing")
print("\n[Optional packages]")
for pkg in OPTIONAL:
    try:
        m = importlib.import_module(pkg)
        print(f"✓ {pkg}: {getattr(m, '__version__', 'installed')}")
    except Exception:
        print(f"- {pkg}: not installed")
