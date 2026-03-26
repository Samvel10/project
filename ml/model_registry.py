try:
    import joblib
    _HAVE_JOBLIB = True
except ImportError:
    import pickle
    joblib = None  # type: ignore
    _HAVE_JOBLIB = False
import time
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def save_model(model, score):
    ts = int(time.time())
    path = MODEL_DIR / f"model_{ts}_{round(score,4)}.pkl"
    if _HAVE_JOBLIB:
        joblib.dump(model, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return path


def load_latest():
    models = sorted(MODEL_DIR.glob("model_*.pkl"))
    if not models:
        raise FileNotFoundError("No trained models found")
    latest = models[-1]
    if _HAVE_JOBLIB:
        return joblib.load(latest)
    with open(latest, "rb") as f:
        return pickle.load(f)
