import time
import joblib
import logging

def setup_logger(name="RecaptureDetection"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def save_model(model, scaler, threshold, path="model.pkl"):
    joblib.dump({'model': model, 'scaler': scaler, 'threshold': threshold}, path)

def load_model(path="model.pkl"):
    data = joblib.load(path)
    return data['model'], data['scaler'], data['threshold']

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start