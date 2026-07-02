import os
import glob
import numpy as np
from joblib import Parallel, delayed
from features import extract_all_features
from classifier import RecaptureClassifier
from evaluate import evaluate_pipeline
from utils import setup_logger, save_model
from sklearn.model_selection import train_test_split

logger = setup_logger()

def load_dataset(base_dir="dataset"):
    logger.info("Loading dataset and extracting features...")
    paths = glob.glob(f"{base_dir}/real/*") + glob.glob(f"{base_dir}/screen/*")
    labels = [0]*len(glob.glob(f"{base_dir}/real/*")) + [1]*len(glob.glob(f"{base_dir}/screen/*"))
    
    X = Parallel(n_jobs=-1)(delayed(extract_all_features)(p) for p in paths)
    return np.array(X), np.array(labels)

if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    logger.info("Training classifiers & tuning hyperparameters...")
    clf = RecaptureClassifier()
    clf.train_and_select(X_train, y_train)
    
    logger.info("Evaluating on hidden test split...")
    y_pred, y_probs = clf.predict(X_test)
    
    feat_names = [f"F_{i}" for i in range(X.shape[1])]
    evaluate_pipeline(y_test, y_pred, y_probs, feat_names, clf.best_model)
    
    save_model(clf.best_model, clf.scaler, clf.optimal_threshold)
    logger.info("Model pipeline saved successfully as model.pkl")