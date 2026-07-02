import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
class RecaptureClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.optimal_threshold = 0.5
        
        # Constrained hyperparameters for sub-20ms mobile inference
        self.models = {
            'XGB': (XGBClassifier(eval_metric='logloss', random_state=42),
                    {
                        'n_estimators': [50, 100], 
                        'learning_rate': [0.05, 0.1], 
                        'max_depth': [3, 5],
                        'subsample': [0.8]
                    }),
            'RF': (RandomForestClassifier(class_weight='balanced', random_state=42),
                   {
                        'n_estimators': [50, 100], 
                        'max_depth': [5, 10], 
                        'min_samples_leaf': [2, 5]
                   })
        }

    def train_and_select(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        best_score = 0
        best_name = ""
        
        for name, (model, params) in self.models.items():
            grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_scaled, y)
            
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                self.best_model = grid.best_estimator_
                best_name = name
                
        print(f"Selected Best Model: {best_name} (CV ROC-AUC: {best_score:.4f})")
        self._optimize_threshold(X_scaled, y)

    def _optimize_threshold(self, X_scaled, y):
        """Optimizes threshold on training distribution to maximize total Accuracy."""
        probs = self.best_model.predict_proba(X_scaled)[:, 1]
        
        # Force the threshold to stay between 0.40 and 0.70 to stop paranoia
        thresholds = np.arange(0.40, 0.70, 0.02)
        
        # Optimize for Accuracy instead of F1
        acc_scores = [accuracy_score(y, (probs >= t).astype(int)) for t in thresholds]
        self.optimal_threshold = thresholds[np.argmax(acc_scores)]
        
        print(f"Optimized Decision Threshold (Max Accuracy): {self.optimal_threshold:.2f}")
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        probs = self.best_model.predict_proba(X_scaled)[:, 1]
        return (probs >= self.optimal_threshold).astype(int), probs