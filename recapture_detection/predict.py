import sys
from features import extract_all_features
from utils import load_model

def predict(image_path):
    model, scaler, _ = load_model()
    features = extract_all_features(image_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
   
    score = model.predict_proba(features_scaled)[0, 1]
    
    
    print(f"-> {score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py some_image.jpg")
    else:
        predict(sys.argv[1])