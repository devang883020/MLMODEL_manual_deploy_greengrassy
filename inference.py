import pickle
import numpy as np
import os
import time
import json
from datetime import datetime

def load_model(model_path="iris_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def run_inference(model):
    # Sample input: sepal length, sepal width, petal length, petal width
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)
    probability = model.predict_proba(sample)
    
    # Map prediction to class names
    class_names = ['setosa', 'versicolor', 'virginica']
    predicted_class = class_names[prediction[0]]
    
    return {
        'prediction': predicted_class,
        'confidence': float(np.max(probability)),
        'probabilities': {
            class_names[i]: float(prob) for i, prob in enumerate(probability[0])
        },
        'timestamp': datetime.now().isoformat()
    }

def continuous_inference(model, interval=30):
    """Run inference continuously every interval seconds"""
    while True:
        try:
            result = run_inference(model)
            print(f"🔄 {result['timestamp']}: Prediction = {result['prediction']}, Confidence = {result['confidence']:.3f}")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("🛑 Stopping continuous inference...")
            break
        except Exception as e:
            print(f"❌ Error during continuous inference: {str(e)}")
            time.sleep(interval)

if __name__ == "__main__":
    try:
        print("📦 Loading model...")
        model = load_model()
        print("🤖 Running initial prediction...")
        result = run_inference(model)
        print(f"✅ Initial Prediction: {json.dumps(result, indent=2)}")
        
        # Run continuous inference
        print("🔄 Starting continuous inference mode...")
        continuous_inference(model)
        
    except Exception as e:
        print(f"❌ Error occurred during inference: {str(e)}")
        exit(1)
