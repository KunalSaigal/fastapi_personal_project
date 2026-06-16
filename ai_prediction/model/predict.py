import pandas as pd
import pickle

with open("ai_prediction/model/python_Ai_demo_model.pkl", "rb") as file:
    model = pickle.load(file)

MODEL_VERSION = "1.0.0"

class_labels = model.classes_.tolist()

def get_prediction(data: dict):
    user_data = pd.DataFrame([data])

    # Predict the class label
    prediction = model.predict(user_data)[0]

    # Predict the probability of the class label
    probability = model.predict_proba(user_data)[0]

    # Calculate the confidence of the prediction
    confidence = max(probability)

    # Create a dictionary of the class labels and their probabilities
    class_probability = dict(zip(class_labels, map(lambda p: round(p,4), probability)))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "class_probability": class_probability
    }
