import pandas as pd
import pickle

with open("model/python_Ai_demo_model.pkl", "rb") as file:
    model = pickle.load(file)

MODEL_VERSION = "1.0.0"

def get_prediction(data: dict):
    user_data = pd.DataFrame([data])
    return model.predict(user_data)[0]
