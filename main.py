from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np


with open("diagnostic_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentData(BaseModel):
    video_time: float
    text_time: float
    interactive_time: float
    test_score: float
    avg_response_time: float

@app.get("/")
def read_root():
    return {"message": "EduZone AI Diagnostika API ishlayapti!"}

@app.post("/predict")
def predict_learning_style(data: StudentData):
    input_data = np.array([
        data.video_time,
        data.text_time,
        data.interactive_time,
        data.test_score,
        data.avg_response_time
    ]).reshape(1, -1)
    
    prediction = model.predict(input_data)
    return {"learning_style": prediction[0]}
