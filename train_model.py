import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("eduzone_ai_diagnostic_dataset.csv")

X = df[["video_time", "text_time", "interactive_time", "test_score", "avg_response_time"]]
y = df["learning_style"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("diagnostic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model tayyorlandi va saqlandi: diagnostic_model.pkl")
