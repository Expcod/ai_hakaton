import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Dataset yaratish (sun'iy)
data = {
    "video_time": [45, 10, 15, 60, 5, 30, 40],
    "text_time": [10, 50, 20, 5, 60, 20, 10],
    "interactive_time": [15, 5, 60, 20, 10, 40, 30],
    "test_score": [85, 70, 88, 90, 60, 75, 80],
    "avg_response_time": [12, 15, 9, 11, 14, 13, 10],
    "subject_focus": ["Math", "Biology", "Chemistry", "Math", "Biology", "Physics", "History"],
    "learning_style": ["visual", "reading", "kinesthetic", "visual", "reading", "kinesthetic", "visual"]
}

df = pd.DataFrame(data)

# 2. Kategorik ustunni kodlash
df = pd.get_dummies(df, columns=["subject_focus"])

# 3. Modelga tayyorlash
X = df.drop("learning_style", axis=1)
y = df["learning_style"]

# 4. Train/test bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Model yaratish
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Baholash
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
