import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import tensorflow as tf
from tensorflow.keras import models, layers
import os

# Contoh dataset dummy
def load_dataset():
    # Ganti dengan dataset mental health kamu
    data = pd.read_csv("dataset.csv")  # misal file kamu bernama dataset.csv
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y

def train_sklearn_models(X, y):
    models_scores = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifiers = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "SVC": SVC(probability=True)
    }

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        models_scores[name] = (model, score)
        print(f"{name} accuracy: {score:.4f}")

    return models_scores

def train_tensorflow_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow model accuracy: {acc:.4f}")
    return model, acc

def save_best_model(sklearn_models, tf_model, tf_score):
    # Bandingkan skor dan simpan model terbaik
    best_model = None
    best_score = -1
    best_name = ""

    for name, (model, score) in sklearn_models.items():
        if score > best_score:
            best_model = model
            best_score = score
            best_name = name
            joblib.dump(model, "app/models/sklearn_model.pkl")

    if tf_score > best_score:
        best_model = tf_model
        best_score = tf_score
        best_name = "TensorFlow"
        tf_model.save("app/models/tf_model.h5")

    print(f"Best model: {best_name} with accuracy {best_score:.4f}")

if __name__ == "__main__":
    X, y = load_dataset()
    sk_models = train_sklearn_models(X, y)
    tf_model, tf_score = train_tensorflow_model(X, y)
    save_best_model(sk_models, tf_model, tf_score)
