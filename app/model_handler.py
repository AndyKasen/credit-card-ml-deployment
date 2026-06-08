#=====================================================
# Модуль для загрузки модели.
#=====================================================
import joblib
import numpy as np
import os

# Глобальные переменные для модели и scaler'а
_model = None
_scaler = None


def loadModelScaler():

    # Загружает модель и scaler из файлов проекта.
    
    global _model, _scaler
    
    modelPath = os.path.join('models', 'model_v1.pkl')
    scalerPath = os.path.join('models', 'scaler.pkl')
    
    _model = joblib.load(modelPath)
    _scaler = joblib.load(scalerPath)
    
    print("Модель и scaler успешно загружены")


def predict(features: list) -> dict:
    """
    Принимает список признаков клиента, преобразует его в массив numpy.
    Возвращает словарь с прогнозом (0/1) и вероятностью дефолта.
    
    Аргументы:
        features: список из 23 чисел (те же признаки, что и при обучении)
    
    Возвращает:
        dict: {'prediction': int, 'probability': float}
    """
    if _model is None or _scaler is None:
        raise RuntimeError("Модель не загружена. Сначала вызовите loadModelScaler()")
    
    X = np.array(features).reshape(1, -1)
    
    X_scaled = _scaler.transform(X)
    
    # Предсказание класса (0 или 1)
    prediction = int(_model.predict(X_scaled)[0])
    
    # Вероятность дефолта 
    probability = float(_model.predict_proba(X_scaled)[0, 1])
    
    return {
        'prediction': prediction,
        'probability': probability
    }