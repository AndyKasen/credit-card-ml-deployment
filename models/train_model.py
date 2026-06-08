# models/train_model.py
# EDA

# 1. Импортируем библиотеки
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pickle
import joblib
import os

# 2. Загружаем данные 
# PATH, по которому лежит файл с данными
path = 'data/UCI_Credit_Card.csv'

df = pd.read_csv(path, header=0)
print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПервые 3 строки:")
print(df.head(3))

# 3 Информация о датасете
print(df.info())

# 4. Определяем целевую переменную - дефолт в следующем месяце

targetColumn = 'default.payment.next.month'

# 5. Удаляем ID (это просто номер клиента и не влияет на target)
X = df.drop(columns=[targetColumn, 'ID'])
y = df[targetColumn]

print(f"\nПризнаки (X): {X.shape[1]} столбцов")
print(f"Целевая переменная (y): {y.shape[0]} значений")

# 6. Проверяем классы - сколько дефолтнулось (1), сколько - нет (0)
print("\nКлассы в target:")
print(y.value_counts())
print(f"Доля дефолтов: {y.sum() / len(y) * 100:.1f}%")

# Вывод: классы несбалансированы. Доля дефолтов 22.1%, доля без дефолтов 77.9%. 
# Это важно для оценки модели на accuracy, ROC-AUC, precision, recall.

# 7. Разделяем данные на обучающую (80%) и тестовую (20%) выборки
# stratify=y сохраняем пропорцию классов в обеих выборках (у нас как раз дисбаланс)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазделение выполнено:")
print(f"   Обучающая (train): {X_train.shape[0]} строк")
print(f"   Тестовая (test): {X_test.shape[0]} строк")

# Проверяем пропорции классов
print(f"\nДефолтнулось в обучающей: {y_train.sum() / len(y_train) * 100:.1f}%")
print(f"Дефолтнулось в тестовой: {y_test.sum() / len(y_test) * 100:.1f}%")

# 8. Масштабирование признаков

scaler = StandardScaler()

# Масштабируем обучающие данные
X_trainScaled = scaler.fit_transform(X_train)
# Масштабируем тестовые данные
X_testScaled = scaler.transform(X_test)

print(f"\nМасштабирование выполнено")
print(f"Размер X_trainScaled: {X_trainScaled.shape}")
print(f"Размер X_testScaled: {X_testScaled.shape}")

# 9. Обучение модели Random Forest - выбрал этот алгоритм потому, что он мне нравится и ....

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_trainScaled, y_train)

print(f"\nМодель обучена")

# 10. Оценка качества модели на тестовых данных
y_pred = model.predict(X_testScaled)          
y_predProba = model.predict_proba(X_testScaled)[:, 1]  

# Метрики качества

roc_auc = roc_auc_score(y_test, y_predProba)
print(f"\nROC-AUC: {roc_auc:.4f}")
print("\nКлассификационный отчёт:")
print(classification_report(y_test, y_pred, target_names=['Не дефолтнулся', 'Дефолтнулся']))

print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred))

# 11. Сохраняю модель с масштабированием

import os

os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/model_v1.pkl')
print("Модель сохранена в models/model_v1.pkl")

# Сохраняем scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler сохранён в models/scaler.pkl")