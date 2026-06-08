#=====================================================
# Flask-приложение: веб-сервис прогнозирования дефолта
#=====================================================
from flask import Flask, request, jsonify
from app.model_handler import loadModelScaler, predict
import logging
import json
import os
from datetime import datetime

# Логи пишутся в эту папку
os.makedirs('logs', exist_ok=True)

# В этой строчке настройка логгера
logger = logging.getLogger('credit_default_api')
logger.setLevel(logging.INFO)

# Обработчик записи в лог-файл
fileHandler = logging.FileHandler('logs/api.log')
fileHandler.setLevel(logging.INFO)

# Обработчик вывода логов в терминал
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)

# Формат сообщений
formatter = logging.Formatter('%(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

app = Flask(__name__)

# Загрузка модели при старте
loadModelScaler()

# Лог: сервис запущен
logger.info(json.dumps({
    'event': 'startup',
    'message': 'Модель и scaler загружены',
    'timestamp': datetime.now().isoformat()
}))


@app.route('/predict', methods=['POST'])
def predictEndpoint():
    """Эндпоинт для предсказания дефолта"""
    data = request.get_json()
    
    # Лог: входящий запрос
    featuresCount = len(data.get('features', [])) if data else 0
    logger.info(json.dumps({
        'event': 'request',
        'endpoint': '/predict',
        'features_count': featuresCount,
        'timestamp': datetime.now().isoformat()
    }))
    
    # Проверка списка, должен быть ключ 'features'
    if not data or 'features' not in data:
        logger.warning(json.dumps({
            'event': 'error',
            'error': 'Missing features key',
            'timestamp': datetime.now().isoformat()
        }))
        return jsonify({'error': 'Ожидается JSON с ключом "features"'}), 400
    
    features = data['features']
    
    # Проверка количества признаков, должно быть 23
    if len(features) != 23:
        logger.warning(json.dumps({
            'event': 'error',
            'error': f'Expected 23 features, got {len(features)}',
            'timestamp': datetime.now().isoformat()
        }))
        return jsonify({'error': f'Ожидается 23 признака, получено {len(features)}'}), 400
    
    # Вызов функции предсказания из model_handler
    result = predict(features)
    
    # Лог: результат предсказания
    logger.info(json.dumps({
        'event': 'prediction',
        'prediction': result['prediction'],
        'probability': round(result['probability'], 4),
        'timestamp': datetime.now().isoformat()
    }))
    
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья сервиса"""
    return jsonify({'status': 'ok'}), 200


# Точка входа для запуска сервиса
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)