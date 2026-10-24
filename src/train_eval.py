import os
import yaml
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from extract_color_texture import extract_features
from extract_HOG import extract_Hog

def extract_features_and_labels(df, images_dir):
    """
    Преобразует изображения в табличный формат (векторы) и извлекает метки.
    """
    X = []
    
    # Определение формата меток (Plant Pathology 2020 vs 2021)
    if 'labels' in df.columns:
        # Для датасета 2021 года (одна колонка с текстом)
        y = df['labels'].values
    else:
        # Для датасета 2020 года (one-hot кодирование в 4 колонках)
        y_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
        # Превращаем 4 колонки в один индекс (0, 1, 2, 3)
        y = np.argmax(df[y_cols].values, axis=1)
        
    print(f"Чтение и преобразование {len(df)} изображений из {images_dir}...")
    
    for image_id in df['image_id']:
        img_path = os.path.join(images_dir, f"{image_id}.jpg")
        
        # Открываем изображение и гарантируем формат RGB
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        
        # === ПРЕОБРАЗОВАНИЕ В ТАБЛИЧНЫЙ ВИД ===
        # Вытягиваем массив: (128, 128, 3) -> (49152,)
        flat_array = img_array.flatten()
        
        # Нормализация: масштабируем значения пикселей от 0..255 к 0..1
        # Это критически важно для работы SVM и KNN
        flat_array = flat_array / 255.0
        
        X.append(flat_array)
        
    return np.array(X), y

if __name__ == "__main__":
    # 1. Загрузка параметров
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model_name = params["train"]["model_name"]
    output_dir = "data/features"
    seed = params["data"]["seed"]
    
    # 2. Загрузка признаков и меток
    print("Загрузка признаков ResNet50...")
    X_train = np.load(f"{output_dir}/X_train_resnet.npy")
    y_train = np.load(f"{output_dir}/y_train_resnet.npy")
    X_test = np.load(f"{output_dir}/X_test_resnet.npy")
    y_test = np.load(f"{output_dir}/y_test_resnet.npy")
    
    # 3. Инициализация модели
    print(f"Инициализация модели: {model_name}")
    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "svm":
        model = SVC(kernel="rbf", C=1.0, random_state=42)
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        
    # 4. Обучение
    print("Начало обучения (это может занять время)...")
    model.fit(X_train, y_train)
    
    # 5. Предсказание и расчет метрик
    print("Получение предсказаний и расчет метрик...")
    preds = model.predict(X_test)
    
    # Используем average='weighted' для мультиклассовой классификации
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds, average="weighted")),
        "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, preds, average="weighted", zero_division=0))
    }
    
    print(f"Результаты: {metrics}")
    
    # 6. Сохранение метрик для DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)