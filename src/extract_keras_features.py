import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

def extract_keras_features(df, images_dir, model_path, output_x, output_y):
    print(f"Загрузка обученной Keras модели из {model_path}...")
    
    # 1. Загрузка полной сохраненной модели
    full_model = tf.keras.models.load_model(model_path)
    
    # 2. Создание модели-экстрактора
    # Мы берем вход оригинальной модели и направляем его на выход предпоследнего слоя.
    # Предполагается, что full_model.layers[-1] — это Dense(4) классификатор, 
    # а full_model.layers[-2] — это слой GlobalAveragePooling2D (вектор на 2048).
    feature_extractor = tf.keras.Model(
        inputs=full_model.input, 
        outputs=full_model.layers[-3].output
    )
    
    # Определение меток
    if 'labels' in df.columns:
        y = df['labels'].values
    else:
        y_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
        y = np.argmax(df[y_cols].values, axis=1)
        
    X = []
    print("Генерация признаков...")
    
    for img_id in df['image_id']:
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        
        # Загрузка и ресайз до стандарта ResNet50
        img = image.load_img(img_path, target_size=(512, 512))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Формат батча (1, 512, 512, 3)
        
        # Специфичная для ResNet50 нормализация (перевод BGR и центрирование)
        img_array = preprocess_input(img_array)
        
        # Получение вектора признаков (verbose=0 отключает вывод логов на каждом шаге)
        features = feature_extractor.predict(img_array, verbose=0)
        X.append(features.flatten())
        
    np.save(output_x, np.array(X))
    np.save(output_y, y)
    print("Извлечение завершено.")

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    custom_model_path = params["data"]["resnet_weights_path"]
    input_dir = "data/processed"
    output_dir = "data/features"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(input_dir, "train_processed.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test_processed.csv"))
    
    extract_keras_features(train_df, os.path.join(input_dir, "images_train"), custom_model_path,
                           os.path.join(output_dir, "X_train_resnet.npy"), 
                           os.path.join(output_dir, "y_train_resnet.npy"))
                                   
    extract_keras_features(test_df, os.path.join(input_dir, "images_test"), custom_model_path,
                           os.path.join(output_dir, "X_test_resnet.npy"), 
                           os.path.join(output_dir, "y_test_resnet.npy"))