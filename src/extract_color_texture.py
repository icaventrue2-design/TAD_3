import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops

def extract_features(df, images_dir):
    """
    Функция принимает путь к изображению и возвращает одномерный вектор признаков, 
    готовый для подачи в Random Forest.
    """

    X = []

    y_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
    # Превращаем 4 колонки в один индекс (0, 1, 2, 3)
    y = np.argmax(df[y_cols].values, axis=1)
    
    for img_id in df["image_id"]:

        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        # 1. Загрузка изображения
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Изображение не найдено!")
            
        # --- ИЗВЛЕЧЕНИЕ ЦВЕТОВЫХ ПРИЗНАКОВ ---
        # Переводим из BGR (формат OpenCV по умолчанию) в HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Вычисляем гистограмму (3 канала: Hue, Saturation, Value)
        # Используем 8 бинов для каждого канала для уменьшения размерности (8x8x8 = 512 признаков)
        hist = cv2.calcHist([hsv_img], channels=[0, 1, 2], mask=None, 
                            histSize=[8, 8, 8], ranges=[0, 180, 0, 256, 0, 256])
        
        # Нормализуем гистограмму, чтобы размер изображения не влиял на признаки
        cv2.normalize(hist, hist)
        color_features = hist.flatten() # Превращаем в одномерный вектор (512 элементов)
        
        # --- ИЗВЛЕЧЕНИЕ ТЕКСТУРНЫХ ПРИЗНАКОВ (GLCM) ---
        # Переводим в оттенки серого, так как GLCM работает только с одним каналом
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Строим матрицу GLCM. 
        # distances=[1] - смотрим на соседние пиксели (расстояние 1)
        # angles=[0, np.pi/4, np.pi/2, 3*np.pi/4] - под углами 0, 45, 90 и 135 градусов
        glcm = graycomatrix(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
        
        # Извлекаем метрики Харалика из матрицы GLCM (усредняем по всем углам)
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        texture_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
        
        # --- ОБЪЕДИНЕНИЕ ПРИЗНАКОВ ---
        # Сцепляем 512 цветовых и 5 текстурных признаков в один вектор (517 признаков)
        global_features = np.hstack([color_features, texture_features])
        X.append(global_features)
        
    return np.array(X), y