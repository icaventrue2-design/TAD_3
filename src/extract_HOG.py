import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def extract_Hog(df, images_dir):
    # Списки image_paths (пути к фото) и labels (классы заболеваний)
    X_hog = []

    y_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
    # Превращаем 4 колонки в один индекс (0, 1, 2, 3)
    y = np.argmax(df[y_cols].values, axis=1)

    for img_id in df["image_id"]:
        img_path = os.path.join(images_dir, f"{img_id}.jpg")

        # 1. Загрузка в оттенках серого (HOG отлично работает без цвета)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Фиксация размера (все векторы должны быть одинаковой длины)
        img_resized = cv2.resize(img, (128, 128))
        
        # 3. Извлечение признаков HOG
        # orientations: количество корзин углов
        # pixels_per_cell: размер одной ячейки (влияет на детализацию)
        # cells_per_block: размер блока для локальной нормализации (борьба с освещением)
        features = hog(img_resized, 
                    orientations=9, 
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), 
                    block_norm='L2-Hys',
                    visualize=False) # visualize=False отдает только математический вектор
        
        X_hog.append(features)

    X_hog = np.array(X_hog)

    return X_hog, y