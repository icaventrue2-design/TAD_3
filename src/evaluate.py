import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_on_test_set(model_path, test_df, data_dir, target_columns):
    if not os.path.exists(model_path):
        return None
    
    test_df = test_df.copy()
    test_df['filename'] = test_df['image_id'] + '.jpg'
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=data_dir,
        x_col='filename',
        y_col=target_columns,
        target_size=(128, 128),
        batch_size=32,
        class_mode='raw',
        shuffle=False
    )
    
    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_gen, verbose=0)
    
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Извлекаем честные метки классов
    y_true_labels = test_df[target_columns].values
    y_true = np.argmax(y_true_labels, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=target_columns, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        "loss": round(float(loss), 4),
        "accuracy": round(float(accuracy) * 100, 2),
        "precision_macro": round(float(report['macro avg']['precision']), 4),
        "recall_macro": round(float(report['macro avg']['recall']), 4),
        "f1_macro": round(float(report['macro avg']['f1-score']), 4),
        "f1_weighted": round(float(report['weighted avg']['f1-score']), 4)
    }
    
    return metrics, cm

def main():
    target_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
    metrics_summary = {}
    cm_plots = {}
    
    test_data = "data/processed/images_test"
    test_csv = "data/processed/test_processed.csv"
    if not os.path.exists(test_csv):
        print("Файл test.csv не найден. Запустите сначала preprocess.py!")
        return
        
    test_df = pd.read_csv(test_csv)

    # Поиск обученных моделей
    path_to_models = "models/"
    models = {}
    for filename in os.listdir(path_to_models):
        file_path = os.path.join(path_to_models, filename)
        models[filename] = file_path
    
    metrics_summary["test"] = {}
    for name, path in models.items():
        res = evaluate_on_test_set(path, test_df, "data/processed/images_test", target_columns)
        if res:
            metrics_summary["test"][name], cm_plots[f"test_{name}"] = res

    # Сохранение итоговых отчетов
    with open("./metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
        
    os.makedirs("plots", exist_ok=True)
    with open("plots/confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm_plots, f, indent=4, ensure_ascii=False)

    print("Оценка на изолированной ТЕСТОВОЙ выборке завершена!")
    print("Результаты сохранены в metrics.json и plots/confusion_matrix.json")

if __name__ == "__main__":
    main()