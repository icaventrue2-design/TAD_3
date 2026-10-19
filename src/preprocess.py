import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def change_skale(df, input_dir="data/raw", output_dir="data/processed", img_size=(128, 128), train=True):
    print("Ресайзим изображения...")
    for idx, row in df.iterrows():
        image_id = row['image_id']
        # Путь к оригинальной картинке (в Plant Pathology картинки обычно лежат в images/)
        img_path = os.path.join(input_dir, "images", f"{image_id}.jpg")
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize(img_size)
            
            # Сохраняем в обработанную папку
            if train:
                out_img_path = os.path.join(output_dir, "images_train", f"{image_id}.jpg")
            else:
                out_img_path = os.path.join(output_dir, "images_test", f"{image_id}.jpg")
            img.save(out_img_path)
            
    # Сохраняем копию CSV в обработанные данные
    if train:
        df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    else:
        df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)  
    print("Предобработка завершена успешно!")


def preprocess_data(input_dir="data/raw", output_dir="data/processed", img_size=(2048, 1365)):
    os.makedirs(os.path.join(output_dir, "images_train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images_test"), exist_ok=True)

    # Читаем CSV с разметкой
    train_csv_path = os.path.join(input_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Файл {train_csv_path} не найден! Проверьте структуру data/raw/")        
    df = pd.read_csv(train_csv_path)

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    change_skale(train_df)
    change_skale(test_df, train=False)

    

if __name__ == "__main__":
    preprocess_data()