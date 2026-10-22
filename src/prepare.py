import os
import yaml
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def change_skale(df, input_dir="data/raw", output_dir="data/processed", img_size=(128, 128), train=True):
    print(f"Ресайзим изображения для {'train' if train else 'test'}...")
    for idx, row in df.iterrows():
        image_id = row['image_id']
        img_path = os.path.join(input_dir, "images", f"{image_id}.jpg")
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize(img_size)
            
            if train:
                out_img_path = os.path.join(output_dir, "images_train", f"{image_id}.jpg")
            else:
                out_img_path = os.path.join(output_dir, "images_test", f"{image_id}.jpg")
            img.save(out_img_path)
            
    if train:
        df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    else:
        df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)  


def augment_train_data(train_df, output_dir="data/processed"):
    print("Аугментация обучающей выборки...")
    augmented_rows = []
    
    for idx, row in train_df.iterrows():
        image_id = row['image_id']
        # Берем уже сохраненную базовую картинку из папки train
        img_path = os.path.join(output_dir, "images_train", f"{image_id}.jpg")
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            
            # 1. Горизонтальное отражение
            flip_img = ImageOps.mirror(img)
            flip_id = f"{image_id}_flip"
            flip_img.save(os.path.join(output_dir, "images_train", f"{flip_id}.jpg"))
            
            new_row1 = row.copy()
            new_row1['image_id'] = flip_id
            augmented_rows.append(new_row1)
            
            # 2. Поворот на 90 градусов
            rot_img = img.rotate(90)
            rot_id = f"{image_id}_rot90"
            rot_img.save(os.path.join(output_dir, "images_train", f"{rot_id}.jpg"))
            
            new_row2 = row.copy()
            new_row2['image_id'] = rot_id
            augmented_rows.append(new_row2)
            
    # Объединяем оригинальный датафрейм с новыми строками
    aug_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([train_df, aug_df], ignore_index=True)
    
    # Перезаписываем CSV с учетом новых изображений
    final_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    print(f"Аугментация завершена. Создано новых изображений: {len(augmented_rows)}.")


def load_scale_split_data(input_dir="data/raw", output_dir="data/processed", img_size=(128, 128), seed=42, augment_flag=False):
    os.makedirs(os.path.join(output_dir, "images_train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images_test"), exist_ok=True)

    train_csv_path = os.path.join(input_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Файл {train_csv_path} не найден! Проверьте структуру data/raw/")  

    df = pd.read_csv(train_csv_path)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=seed)
    
    # 1. Базовый ресайз
    change_skale(train_df, input_dir, output_dir, img_size, train=True)
    change_skale(test_df, input_dir, output_dir, img_size, train=False)

    # 2. Аугментация только для train, если включен флаг
    if augment_flag:
        augment_train_data(train_df, output_dir)
        
    print("Предобработка полностью завершена!")


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    augment_flag = params["data"]["augment"]
    
    # Обработка размера картинки. Если в yaml указано одно число (например, 128), делаем кортеж (128, 128)
    size_val = params["data"]["image_size"] 
    img_size = (size_val, size_val) if isinstance(size_val, int) else tuple(size_val)
    
    seed = params.get("data", {}).get("seed", 42)
    
    load_scale_split_data(img_size=img_size, seed=seed, augment_flag=augment_flag)