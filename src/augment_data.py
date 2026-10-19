import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

def augment_dataset(input_dir="data/processed", output_dir="data/augmented"):
    images_in_dir = os.path.join(input_dir, "images_train")
    csv_path = os.path.join(input_dir, "train_processed.csv")
    
    out_images_dir = os.path.join(output_dir, "images")
    os.makedirs(out_images_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден! Сначала запустите предобработку.")
        
    df = pd.read_csv(csv_path)
    df['filename'] = df['image_id'] + '.jpg'
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1
    )
    
    augmented_rows = []
    
    print("Генерация аугментированных данных...")
    for idx, row in df.iterrows():
        img_path = os.path.join(images_in_dir, row['filename'])
        if not os.path.exists(img_path):
            continue
            
        # Копируем оригинал
        orig_img = load_img(img_path, target_size=(128, 128))
        orig_img.save(os.path.join(out_images_dir, row['filename']))
        augmented_rows.append(row)
        
        # Создаем аугментированную версию
        x = img_to_array(orig_img)
        x = x.reshape((1,) + x.shape)
        
        for batch in datagen.flow(x, batch_size=1):
            aug_img = array_to_img(batch[0])
            aug_filename = f"{row['image_id']}_aug.jpg"
            aug_img.save(os.path.join(out_images_dir, aug_filename))
            
            new_row = row.copy()
            new_row['image_id'] = f"{row['image_id']}_aug"
            new_row['filename'] = aug_filename
            augmented_rows.append(new_row)
            break
            
    aug_df = pd.DataFrame(augmented_rows)
    aug_df.drop(columns=['filename'], inplace=True, errors='ignore')
    aug_df.to_csv(os.path.join(output_dir, "train_augmented.csv"), index=False)
    print(f"Аугментация завершена! Создан датасет: {len(aug_df)} записей в {output_dir}")

if __name__ == "__main__":
    augment_dataset()