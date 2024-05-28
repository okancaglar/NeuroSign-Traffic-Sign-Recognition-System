import os
import shutil
from sklearn.model_selection import train_test_split

#paths

raw_data_path = ""
training_image = ""
val_image = ""
training_label = ""
val_label = ""

def collect_image_pairs(raw_dataset_dir) -> list:

    image_annotation_pairs=[]
    for file in os.listdir(raw_data_path):
        if file.endswith(".jpg"):
            annotation_file = raw_dataset_dir + "/" + file.replace(".jpg", ".txt")
            if os.path.isfile(annotation_file):
                image_annotation_pairs.append([os.path.join(raw_data_path, file), annotation_file])
            else:
                print(f"annotation is not found for image : {file}")

    return image_annotation_pairs

image_annotation_pairs = collect_image_pairs(raw_data_path)

train_pairs, val_pairs = train_test_split(image_annotation_pairs, test_size=0.2, random_state=42)

def copy_files(pairs, output_images_dir, output_labels_dir):

    for image_path, label_path in pairs:
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)

        img_dst_path = os.path.join(output_images_dir, image_name)
        label_dst_path = os.path.join(output_labels_dir, label_name)

        shutil.copy2(image_path, img_dst_path)
        shutil.copy2(label_path, label_dst_path)

copy_files(train_pairs, training_image,  training_label)
copy_files(val_pairs, val_image, val_label)

