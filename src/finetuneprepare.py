import glob
import os
import shutil
import sys
import yaml
import zipfile

import numpy as np

from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = [".jpg", ".png"]
YOLO_IMAGE_FOLDER = "images"
YOLO_LABEL_FOLDER = "labels"
YOLO_CLASSES_FILE = "classes.txt"
YOLO_NOTES_FILE = "notes.json"
YOLO_LABEL_FILE_EXTENSION = ".txt"



def copy_files_finetune(
    src_directory: str,
    dest_directory: str,
    files: list[str],
    task:str,
    train: bool = True,
) -> None:
    labels_folder = os.path.join(src_directory, YOLO_LABEL_FOLDER)
    sub_directory_name = "merged"

    out_dir = os.path.join(dest_directory, sub_directory_name)

    dest_images_folder = os.path.join(out_dir, YOLO_IMAGE_FOLDER)
    dest_labels_folder = os.path.join(out_dir, YOLO_LABEL_FOLDER)
    if(task== 'train' or 'test'):
        if os.path.exists(out_dir):
            all_content = glob.glob(os.path.join(out_dir, "*"))
            for content in all_content:
                if os.path.isfile(content):
                    os.remove(content)
                else:
                    shutil.rmtree(content)
        else:
            os.makedirs(out_dir, exist_ok=True)

        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_labels_folder, exist_ok=True)
    else:
        print('Fine-tuning prep on progress')

    for image in files:
        filename, img_extension = os.path.splitext(os.path.basename(image))
        # get label file
        label_file_path = os.path.join(
            labels_folder, filename + YOLO_LABEL_FILE_EXTENSION
        )

        # move image and label files
        shutil.copy2(
            image,
            os.path.join(dest_images_folder, filename + img_extension),
        )
        shutil.copy2(
            label_file_path,
            os.path.join(dest_labels_folder, filename + YOLO_LABEL_FILE_EXTENSION),
        )

    shutil.copy2(
        os.path.join(src_directory, YOLO_CLASSES_FILE),
        os.path.join(out_dir, YOLO_CLASSES_FILE),
    )
    shutil.copy2(
        os.path.join(src_directory, YOLO_NOTES_FILE),
        os.path.join(out_dir, YOLO_NOTES_FILE),
    )

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))

    train_split = params["prepare"]["train_split"]
    shuffle = params["prepare"]["shuffle"]

    data_file = params["prepare"]["data"]
    output_dir = params["data_dir"]
    merged_dir = os.path.join(output_dir, "merged")

    feed_split = params["fine-tune-prepare"]["feed_split"]
    shuffle = params["fine-tune-prepare"]["shuffle"]
    old_data_file = params["fine-tune-prepare"]["data_old"]
    new_data_file = params["fine-tune-prepare"]["data_new"]
    temp_dir = os.path.join(output_dir,"temp")

    if not os.path.exists(old_data_file):
        sys.stderr.write("Data file not found\n")
        sys.exit(1)
    
    with zipfile.ZipFile(old_data_file, "r") as f:
        f.extractall(temp_dir)

    images_feed_folder = os.path.join(temp_dir, YOLO_IMAGE_FOLDER)
    labels_feed_folder = os.path.join(temp_dir, YOLO_LABEL_FOLDER)

    if not os.path.exists(images_feed_folder) or not os.path.exists(labels_feed_folder):
        raise Exception(f"Invalid dataset {merged_dir}")

    temp_images = []
    for _img_extension in IMAGE_EXTENSIONS:
        temp_images.extend(glob.glob(os.path.join(images_feed_folder, "*" + _img_extension)))

    temp_annotations = glob.glob(
        os.path.join(labels_feed_folder, "*" + YOLO_LABEL_FILE_EXTENSION)
    )


    if len(temp_images) == 0 or len(temp_annotations) == 0 or len(temp_images) != len(temp_annotations):
        raise Exception(f"Invalid dataset {merged_dir}")

    feed_images, unfeed_images = train_test_split(
        np.array(temp_images),
        test_size=(1 - feed_split),
        shuffle=shuffle,
        random_state=params["fine-tune-prepare"]["seed"],
)

    copy_files_finetune(temp_dir, os.path.join(output_dir), list(feed_images),task='train')
    # copy_files_finetune(merged_dir, output_dir, list(test_images), train=False)
    with zipfile.ZipFile(new_data_file, "r") as f:
        f.extractall(merged_dir)

    data_zip_filename = fr'data\data'
    dir_name = merged_dir
    shutil.make_archive(data_zip_filename, 'zip', dir_name)

    # remove the temp and 
    rm_dir =["merged","temp"]
    for dir in rm_dir:
        if os.path.exists(os.path.join(output_dir,dir)):
            all_content = glob.glob(os.path.join(os.path.join(output_dir,dir), "*"))
        for content in all_content:
            if os.path.isfile(content):
                os.remove(content)
            else:
                shutil.rmtree(content)






