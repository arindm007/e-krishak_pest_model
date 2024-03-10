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
YOLO_LABEL_FILE_EXTENSION = ".txt"


def copy_files(
    src_directory: str,
    dest_directory: str,
    files: list[str],
    train: bool = True,
) -> None:
    labels_folder = os.path.join(src_directory, YOLO_LABEL_FOLDER)
    sub_directory_name = "train" if train else "test"

    out_dir = os.path.join(dest_directory, sub_directory_name)

    dest_images_folder = os.path.join(out_dir, YOLO_IMAGE_FOLDER)
    dest_labels_folder = os.path.join(out_dir, YOLO_LABEL_FOLDER)

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


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))

    train_split = params["prepare"]["train_split"]
    shuffle = params["prepare"]["shuffle"]

    data_file = params["prepare"]["data"]
    output_dir = params["data_dir"]
    merged_dir = os.path.join(output_dir, "merged")

    if not os.path.exists(data_file):
        sys.stderr.write("Data file not found\n")
        sys.exit(1)

    with zipfile.ZipFile(data_file, "r") as f:
        f.extractall(merged_dir)

    images_folder = os.path.join(merged_dir, YOLO_IMAGE_FOLDER)
    labels_folder = os.path.join(merged_dir, YOLO_LABEL_FOLDER)

    if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
        raise Exception(f"Invalid dataset {merged_dir}")

    images = []
    for _img_extension in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(images_folder, "*" + _img_extension)))

    annotations = glob.glob(
        os.path.join(labels_folder, "*" + YOLO_LABEL_FILE_EXTENSION)
    )

    if len(images) == 0 or len(annotations) == 0 or len(images) != len(annotations):
        raise Exception(f"Invalid dataset {merged_dir}")

    train_images, test_images = train_test_split(
        np.array(images),
        test_size=(1 - train_split),
        shuffle=shuffle,
        random_state=params["prepare"]["seed"],
    )

    copy_files(merged_dir, output_dir, list(train_images))
    copy_files(merged_dir, output_dir, list(test_images), train=False)

    classes = {}
    with open(os.path.join(output_dir, "train", "classes.txt"), "r") as f:
        for i, classname in enumerate(f.readlines()):
            classname = classname.strip()
            classes[i] = classname

    yaml_data = {
        "path": os.path.join(os.getcwd(), output_dir),
        "train": os.path.join("train", YOLO_IMAGE_FOLDER),
        "val": os.path.join("test", YOLO_IMAGE_FOLDER),
        "names": classes
    }

    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
