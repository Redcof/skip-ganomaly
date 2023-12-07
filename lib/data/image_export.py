import os
import pathlib
import random
import shutil

import cv2
from hif_reader.hif_file_reader import HifFileReader
from hif_reader.tools.matlum_rgb import matlum_float_to_rgb
from ray.experimental.tqdm_ray import tqdm

sd3_hif_path = r"/data/bags_sd3"
sixray_path = r"/data/Sixray_easy"
sixray_sd3_dataset = r"/data/sixray_sd3_anomaly"
sd3_img_export_path = os.path.join(sd3_hif_path, "exported")


def crawl(root, file_type=".hif"):
    """
    This is a file path generator for a given root directory.
    It crawls through all subdirectories recursively
    @param file_type:  The file type to search for
    @param root: The root directory
    @return:
    """
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(file_type):
                yield os.path.join(path, name)


def export(hif_path, save_path):
    """
    Read one HIF file and save each view as JPEG image in a given export directory as
    the following formal: HIF_NAME.hif-<view_index>.jpg,
    For example, HIF file: BAGGAGE_20231203_000000_12345.hif has 4 views, then the following files will be created:
        BAGGAGE_20231203_000000_12345.hif-0.jpg
        BAGGAGE_20231203_000000_12345.hif-1.jpg
        BAGGAGE_20231203_000000_12345.hif-2.jpg
        BAGGAGE_20231203_000000_12345.hif-3.jpg
    @param hif_path: The HIF path
    @param save_path: The path to export jpg images
    @return: None
    """
    src = os.path.join(save_path)
    path = os.path.join(src, "%s-%d.jpg" % (os.path.basename(hif_path), 0))
    if os.path.exists(path):
        return 0
    path = os.path.join(src, "train", "%s-%d.jpg" % (os.path.basename(hif_path), 0))
    if os.path.exists(path):
        return 0
    path = os.path.join(src, "test", "%s-%d.jpg" % (os.path.basename(hif_path), 0))
    if os.path.exists(path):
        return 0
    
    reader = HifFileReader()
    reader.read(hif_path, section_type_filter=["run_len_matlum"], )
    views_matlum = reader.get_sections_of_type("run_len_matlum")
    for view_idx, matlum in enumerate(views_matlum):
        channel_data = matlum["data"]["image_data_matlum_float"]
        rgb = matlum_float_to_rgb(channel_data)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        path = os.path.join(save_path, "%s-%d.jpg" % (os.path.basename(hif_path), view_idx))
        cv2.imwrite(path, rgb)
    return len(views_matlum)


def export_form_dir(hif_path, img_export_path):
    """
    Read all HIF files available in a directory and save each view as JPEG image in a given export directory as
    the following formal: HIF_NAME.hif-<view_index>.jpg,
    For example, HIF file: BAGGAGE_20231203_000000_12345.hif has 4 views, then the following files will be created:
        BAGGAGE_20231203_000000_12345.hif-0.jpg
        BAGGAGE_20231203_000000_12345.hif-1.jpg
        BAGGAGE_20231203_000000_12345.hif-2.jpg
        BAGGAGE_20231203_000000_12345.hif-3.jpg
    @param hif_path: The HIF path root directory
    @param img_export_path: The path to export jpg images
    @return: None
    """
    try:
        os.mkdir(img_export_path)
    except:
        ...
    print("Calculating files...", end="")
    files = list(crawl(hif_path))
    print("Done")
    print("Exporting...")
    counter = 0
    for path in tqdm(files):
        counter += export(path, img_export_path)
    print("\n%d images exported form %d HIF files" % (counter, len(files)))


def copy_files(src, files, dest, delete_at_source=False, overwrite=False):
    print("Moving" if delete_at_source else "Copying", end=" ")
    print("file form '%s' -> '%s'" % (src, dest))
    try:
        os.mkdir(dest)
    except:
        ...
    for file in tqdm(files):
        # `dest` needs to empty otherwise override needs to set to be True
        if os.path.exists(os.path.join(dest, file)) and not overwrite:
            continue
        # `src` needs it to be a file
        if os.path.isfile(os.path.join(src, file)):
            shutil.copyfile(os.path.join(src, file), os.path.join(dest, file))  # copy
            if delete_at_source:
                os.remove(os.path.join(src, file))  # remove copied
    print("")


def move_files(src, files, dest):
    copy_files(src, files, dest, True)


def split(src, portion=.8):
    """
    This will create `train` and `test` directory and move images to two directories while following
    the portion given.
    For example, if we have 1000 images and portion=0.8, then the script will move 20%, 200 images
    to the newly created `test` subdirectory and 800 images to the `train` directory.
    @param src: the path where all exported jpg images is stored.
    @param portion: The percent to split. A float value between 0 and 1
    @return: None
    """
    files = os.listdir(src)
    random.seed(47)
    random.shuffle(files)  # shuffle once
    random.seed(50)
    random.shuffle(files)  # shuffle twice
    first_half = files[:int(len(files) * portion)]
    second_half = files[int(len(files) * portion):]
    dest_train = os.path.join(src, "train")
    dest_test = os.path.join(src, "test")
    print("Splitting exported images")
    move_files(src, first_half, dest_train)
    move_files(src, second_half, dest_test)


def copy_sixray_easy(sixray_path, anomaly_dataset_path):
    sixray = pathlib.Path(sixray_path)
    src_train = sixray / "train" / "JPEGImages"
    src_test = sixray / "test" / "JPEGImages"
    anomaly = pathlib.Path(anomaly_dataset_path)
    train_files = list(crawl(src_train, ".jpg"))
    test_files = list(crawl(src_test, ".jpg"))
    dest_test = anomaly / "test" / "abnormal"
    print("Copy Sixray data to anomaly dataset")
    copy_files(src_train, list(map(os.path.basename, train_files)), dest_test, overwrite=True)
    copy_files(src_test, list(map(os.path.basename, train_files)), dest_test)


def copy_sd3_dataset(sd3_path, anomaly_dataset_path):
    """copy hif exported images to anomaly dataset."""
    sd3 = pathlib.Path(sd3_path)
    anomaly = pathlib.Path(anomaly_dataset_path)
    dest_train = anomaly / "train" / "normal"
    dest_test = anomaly / "test" / "normal"
    src_train = sd3 / "train"
    src_test = sd3 / "test"
    print("Copy SD3 data to anomaly dataset")
    copy_files(src_train, os.listdir(src_train), dest_train)
    copy_files(src_test, os.listdir(src_test), dest_test)


if __name__ == '__main__':
    export_form_dir(sd3_hif_path, sd3_img_export_path)
    split(sd3_img_export_path, .8)
    copy_sd3_dataset(sd3_img_export_path, sixray_sd3_dataset)
    copy_sixray_easy(sixray_path, sixray_sd3_dataset)