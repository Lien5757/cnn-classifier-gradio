import os
import cv2
import random
import shutil
from glob import glob
from tqdm import tqdm
from datetime import datetime

from utils.data_augmant_utils import augment_image

import gradio as gr


def split_dataset(source_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    classes = sorted(os.listdir(source_dir))
    split_result = {"train": {}, "val": {}, "test": {}}

    for c in classes:
        class_dir = os.path.join(source_dir, c)
        images = glob(os.path.join(class_dir, "*"))
        random.shuffle(images)

        total = len(images)
        val_count = int(total * val_ratio)
        test_count = int(total * test_ratio)

        split_result["val"][c] = images[:val_count]
        split_result["test"][c] = images[val_count:val_count + test_count]
        split_result["train"][c] = images[val_count + test_count:]

    return split_result


def save_split_dataset(split_result, output_dir):
    splits = ["train", "val", "test"]
    if "test" not in split_result or all(
        len(files) == 0 for files in split_result["test"].values()):
        splits = ["train", "val"]

    print(f'Saving split dataset to: {output_dir}')
    print(f'Split Summary:')
    for cls in split_result["train"].keys():
        msg = f"  {cls:<12}"

        for split in splits:
            num_samples = len(split_result.get(split, {}).get(cls, []))
            msg += f" | {split}: {num_samples}"

        print(msg)

    # Save files
    for split in splits:
        for cls, file_list in split_result[split].items():
            save_dir = os.path.join(output_dir, split, cls)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for f in file_list:
                filename = os.path.basename(f)
                shutil.copy(f, os.path.join(save_dir, filename))


def offline_augment_train_data(output_dir, target_min=200):
    train_dir = os.path.join(output_dir, "train")
    classes = sorted(os.listdir(train_dir))
    print('Offline augment train data per class')

    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        images = sorted(glob(os.path.join(cls_dir, "*")))
        num_images = len(images)

        if num_images >= target_min:
            print(f"[{cls}] original count = {num_images} → OK, there's no need for offline augmentation.")
            continue

        need = target_min - num_images
        print(f"[{cls}] original count = {num_images} → augment {need} images")

        for i in tqdm(range(need)):
            img_path = random.choice(images)
            img = cv2.imread(img_path)

            aug_img = augment_image(img)

            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            save_name = f"{base_name}_aug_{i:04d}{ext}"
            save_path = os.path.join(cls_dir, save_name)

            cv2.imwrite(save_path, aug_img)

def copy_dataset_structure(source_dir, output_dir):
    """
    Copy existing train/val/test dataset from source_dir to output_dir.
    Keeps folder structure and files unchanged.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    os.makedirs(output_dir, exist_ok=True)

    items = ["train", "val", "test"]

    for split in items:
        src_path = os.path.join(source_dir, split)
        dst_path = os.path.join(output_dir, split)

        if not os.path.exists(src_path):
            print(f"[WARNING] Missing split folder: {src_path}, skipping.")
            continue

        # shutil.copytree() requires dst not to exist, so we handle manually
        os.makedirs(dst_path, exist_ok=True)

        for root, dirs, files in os.walk(src_path):
            relative = os.path.relpath(root, src_path)
            new_root = os.path.join(dst_path, relative)
            os.makedirs(new_root, exist_ok=True)

            for f in files:
                shutil.copy(os.path.join(root, f), os.path.join(new_root, f))

    print(f"Copy done → {output_dir}")


def main(source_dir, save_name, val_ratio, test_ratio, isAugment, target_num):
    """
    Preprocess an raw classified dataset (class0/class1/...).
    1. Split original dataset.
    2. Save split dataset.
    3. Check whether to Offline augment train set.
    """
    base_output_dir = os.path.join('datasets', save_name)
    os.makedirs(base_output_dir, exist_ok=True)

    ## Step 1: Data splitting
    splits = split_dataset(source_dir, val_ratio, test_ratio)

    ## Step 2: Save the result under ./datasets
    save_split_dataset(splits, base_output_dir)

    ## Step 3: Check whether to Offline augment train set.
    if isAugment:
        time_frame = datetime.now().strftime("%Y%m%d")
        aug_output_dir = os.path.join('datasets', f'{save_name}_processed_{time_frame}')
        
        print(f"Creating processed dataset at: {aug_output_dir}")

        ## Step 4: Copy train/val/test structure
        copy_dataset_structure(base_output_dir, aug_output_dir)

        ## Step 5: Offline augment train only
        offline_augment_train_data(aug_output_dir, target_num)

if __name__ == "__main__":
    # main(
    #     source_dir=r'D:\Lien-master\113-1\類神經網路\dataset\ant_bee\train',
    #     save_name='ant_bee_3',
    #     val_ratio=0.15,
    #     test_ratio=0.15,
    #     isAugment=False,
    #     target_num=100
    # )

    with gr.Blocks() as demo:            
        with gr.Tab("Data preparation"):
            gr.Markdown("""
            ### For classification tasks with stratified split:
            - If total samples < 20k, use 70/15/15 to ensure stable validation and test estimates.
            - If total samples ≥ 20k, use 80/10/10 to maximize training data while keeping evaluation reliable.
            - If you only need to split in train/val, set test ratio to be 0.
            
            ### If Train per class < 100, please be aware of the result!
                        
            ### Splitted data folder would be saved under ./datasets root with the data name you set.
            """)
            with gr.Column():
                data_root = gr.Textbox(label='Data root')
                save_name = gr.Textbox(label='Save data name', value='Splitted_data')
                with gr.Row():
                    val_ratio = gr.Number(label='Val ratio', value=0.15)
                    test_ratio = gr.Number(label='Test ratio', value=0.15)
                with gr.Row():
                    isAugment = gr.Checkbox(label='Offline augmentation')
                    target_num = gr.Number(label='Minimun num of training data per class', value=300)
                offline_aug_btn = gr.Button('Split & Offline Augmentation')
                offline_aug_btn.click(
                    fn=main,
                    inputs=[data_root, save_name, val_ratio, test_ratio, isAugment, target_num],
                ) 
    demo.launch()

