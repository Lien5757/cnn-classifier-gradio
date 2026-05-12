import os
import random
from PIL import Image
import os
import cv2
import numpy as np
import random

def add_noise(image, intensity=15):
    """Add random noise to the image."""
    row, col, ch = image.shape
    noise = np.random.uniform(-intensity, intensity, (row, col, ch)).astype('int16')
    noisy = cv2.add(image, noise, dtype=cv2.CV_8U)
    return noisy

def rotate_image(image):
    """Rotate the image by a random angle."""
    angle = random.choice([90, 180, 270])
    return cv2.rotate(image, {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }[angle])

def flip_image(image):
    """Flip the image horizontally or vertically."""
    flip_code = random.choice([-1, 0, 1])  # -1: both, 0: vertical, 1: horizontal
    return cv2.flip(image, flip_code)

def shear_image(image):
    """Shear the image by a random factor."""
    shear_factor = random.uniform(-0.2, 0.2)  # Shear between -20% to 20%
    rows, cols, ch = image.shape
    M = np.array([[1, shear_factor, 0],
                   [0, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

def adjust_brightness(image):
    """Adjust brightness of the image."""
    beta = random.randint(-50, 50)  # Brightness adjustment range
    return cv2.convertScaleAbs(image, beta=beta)

def adjust_contrast(image):
    """Adjust contrast of the image."""
    alpha = random.uniform(0.5, 1.5)  # Contrast adjustment range
    return cv2.convertScaleAbs(image, alpha=alpha)

def blur_image(image):
    """Blur the image using GaussianBlur."""
    ksize = random.choice([3, 5, 7])  # Choose a kernel size
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def augment_image(image):
    """Apply random augmentations: noise, rotate, flip, scale, shear, brightness, contrast, blur."""
    image = rotate_image(image)
    image = adjust_brightness(image)
    image = adjust_contrast(image)
    image = flip_image(image)
    if random.choice([True, False]):
        image = add_noise(image)
    if random.choice([True, False]):
        image = shear_image(image)
    if random.choice([True, False]):
        image = blur_image(image)
    return image

def data_augmentation(input_dir, output_dir, augment_per_image=5):
    """
    Augment images in a directory and save them to a new directory using OpenCV.

    Args:
        input_dir (str): Path to the input directory with original images.
        output_dir (str): Path to save the augmented images.
        augment_per_image (int): Number of augmented images to generate per original image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            try:
                # Read the image with OpenCV
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Skipping file {filename}, not a valid image.")
                    continue

                for i in range(augment_per_image):
                    # Apply augmentations
                    augmented_image = augment_image(image)

                    # Save the augmented image with a unique name
                    base_name, ext = os.path.splitext(filename)
                    augmented_filename = f"{base_name}_aug_{i+1}{ext}"
                    output_path = os.path.join(output_dir, augmented_filename)
                    cv2.imwrite(output_path, augmented_image)
                    # cv2.imshow('Data Augmentation:', augmented_image)
                    # cv2.waitKey(0)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

def delete_augmented_files(directory):
    """
    Delete all files in the specified directory that contain '_aug' in their filenames.

    Args:
        directory (str): The path to the directory where files will be checked.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    deleted_files = 0

    for filename in os.listdir(directory):
        if "_aug" in filename:
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                deleted_files += 1
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    print(f"Deletion completed. {deleted_files} files deleted.")

if __name__ == "__main__":
    data_augmentation(
        input_dir=r'datasets\ant_bee\train\ants',  # Path to the original images
        output_dir='datasets/augmented/cats',  # Path to save augmented images
        augment_per_image=5  # Number of augmented images per input image
    )