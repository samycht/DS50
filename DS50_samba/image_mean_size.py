import os
from PIL import Image

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size

def get_average_image_dimensions(directory):
    total_width = 0
    total_height = 0
    num_images = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(root, file)
                width, height = get_image_dimensions(image_path)
                total_width += width
                total_height += height
                num_images += 1

    if num_images == 0:
        return 0, 0
    else:
        avg_width = total_width // num_images
        avg_height = total_height // num_images
        return avg_width, avg_height

directory_path = "C:/Users/galse/DS50/new_image_folder"
avg_width, avg_height = get_average_image_dimensions(directory_path)
print("Dimensions moyennes des images :", avg_width, "x", avg_height, "pixels")
