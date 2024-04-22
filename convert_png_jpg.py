import os
from PIL import Image

input_path = 'cropped_test_dataset'
output_path = 'cropped_test_dataset_jpg'

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

for label in os.listdir(input_path):
    label_path = os.path.join(input_path, label)
    output_label_path = os.path.join(output_path, label)
    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)
    print(label)
    for image_folder in os.listdir(label_path):
        image_path = os.path.join(label_path, image_folder)
        output_image_folder_path = os.path.join(output_label_path, image_folder)
        if not os.path.exists(output_image_folder_path):
            os.makedirs(output_image_folder_path)
        for img_file in os.listdir(image_path):
            img_path = os.path.join(image_path, img_file)
            if os.path.isfile(img_path):
                try:
                    with Image.open(img_path) as img:
                        # Remove file extension and add .jpg
                        base_filename = os.path.splitext(img_file)[0]
                        output_file_path = os.path.join(output_image_folder_path, f"{base_filename}.jpg")
                        img.convert('RGB').save(output_file_path, "JPEG")
                        print(output_file_path)
                except Exception as e:
                    print(f"Error converting {img_path}: {e}")
