import os
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import tqdm

# Initialize and load the model
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Define input and output paths
input_images_path = 'test-dataset'
output_folder = 'cropped_test_dataset'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)


for foldername in os.listdir(input_images_path):
    output_single_folder = os.path.join(output_folder, foldername)
    os.makedirs(output_single_folder, exist_ok=True)
    folder_path = os.path.join(input_images_path, foldername)
    print(folder_path)
    
    # Process each image in the folder
    if 'Midjourney' not in foldername:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Validate file extensions
                print(filename)
                file_path = os.path.join(folder_path, filename)
                # print(file_path)
                image = cv2.imread(file_path)
                
                if image is None:
                    print(f"Failed to load image {filename}. Skipping.")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                output_image_path = os.path.join(output_single_folder, os.path.splitext(filename)[0])
                os.makedirs(output_image_path, exist_ok=True)  # Create a folder for each image

                # Save the original image in RGB format (for consistency with cropped images)
                original_image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_image_path, filename), original_image_bgr)

                # Generate masks and bounding boxes
                masks = mask_generator.generate(image_rgb)
                if masks:
                    for i, mask in enumerate(masks):
                        COCO_segmentation = mask['segmentation']
                        bool_mask = np.array(COCO_segmentation, dtype=np.uint8) * 255  # Convert boolean array to uint8 with proper scaling

                        # Create an RGBA image with the original image and the boolean mask as the alpha channel
                        image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
                        image_rgba[:, :, 3] = bool_mask  # Set alpha channel

                        # Find contours and get the bounding rectangle for each mask
                        contours, _ = cv2.findContours(bool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            x, y, w, h = cv2.boundingRect(contours[0])
                            # Crop the RGBA image using the bounding rectangle
                            cropped_image_rgba = image_rgba[y:y+h, x:x+w]
                        else:
                            print(f"No contours found for mask {i} in image {filename}. Skipping.")
                            continue

                        # Define the output image path with the PNG extension
                        cropped_filename = f'{i}.png'
                        cropped_image_path = os.path.join(output_image_path, cropped_filename)

                        # Save the cropped image in PNG format, which supports the alpha channel
                        cv2.imwrite(cropped_image_path, cropped_image_rgba)
                    # break
                else:
                    print(f"No bounding box found for {filename}")
        # break
            

print("Processing complete. All images and their cropped versions have been saved.")
