import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import timm


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label=1):
        self.root_dir = f"{root_dir}"
        self.transform = transform
        self.images = []
        self.labels = []
        self.label = label
        for img in os.listdir(self.root_dir):
            img_path = os.path.join(self.root_dir, img)
            if os.path.isfile(img_path):
                self.images.append(img_path)
                self.labels.append(label)

        # print(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
prediction_dataset_path = 'cropped_test_dataset'
transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

num_classes = 5  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("resnet152", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('resnet_152_trained_state_dict.pth', map_location=device))
model.to(device)
model.eval()


ai_labels = ["Stable-Diffusion", "Latent-Diffusion", "Dalle", "Midjourney"]
ai_idx = [0, 1, 2, 3]
accuracy_list = []

# iterate through the segmented images
for label in os.listdir(prediction_dataset_path):
    label_path = os.path.join(prediction_dataset_path, label)
    print(label)
    for image_folder in os.listdir(label_path):
        image_path = os.path.join(label_path, image_folder)
        # print(image_path)
        idx = 1 if label in ai_labels else 0
        prediction_dataset = CustomImageDataset(root_dir=image_path, transform=transform, label=idx)
        prediction_loader = DataLoader(prediction_dataset, batch_size=64, shuffle=False)
        all_preds = []
        for images, labels in prediction_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
        all_preds = [1 for pred in all_preds if pred in ai_idx]
        # print(all_preds)
        accuracy = sum([1 for pred in all_preds if pred == idx]) / len(all_preds)  if len(all_preds) > 0 else 0
        threshold = 0.5
        accuracy_list.append(1 if accuracy > threshold else 0) # if the proportion is greater than the threshold, we assume it is predicted correctly

accuracy = sum(accuracy_list) / len(accuracy_list)
print(accuracy)


# print(f"Accuracy: {accuracy}")
# print(predicted_labels)
