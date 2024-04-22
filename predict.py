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
    def __init__(self, root_dir, transform=None):
        self.root_dir = f"{root_dir}"
        self.transform = transform

        self.images = []
        self.labels = []
        self.label_to_idx = {'Latent-Diffusion': 0, 'Stable-Diffusion': 1, 'Dalle': 2, 'Midjourney': 3, 'Real-Art': 4}
        tag = 0
        for idx, label in enumerate(sorted(os.listdir(self.root_dir))):
            print(label)
            folder_path = os.path.join(self.root_dir, label)
            # print(idx)
            if os.path.isdir(folder_path):
                # print(folder_path)
                if "Stable" in label:
                    label = "Stable-Diffusion"
                elif "Latent" in label:
                    # print('ld')
                    label = "Latent-Diffusion"
                elif "Dalle" in label:
                    label = "Dalle"
                elif "journey" in label:
                    label = "Midjourney"
                else:
                    label = "Real-Art"
                label_idx = self.label_to_idx.get(label, None)

                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    # print(img_path)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)
                        self.labels.append(label_idx)
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
    
    
dataset_path = 'test-dataset'

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = CustomImageDataset(root_dir=dataset_path,transform=transform)
# print(dataset.images)

# Splitting the dataset into training and validation
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

num_classes = 5  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("resnet152", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('resnet_152_trained_state_dict.pth', map_location=device))
model.to(device)
model.eval()


def evaluate_model(test_loader, model, device, num_classes):
    ai_labels = [0, 1, 2, 3]
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        all_preds = [1 if pred in ai_labels else 0 for pred in all_preds]
        all_labels = [1 if label in ai_labels else 0 for label in all_labels]
        # print(all_preds)
        # print(all_labels)            

    overall_accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return overall_accuracy, f1

# Evaluate the model
overall_accuracy, f1 = evaluate_model(test_loader, model, device, num_classes)

# Print the metrics

print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
