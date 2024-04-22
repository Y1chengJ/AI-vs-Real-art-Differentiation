import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []
        self.label_to_idx = {}
        self.label_counts = {}  # Dictionary to store counts of images per label
        tag = 0

        for idx, label in enumerate(sorted(os.listdir(self.root_dir))):
            folder_path = os.path.join(self.root_dir, label)
            if os.path.isdir(folder_path):
                # if "SD" in label:
                #     label = "Stable-Diffusion"
                # elif "LD" in label:
                #     label = "Latent-Diffusion"
                # elif "Dalle" in label:
                #     label = "Dalle"
                # elif "journey" in label:
                #     label = "Midjourney"
                # else:
                #     label = "Real-Art"

                label_idx = self.label_to_idx.get(label, None)
                if label_idx is None:
                    self.label_to_idx[label] = tag
                    label_idx = tag
                    tag += 1

                # Initialize count for this label
                self.label_counts[label] = self.label_counts.get(label, 0)

                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)
                        self.labels.append(label_idx)
                        # Increment count for this label
                        self.label_counts[label] += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_label_counts(self):
        """ Returns the count of images for each label. """
        return self.label_counts

# Define transformations and dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset_path = 'train-dataset'
dataset = CustomImageDataset(root_dir=dataset_path, transform=transform)
label_counts = dataset.get_label_counts()
print(dataset.label_to_idx)

# for label, count in label_counts.items():
#     print(f"Label '{label}' has {count} images.")

# Splitting the dataset into training and validation
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("rexnet_150", pretrained=True, num_classes=len(dataset.label_to_idx))
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=10, device=device):
    metrics_path = 'new_rexnet_training_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write('Epoch,Train Loss,Train Accuracy,Train F1,Val Loss,Val Accuracy,Val F1\n')
        
        for epoch in tqdm(range(epochs)):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            train_preds, train_targets = [], []

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/ Training', leave=False)
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                train_preds.extend(preds.view(-1).cpu().numpy())
                train_targets.extend(labels.view(-1).cpu().numpy())

            train_accuracy = train_correct / train_total
            train_f1 = f1_score(train_targets, train_preds, average='weighted')
            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            val_preds, val_targets = [], []
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/ Validation', leave=False)
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    val_preds.extend(preds.view(-1).cpu().numpy())
                    val_targets.extend(labels.view(-1).cpu().numpy())

            val_accuracy = val_correct / val_total
            val_f1 = f1_score(val_targets, val_preds, average='weighted')
            val_loss /= len(val_loader)

            # Writing metrics to file
            f.write(f'{epoch+1},{train_loss},{train_accuracy},{train_f1},{val_loss},{val_accuracy},{val_f1}\n')
            print(f'Epoch {epoch+1}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Train F1: {train_f1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, Val F1: {val_f1}')

train_model(model, train_loader, val_loader, loss_function, optimizer, epochs=10, device=device)

model_save_path = 'new_rexnet_150_trained_state_dict.pth'

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)