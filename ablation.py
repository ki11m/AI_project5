import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文本读取函数
def read_text(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {text_path}: {e}")
        return ""

# 数据集类
class MultimodalDataset(Dataset):
    def __init__(self, text_file, image_dir, tokenizer, transform=None):
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.data = self.load_data()
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}

    def load_data(self):
        data = []
        with open(self.text_file, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 2:
                guid, label = parts
                data.append((guid, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid, label = self.data[idx]

        text_path = os.path.join(self.image_dir, f'{guid}.txt')
        text = read_text(text_path).strip()
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        text_input = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

        image_path = os.path.join(self.image_dir, f'{guid}.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.label_map.get(label, -1)
        return text_input, image, label

# 消融模型
class AblationModel(nn.Module):
    def __init__(self, use_text=True, use_image=True):
        super(AblationModel, self).__init__()
        self.use_text = use_text
        self.use_image = use_image

        if self.use_text:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert_fc = nn.Linear(self.bert.config.hidden_size, 256)

        if self.use_image:
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.fc = nn.Identity()

        input_size = 0
        if self.use_text:
            input_size += 256
        if self.use_image:
            input_size += 2048

        self.fc = nn.Linear(input_size, 3)

    def forward(self, text_input, image_input):
        features = []

        if self.use_text:
            text_features = self.bert(input_ids=text_input['input_ids'],
                                      attention_mask=text_input['attention_mask']).pooler_output
            text_features = self.bert_fc(text_features)
            features.append(text_features)

        if self.use_image:
            image_features = self.resnet(image_input)
            features.append(image_features)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

# 数据预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_file = 'train.txt'
image_dir = 'data'
train_dataset = MultimodalDataset(train_file, image_dir, tokenizer, transform)
train_data, val_data = train_test_split(train_dataset.data, test_size=0.2, random_state=42)
train_dataset.data = train_data
val_dataset = MultimodalDataset(train_file, image_dir, tokenizer, transform)
val_dataset.data = val_data

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# 训练与验证函数
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs=1):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for text_input, image_input, labels in tqdm(train_loader):
            labels = labels.to(DEVICE)
            if model.use_text:
                text_input = {key: value.to(DEVICE) for key, value in text_input.items()}
            if model.use_image:
                image_input = image_input.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(text_input, image_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {train_acc}")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for text_input, image_input, labels in tqdm(val_loader):
                labels = labels.to(DEVICE)
                if model.use_text:
                    text_input = {key: value.to(DEVICE) for key, value in text_input.items()}
                if model.use_image:
                    image_input = image_input.to(DEVICE)

                outputs = model(text_input, image_input)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_acc}")

# 消融实验主程序
if __name__ == '__main__':
    for ablation in ['text_only', 'image_only']:
        print(f"Running ablation experiment: {ablation}")
        use_text = ablation == 'text_only'
        use_image = ablation == 'image_only'

        model = AblationModel(use_text=use_text, use_image=use_image).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs=1)
