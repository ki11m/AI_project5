import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import chardet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_text(text_path):
    try:
        # 自动检测文件编码
        with open(text_path, 'rb') as f:  # 使用二进制模式读取文件
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        # 使用检测到的编码读取文件，忽略解码错误
        with open(text_path, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {text_path}: {e}")
        return ""

# 1. 数据集类 (MultimodalDataset)
class MultimodalDataset(Dataset):
    def __init__(self, text_file, image_dir, tokenizer, transform=None):
        self.text_file = text_file
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.data = self.load_data()

        # 标签映射字典
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}

    def load_data(self):
        """加载数据，跳过文件第一行（标题行）"""
        data = []
        try:
            with open(self.text_file, 'r') as f:
                lines = f.readlines()

            # 从第二行开始读取
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) != 2:
                    print(f"Skipping invalid line: {line}")
                    continue

                guid, label = parts
                data.append((guid, label))
        except Exception as e:
            print(f"Error loading data from {self.text_file}: {e}")
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


# 2. 多模态模型 (MultimodalModel)
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()

        # 文本模型（BERT）
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_fc = nn.Linear(self.bert.config.hidden_size, 256)

        # 图像模型
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 替换全连接层为 Identity

        # 融合层
        self.fc = nn.Linear(256 + 2048, 3)

    def forward(self, text_input, image_input):
        text_features = self.bert(input_ids=text_input['input_ids'],
                                  attention_mask=text_input['attention_mask']).pooler_output
        text_features = self.bert_fc(text_features)

        image_features = self.resnet(image_input)

        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined_features)
        return output


# 3. 数据加载与预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_file = 'train.txt'
image_dir = 'data'
train_dataset = MultimodalDataset(train_file, image_dir, tokenizer, transform)

train_data, val_data = train_test_split(train_dataset.data, test_size=0.2, random_state=42)
train_dataset.data = train_data
val_dataset = MultimodalDataset(train_file, image_dir, tokenizer, transform)
val_dataset.data = val_data

# 创建DataLoader时增加num_workers
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# 4. 训练与评估函数
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=1):
    scaler = torch.cuda.amp.GradScaler()  # 自动混合精度

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for text_input, image_input, labels in tqdm(train_loader):
            labels = torch.tensor([int(label) for label in labels]).to(DEVICE)
            text_input = {key: value.to(DEVICE) for key, value in text_input.items()}
            image_input = image_input.to(DEVICE)

            optimizer.zero_grad()

            # 自动混合精度
            with torch.cuda.amp.autocast():
                output = model(text_input, image_input)
                loss = criterion(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        train_accuracy = correct_predictions / total_predictions
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {train_accuracy}")

        # 验证
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for text_input, image_input, labels in tqdm(val_loader):
                labels = torch.tensor([int(label) for label in labels]).to(DEVICE)
                text_input = {key: value.to(DEVICE) for key, value in text_input.items()}
                image_input = image_input.to(DEVICE)

                output = model(text_input, image_input)
                loss = criterion(output, labels)
                val_loss += loss.item()

                _, predicted = torch.max(output, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        val_accuracy = correct_predictions / total_predictions
        print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}")

# 5. 预测函数
def predict_with_guid(model, test_loader, input_file, output_file):
    """
    预测函数，读取 GUID 并输出预测结果到指定文件。
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for text_input, image_input, _ in tqdm(test_loader):
            output = model(text_input, image_input)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.tolist())

    # 标签映射
    label_map = {0: "positive", 1: "neutral", 2: "negative"}
    predictions = [label_map[pred] for pred in predictions]

    # 读取输入文件获取 GUID
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 输出结果到文件
    with open(output_file, 'w') as f:
        f.write("guid,tag\n")  # 写入标题行
        for i, line in enumerate(lines[1:]):  # 跳过标题行
            guid, _ = line.strip().split(',')
            tag = predictions[i]  # 根据索引获取预测标签
            f.write(f"{guid},{tag}\n")

# 6. 训练与评估
if __name__ == '__main__':
    # 检查是否有可用的GPU，如果有，则使用GPU，否则使用CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = MultimodalModel().to(DEVICE)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    train(model, train_loader, val_loader, optimizer, criterion, num_epochs=3)

    # 预测测试集
    test_file = 'test_without_label.txt'  # 测试集文件路径
    output_file = 'predictions_with_guid.txt'  # 输出文件路径

    # 创建测试集 DataLoader
    test_dataset = MultimodalDataset(test_file, image_dir, tokenizer, transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 获取预测结果
    predict_with_guid(model, test_loader, test_file, output_file)
