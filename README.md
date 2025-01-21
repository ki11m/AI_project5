# **README**

本项目实现了一个多模态情感分析模型，结合文本和图像信息完成情感的三分类（positive, neutral, negative）。通过消融实验验证了不同模态（仅文本或仅图像）在情感分析中的作用。

---

## **项目结构**

```plaintext
├── data/                             # 存放训练文本和图片文件
├── train.txt                         # 训练数据的 guid 和情感标签
├── test_without_label.txt            # 测试数据的 guid，无情感标签
├── predictions_with_guid.txt         # 测试数据预测结果
├── main.py                           # 多模态模型的训练和验证代码
├── ablation.py                       # 消融实验代码
├── requirements.txt                  # 依赖库文件
├── README.md                         # 项目说明文档
```

---

## **运行环境**

运行本项目需要以下依赖环境：

- **Python** >= 3.8
- **依赖库**：
  - `torch` >= 1.11
  - `transformers` >= 4.11
  - `torchvision`
  - `Pillow`
  - `scikit-learn`
  - `tqdm`
  - `numpy`
  - `pandas`

### **安装依赖**
```bash
pip install -r requirements.txt
```

---

## **数据说明**

1. **数据文件**：
   - `train.txt`：训练集的 `guid` 和对应情感标签（格式为 `guid,tag`）。
   - `test_without_label.txt`：测试集的 `guid`，情感标签为空（格式为 `guid,`）。
   - 每个 `guid` 对应的文本文件（如 `guid.txt`）和图片文件（如 `guid.jpg`）需存放在 `data/` 目录中。

2. **标签说明**：
   - `positive`：0
   - `neutral`：1
   - `negative`：2

---

## **使用说明**

### **1. 多模态情感分析模型**

- **训练与验证**：
  使用以下命令启动多模态模型的训练与验证：

  ```bash
  python main.py
  ```

  **说明**：
  - 数据集会自动按 80%/20% 划分为训练集和验证集。
  - 验证集的准确率和损失会实时输出。

- **测试集预测**：
  使用以下命令对测试集进行预测：

  ```bash
  python main.py --mode predict
  ```

  预测结果保存在 `predictions_with_guid.txt` 文件中。

---

### **2. 消融实验**

- **运行消融实验**：
  验证仅使用文本或图像的模型性能：

  ```bash
  python ablation.py
  ```

  输出包括每种模型在训练和验证阶段的损失及准确率。

---

## **模型设计说明**

1. **多模态情感分析模型**：
   - **文本处理**：
     - 使用 `BERT` 提取文本特征，通过全连接层降维。
   - **图像处理**：
     - 使用 `ResNet-50` 提取图像特征。
   - **特征融合**：
     - 将文本和图像特征拼接后，输入全连接层完成分类。

2. **消融实验模型**：
   - 分别测试仅使用文本或图像的模型，用于量化不同模态对任务的贡献。

---

## **注意事项**

1. **GPU 支持**：建议使用支持 CUDA 的 GPU 加速模型训练和预测。
2. **数据完整性**：请确保 `data/` 目录下的文件和 `train.txt`、`test_without_label.txt` 中的 `guid` 对应。
3. **测试数据**：如需增加测试集，请按照相同格式更新 `test_without_label.txt`。

---

## **未来改进方向**

- 增加数据增强方法，提高模型泛化能力。
- 引入更先进的融合策略，如多头注意力机制。
- 调整超参数以优化模型性能。

---

如有任何问题，欢迎联系项目开发者。

--- 