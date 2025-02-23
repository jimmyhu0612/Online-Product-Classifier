import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# 讀取資料集
train_df = pd.read_excel('./data/train_data.xlsx')
val_df = pd.read_excel('./data/validation_data.xlsx')
test_df = pd.read_excel('./data/test_data.xlsx')

# 設置超參數
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 3

# 定義要使用的tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.texts['input_ids'][idx],
            'attention_mask': self.texts['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def tokenize_texts(texts, max_length=MAX_LENGTH):
    tokens = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )
    return tokens

# 將訓練集的 'snippet' 欄位轉換為 BERT 可接受的格式
# tokenize_texts 函數會將文字進行分詞、截斷、填充等操作，轉換為 BERT 輸入格式
train_texts = tokenize_texts(train_df['snippet'].tolist())
val_texts = tokenize_texts(val_df['snippet'].tolist())
test_texts = tokenize_texts(test_df['snippet'].tolist())

label_encoder = LabelEncoder()
train_labels = torch.tensor(label_encoder.fit_transform(train_df['Label']), dtype=torch.long)
val_labels = torch.tensor(label_encoder.transform(val_df['Label']), dtype=torch.long)
test_labels = torch.tensor(label_encoder.transform(test_df['Label']), dtype=torch.long)

# 建立資料集與data loader
train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)
test_dataset = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 建立模型
class BERTLSTMClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=1, dropout_rate=0.5):
        super(BERTLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(768, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_output, _ = self.lstm(bert_output)
        lstm_output = lstm_output[:, -1, :]
        dropout_output = self.dropout(lstm_output)
        fc_output = self.fc(dropout_output)
        return self.softmax(fc_output)

# 定義訓練與驗證模型的函數
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS, patience=PATIENCE):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_accuracy = accuracy_score(train_targets, train_preds)

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
                val_targets.extend(labels.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_targets, val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_wts = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_wts)
    return model, train_accuracy, val_accuracy, val_loss

# 定義grid search所要設定的參數
param_grid = {
    'hidden_size': [128, 256],
    'learning_rate': [1e-4, 1e-5],
    'num_layers': [1, 2, 3],
    'dropout_rate': [0.3, 0.5]
}

# 使用ParameterGrid()將參數字典轉換為所有可能的參數組合列表
param_list = list(ParameterGrid(param_grid))

# 保存最好的超參數與成效
best_params = None
best_score = 0.0

# CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用每個參數組合進行訓練和評估
for params in param_list:
    print(f"Evaluating params: {params}")

    # 更新模型參數
    HIDDEN_SIZE = params['hidden_size']
    LEARNING_RATE = params['learning_rate']
    NUM_LAYERS = params['num_layers']
    DROPOUT_RATE = params['dropout_rate']

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(label_encoder.classes_)
    model = BERTLSTMClassifier(num_classes, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model, train_acc, val_acc, val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer)

    print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

    if val_acc > best_score:
        best_score = val_acc
        best_params = params

print(f"Best params: {best_params}, Best score: {best_score:.4f}")

# 使用最佳超參數在訊連集和驗證集上重新訓練模型
HIDDEN_SIZE = best_params['hidden_size']
LEARNING_RATE = best_params['learning_rate']
NUM_LAYERS = best_params['num_layers']
DROPOUT_RATE = best_params['dropout_rate']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = BERTLSTMClassifier(num_classes, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model, train_acc, val_acc, val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer)

# 儲存模型權重
torch.save(model.state_dict(), 'best_model.pth')

# 載入模型權重
model.load_state_dict(torch.load('best_model.pth'))

# 在測試集上評估模型並繪製混淆矩陣
model.eval()
test_preds = []
test_targets = []
test_probabilities = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        test_preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
        test_targets.extend(labels.detach().cpu().numpy())
        test_probabilities.extend(outputs.detach().cpu().numpy())

# 計算混淆矩陣
conf_matrix = confusion_matrix(test_targets, test_preds)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# 繪製混淆矩陣
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(conf_matrix_normalized, cmap=plt.cm.Blues)
fig.colorbar(cax)

# 設定圖表系解
ax.set_xticks(np.arange(len(label_encoder.classes_)))
ax.set_yticks(np.arange(len(label_encoder.classes_)))
ax.set_xticklabels(label_encoder.classes_, rotation=90)
ax.set_yticklabels(label_encoder.classes_)
ax.xaxis.set_ticks_position('bottom')

# 標上數值
for i in range(len(label_encoder.classes_)):
    for j in range(len(label_encoder.classes_)):
        ax.text(j, i, f'{conf_matrix_normalized[i, j]:.2f}', ha='center', va='center', color='black')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Normalized)')
plt.show()

# 計算測試集的準確率
test_accuracy = accuracy_score(test_targets, test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')

# 儲存錯誤分類的結果
misclassified_indices = [i for i, (true, pred) in enumerate(zip(test_targets, test_preds)) if true != pred]
misclassified_texts = [test_df.iloc[i]['snippet'] for i in misclassified_indices]
misclassified_true_labels = [test_targets[i] for i in misclassified_indices]
misclassified_pred_labels = [test_preds[i] for i in misclassified_indices]

misclassified_df = pd.DataFrame({
    'Text': misclassified_texts,
    'True Label': label_encoder.inverse_transform(misclassified_true_labels),
    'Predicted Label': label_encoder.inverse_transform(misclassified_pred_labels)
})

misclassified_df.to_excel('./data/misclassified_results.xlsx', index=False)
print('Misclassified results saved to misclassified_results.xlsx')

# 儲存每個產品的機率分佈
probabilities_df = pd.DataFrame(test_probabilities, columns=label_encoder.classes_)
probabilities_df['Text'] = test_df['snippet']
probabilities_df['True Label'] = label_encoder.inverse_transform(test_targets)
probabilities_df['Predicted Label'] = label_encoder.inverse_transform(test_preds)
probabilities_df['Query'] = test_df['query']  # 新增的 "query" 列

probabilities_df.to_excel('./data/test_probabilities.xlsx', index=False)
print('Test probabilities saved to test_probabilities.xlsx')
