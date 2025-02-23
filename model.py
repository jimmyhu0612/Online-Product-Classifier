import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 讀取資料集
train_df = pd.read_excel('./data/階段3：切割完的資料集/train_data.xlsx')
val_df = pd.read_excel('./data/階段3：切割完的資料集/validation_data.xlsx')
test_df = pd.read_excel('./data/階段3：切割完的資料集/test_data.xlsx')

# 設定超參數
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 50 
LEARNING_RATE = 1e-4 #模型學習率
PATIENCE = 3 #早停法

# word embedding的模型設定
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

# 將DataFrame的"snippet"欄讀取，並將該欄進行word embedding
train_texts = tokenize_texts(train_df['snippet'].tolist())
val_texts = tokenize_texts(val_df['snippet'].tolist())
test_texts = tokenize_texts(test_df['snippet'].tolist())

label_encoder = LabelEncoder() #將label編碼
train_labels = torch.tensor(label_encoder.fit_transform(train_df['Label']), dtype=torch.long) #label的編碼轉為長整數
val_labels = torch.tensor(label_encoder.transform(val_df['Label']), dtype=torch.long)
test_labels = torch.tensor(label_encoder.transform(test_df['Label']), dtype=torch.long)

# 建立數據集和dataloader
train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)
test_dataset = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型本人
class BERTLSTMClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(BERTLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(768, 128, batch_first=True) #input 768-dim; output 128-dim
        self.dropout = nn.Dropout(dropout_rate) #套入dropout
        self.fc = nn.Linear(128, num_classes) #全連接層來對應class的個數
        self.softmax = nn.Softmax(dim=1) #Softmax對應16種類別

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_output, _ = self.lstm(bert_output)
        lstm_output = lstm_output[:, -1, :] #僅拿取最後一個time step的輸出
        dropout_output = self.dropout(lstm_output)
        fc_output = self.fc(dropout_output)
        return self.softmax(fc_output)

num_classes = len(label_encoder.classes_) #得到要有幾個class
model = BERTLSTMClassifier(num_classes)
criterion = nn.CrossEntropyLoss() #loss值計算方式
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #optimizer使用Adam

# 訓練模型並使用GPU運行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
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
    train_losses.append(train_loss)
    train_accuracy = accuracy_score(train_targets, train_preds)
    train_accuracies.append(train_accuracy)

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
    val_losses.append(val_loss)
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE: #早停法
        print('Early stopping')
        break

# 繪製訓練和驗證的LOSS值和ACCURACY值圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')

plt.show()

# 使用最好的模型
model.load_state_dict(torch.load('best_model.pth'))

# 在最好的模型上評估測試集並繪製混淆矩陣
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

# 計算混淆矩陣(算比率而非筆數)
conf_matrix = confusion_matrix(test_targets, test_preds)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# 繪製混淆矩陣
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(conf_matrix_normalized, cmap=plt.cm.Blues)
fig.colorbar(cax)

# 設定標籤
ax.set_xticks(np.arange(len(label_encoder.classes_)))
ax.set_yticks(np.arange(len(label_encoder.classes_)))
ax.set_xticklabels(label_encoder.classes_, rotation=90)
ax.set_yticklabels(label_encoder.classes_)
ax.xaxis.set_ticks_position('bottom')

# 標註數值
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

misclassified_df.to_excel('C:/Users/NCHU-MK-DS/Desktop/hu/ReScraper/0626切割/0629/misclassified_results.xlsx', index=False)
print('Misclassified results saved to misclassified_results.xlsx')

# 儲存每個產品的機率分佈
probabilities_df = pd.DataFrame(test_probabilities, columns=label_encoder.classes_)
probabilities_df['Text'] = test_df['snippet']
probabilities_df['True Label'] = label_encoder.inverse_transform(test_targets)
probabilities_df['Predicted Label'] = label_encoder.inverse_transform(test_preds)
probabilities_df['Query'] = test_df['query']  # 新增的 "query" 列

probabilities_df.to_excel('./data/test_probabilities.xlsx', index=False)
print('Test probabilities saved to test_probabilities.xlsx')

