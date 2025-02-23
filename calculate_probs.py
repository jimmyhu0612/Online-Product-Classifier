import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# 讀取分類結果(各個產品對應機率)的Excel文件
file_path = './data/test_probabilities.xlsx'  # 替換為你的文件路徑
df = pd.read_excel(file_path, dtype=str)  # 以字符串格式讀取數據

# 確保probability_columns中的所有列都是數值類型，將非數值類型強制轉為NaN
probability_columns = df.columns[:16]  # 假設機率列是A到P列(總共16個class)
df[probability_columns] = df[probability_columns].apply(pd.to_numeric, errors='coerce')

query_column = 'Query'  # 假設Query列名為'Query'(產品名稱)

# 計算同一個Query下每個機率列的平均值
average_probabilities = df.groupby(query_column)[probability_columns].mean()

# 選擇每個Query下機率最高的列作為該Query的類別
average_probabilities['Predicted Label'] = average_probabilities.idxmax(axis=1)

# 保留True Label列
true_labels = df[[query_column, 'True Label']].drop_duplicates().set_index(query_column)
average_probabilities = average_probabilities.join(true_labels)

# 計算正確率
correct_predictions = (average_probabilities['Predicted Label'] == average_probabilities['True Label']).sum()
total_predictions = average_probabilities.shape[0]
accuracy = correct_predictions / total_predictions

print(f'正確率: {accuracy:.2%}')

# 計算混淆矩陣
conf_matrix = confusion_matrix(average_probabilities['True Label'], average_probabilities['Predicted Label'])
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# 標籤列表
labels = sorted(df['True Label'].unique())

# 繪製混淆矩陣
fig, ax = plt.subplots(figsize=(10, 7))
cax = ax.matshow(conf_matrix_normalized, cmap='Blues')

# 在每個格子中標註比率，保持兩位小數
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, f'{conf_matrix_normalized[i, j]:.2f}', va='center', ha='center', color='black')

# 設置坐標軸標籤
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# 將x軸標籤顯示在下方
ax.xaxis.set_ticks_position('bottom')

# 添加標題和標籤
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with Ratios')
plt.colorbar(cax)

plt.show()

# 將結果寫入新的Excel文件
output_file_path = './data/(加總平均)test_probabilities.xlsx'
average_probabilities.to_excel(output_file_path)

print(f'計算完成，結果已保存到 {output_file_path}')
