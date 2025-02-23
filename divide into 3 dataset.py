# 定義要切割的比例
train_proportion = 0.8
val_proportion = 0.1
test_proportion = 0.1

# 得到所有query(未重複)的list
unique_queries = list(grouped.groups.keys())

# Shuffle這些query
import random
random.seed(42)
random.shuffle(unique_queries)

# 計算三個資料集的數量
total_queries = len(unique_queries)
train_size = int(total_queries * train_proportion)
val_size = int(total_queries * val_proportion)
test_size = total_queries - train_size - val_size

# 分配query到這些資料集
train_queries = unique_queries[:train_size]
val_queries = unique_queries[train_size:train_size + val_size]
test_queries = unique_queries[train_size + val_size:]

# 建立過濾的條件：確保同一query僅存在單一資料集(避免偷看、data leakage)
train_mask = data['query'].isin(train_queries)
val_mask = data['query'].isin(val_queries)
test_mask = data['query'].isin(test_queries)

# 建立三資料集
train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

# 各資料集中擁有幾筆資料
train_count = train_data.shape[0]
val_count = val_data.shape[0]
test_count = test_data.shape[0]
#顯示結果(筆數)
import ace_tools as tools; tools.display_dataframe_to_user(name="Training Dataset", dataframe=train_data)
tools.display_dataframe_to_user(name="Validation Dataset", dataframe=val_data)
tools.display_dataframe_to_user(name="Testing Dataset", dataframe=test_data)

train_count, val_count, test_count
