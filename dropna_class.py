#刪除missing value
import pandas as pd

# 選擇路徑
file_path = './data/digital_combined.xlsx'

# 將excel讀進DataFrame
df = pd.read_excel(file_path)

# 將class那欄的缺失值刪除
df_cleaned = df.dropna(subset=['class'])

# 儲存excel檔
output_file_path = './data/(deal)digital_combined.xlsx'
df_cleaned.to_excel(output_file_path, index=False)

print(f"Cleaned file saved to {output_file_path}")
#Downsampling
import pandas as pd

# 讀取
file_path = './data/Book_merged.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 檢查資料列數是否足夠
if len(df) < 244:
    raise ValueError("The data has less than 500 rows. Please provide a dataset with at least 500 rows.")

# 隨機抽取244列資料
df_sample = df.sample(n=244, random_state=1)  # 使用 random_state 确保结果可重复

# 儲存
output_file_path = './data/(test)Sample_book.xlsx'  # 替换为你希望保存的文件路径
df_sample.to_excel(output_file_path, index=False)

print(f"Sampled file saved to {output_file_path}")
import os
import pandas as pd

# 設定欲合併的路徑與輸出路徑
root_dir = './GCP'
output_file = './data/merged_sample.xlsx'

# 初始化DataFrame
merged_df = pd.DataFrame()

# 遍歷欲合併的路徑中所有資料夾
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(subdir, file)
            # 讀取excel，並拿取前244列資料(包含header)
            df = pd.read_excel(file_path)
            sample_df = df.head(244)
            # 合併資料
            merged_df = pd.concat([merged_df, sample_df], ignore_index=True)

# 保存合併的DataFrame到新的檔案
merged_df.to_excel(output_file, index=False)
print(f'所有xlsx文件已合併並儲存為 {output_file}')

