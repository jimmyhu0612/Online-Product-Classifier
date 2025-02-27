# Online-Product-Classifier
本研究旨在開發一個適用於電商平台和第三方金流平台的產品分類模型，解決產品分類不一致與資料不足的問題。  
從電商平台蒐集商品資料，在僅擁有產品名稱的情況下從外部資料(GCP的Custom Search API)獲取額外的產品資訊，並使用BERT模型進行詞嵌入，將搜尋結果轉換為向量表示。接著，使用LSTM模型進行訓練，並通過超參數調整和早停法來優化模型性能。  
在追隨某大型電商平台的分類規則下，16種產品種類的分類準確率達80.57%。

# dropna_class.py
爬蟲取得的資料可能因產品售完、下架而導致無法取得完整資訊(產品名稱、類別)，因此需刪除未含有完整類別的產品

# custom search api.py：在本步驟中，我們須將欲處理之產品名稱皆使用於這支程式，得到豐富訓練資料
請先至程式化搜尋引擎(Programmable Search Engine)建立一搜尋引擎，詳情請至https://developers.google.com/custom-search
將API KEY、Search Engine ID輸入至程式的變數當中，
透過這支程式，我們在僅使用產品名稱的情況下，得到產品的**10筆**搜尋結果(Snippet)，並儲存於另一excel

# divide into 3 dataset.py：在本支程式中，我們將程式分割為訓練、驗證、測試集
**WARNING** 請確保不會有資料洩漏(data leakage)，偷看的情況發生；且需避免class imbalance的問題
將資料集進行切割，以進行模型的運算

# model.py：進行模型的建立、測量與視覺化
在這支程式中，使用了bert-base-chinese的tokenizer，並且使用LSTM進行建模時使用早停法(patience = 3)，超參數(hyperparameter)的部分使用grid search來找出最佳的參數組合，最後保留最佳超參數並輸出分類錯誤結果並繪製混淆矩陣

# calculate_probs.py：機率加總並平均
由於在custom search api.py這支程式中，1個產品可以得到10筆搜尋結果(snippet)，因此我們對於這10筆的搜尋結果進行加總平均時，可以減低分類錯誤所造成的影響，並提高模型分類的準確度
