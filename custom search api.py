import pandas as pd
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
import time

API_KEY = '' #輸入gcp的API key
SEARCH_ENGINE_ID = ''  # 填入搜索引擎ID
RESULTS_PER_PAGE = 10  # 一頁要幾個搜尋結果
DELAY_SECONDS = 3  # 每次搜尋要等多久

def search_google(query, page):
    start = (page - 1) * RESULTS_PER_PAGE + 1

    service = build('customsearch', 'v1', developerKey=API_KEY)
    response = service.cse().list(
        cx=SEARCH_ENGINE_ID,
        q=query,
        start=start,
        num=RESULTS_PER_PAGE,
        fields='items(title,link,snippet)'
    ).execute()

    if 'items' in response:
        return response['items']
    else:
        return []

# 讀取各類別產品的excel(建議各類別，單次使用卻有大量搜尋時易被擋)
df = pd.read_excel('C:/Users/NCHU-MK-DS/Desktop/hu/ReScraper/new/book/Sample_book.xlsx')

# 建立新的DataFrame來儲存搜尋結果以及原本的query
merged_data = []

for index, row in df.iterrows():
    prodID = row['prodID']
    web_scraper_start_url = row['web-scraper-start-url']
    query = row['prodName']
    prodName_href = row['prodName-href']
    company = row['company']
    prodName_ = row['prodName_']
    price = row['price']
    slogan = row['slogan']
    classification_overall = row['class(overall)']
    image1_src = row['image1-src']
    image2_src = row['image2-src']
    #image3_src = row['image3-src']
    #image4_src = row['image4-src']
    #image5_src = row['image5-src']
    specification = row['specification']
    label = row['class']

    if pd.notnull(query):  # 檢查商品是否存在
        results = search_google(query, 1)  # 要爬取幾頁的資料，這邊設定為一頁
        print(query)
        for result in results:
            search_title = result.get('title', '')
            link = result.get('link', '')
            snippet = result.get('snippet', '')
            merged_data.append([
                prodID, query, prodName_, label, company, search_title, link, snippet
            ])

        time.sleep(DELAY_SECONDS)

#建立新的DataFrame並包含搜尋結果與原本的query
columns = [
    'prodID', 'query', 'prodName_', 'Label', 'company', 'search_title', 'link', 'snippet'
]
merged_df = pd.DataFrame(merged_data, columns=columns)

# 儲存
merged_df.to_excel('./data/gcp_Sample_book.xlsx', index=False)

