import pandas as pd
from _scrap import scrap_html as sc
from _scrap import search_link as src
from _font_list import font_list
import time

start_time = time.time()

def get_data(champ_link):
    file_name = str(champ_link).split('/')[-1] + '.xlsx'
    
    df_link = src(champ_link).get_link()
    #df_link = df_link[0:2]
    database = pd.DataFrame()

    for idx, row in df_link.itertuples():
        try:
            link = str(df_link.loc[idx,'link'])
            print(link)
            df = sc(link).get_resume()
            database = database.append(df, ignore_index= True)
        except:
            print('---------------------------------')
    
    database.to_excel('documentos\\RAW\\01_SCRAP\\' + file_name, index= False)

for url in font_list:
    get_data(url)

print((time.time() - start_time)/60)
