import pandas as pd
import os
import pathlib
from datetime import datetime
import numpy as np
from _dictionary import int_column,dict_time,dict_day

today = datetime.today().strftime('%Y_%m_%d')

path_ = 'documentos\\'
path_folder = path_ + 'RAW\\01_SCRAP\\'
file_list = list(pathlib.Path(path_folder).glob("**/*xlsx"))

df = pd.DataFrame()
for file_path in file_list:
    if ('old' not in str(file_path)) and ('fbref_compiled_' not in str(file_path)):
        df_file = pd.read_excel(file_path)
        file = str(file_path).split('\\')[-1]
        ano_ref = file.split('-')[0]
        df_file['file'] = file
        df_file['ano_ref'] = ano_ref
        df = df.append(df_file)

df_columns = list(df.columns)
for column in int_column:
    df[column] = (df[column]
                    .astype(str)
                    .str
                    .replace('<strong>','', regex = False)
                    .str
                    .replace('%</strong>','', regex = False)
                    .str
                    .replace('.0','', regex = False)
    )



df = (df.fillna(0)
        .replace('nan',0)
        .replace('Contatos',0)
        .replace('Cruzamentos',0)
        .replace(dict_time)
        .replace(dict_day)
)
for column in int_column:
    df[column] = pd.to_numeric(df[column])
df['item'] = df['item'].astype(str).str.replace('S�rie','Serie', regex = False)

for col in df.columns:
    nCol = str(col).replace('mandante.','m_').replace('visitante.','v_')
    df = df.rename(columns = {col:nCol})

df = df.reset_index(drop =True)

path_folder = path_ + 'PROCESSED\\01_SCRAP\\'
df.to_excel(path_folder+'fbref_compiled_'+today+'.xlsx', index = False)




