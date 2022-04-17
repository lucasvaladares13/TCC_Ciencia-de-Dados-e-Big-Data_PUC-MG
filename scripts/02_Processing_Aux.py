import pandas as pd
import os
import pathlib
from datetime import datetime
import numpy as np
from _dictionary import dict_aux_file, aux_column, dict_time, time_id

today = datetime.today().strftime('%Y_%m_%d')

path = 'documentos\\'
path_folder = path + 'RAW\\02_AUX\\'

file_list = list(pathlib.Path(path_folder).glob("**/*txt"))

df = pd.DataFrame()
for file in file_list:
    df_comp = pd.read_csv(file, usecols=aux_column, sep = '\t')
    df_comp['ano'] = str(file).split('\\')[-1].split('_')[-1].replace('.txt','')
    df_comp['serie'] = str(file).split('\\')[-1].split('_')[-2]
    df = df.append(df_comp)

df.loc[~df['Equipe'].str.contains('vs'),'info_type'] = 1
df.loc[df['Equipe'].str.contains('vs'),'info_type'] = 2
df['Equipe'] = df['Equipe'].astype(str).str.replace('vs ','',regex = True)

df = df.rename(columns= dict_aux_file)

df = (df.fillna(0)
        .replace('nan',0)
        .replace(dict_time)
)
df = df.reset_index(drop =True)
df_id = pd.DataFrame(list(time_id.items()), columns = ['Equipe','Equipe_ID'])
print(df_id)
df = pd.merge(df,df_id, how = 'left', on='Equipe')
path_folder = path + '\\PROCESSED\\02_AUX\\'
print(df)
df.to_excel(path_folder+'fbref_compiled_aux_'+today+'.xlsx', index = False)

path_folder = path + '\\DB_FUT\\'

df.to_excel(path_folder+'dt_temporada.xlsx', index = False)
