import pandas as pd
import datetime as dt
from _dictionary import (int_column,
                         time_id, 
                         col_dt_partida, 
                         col_dt_mandante, 
                         col_dt_visitante, 
                         col_dt_info, 
                         dict_dt_mandante,
                         dict_dt_visitante,
                         dict_dt_info
                            )

path = 'documentos\\PROCESSED\\01_SCRAP\\'
path_DB = 'documentos\\DB_FUT\\'
file = 'fbref_compiled_2022_02_19.xlsx'

def create_partida_id(df):

    df = df.reset_index(drop =True)
    df['Partida_ID'] = df.index
    df['Partida_ID'] = df['Partida_ID'] + 1

    return df
def create_equipe_id(df,dict_id):
    
    df_id = pd.DataFrame(list(dict_id.items()), columns = ['Equipe','Equipe_ID'])
    df = pd.merge(df,df_id, how = 'left', left_on='mandante',right_on='Equipe')
    df = df.rename(columns={'Equipe':'m_Equipe','Equipe_ID':'m_Equipe_ID'})
    df = pd.merge(df,df_id, how = 'left', left_on='visitante',right_on='Equipe')
    df = df.rename(columns={'Equipe':'v_Equipe','Equipe_ID':'v_Equipe_ID'})
    
    return df

def dt_info(df, col, dict_info, path):
    file_name = 'dt_info.xlsx'
    dt = df[col].drop_duplicates()
    dt['mes_num'] = dt['mes'].replace(dict_info)
    dt['Data'] = dt['dia'].astype(str)+'/'+dt['mes_num'].astype(str)+'/'+dt['ano'].astype(str)
    dt['Data'] = pd.to_datetime(dt['Data'])
    dt.to_excel(path+file_name, index = False)

def dt_partida(df,col,path):
    file_name = 'dt_partida.xlsx'
    dt = df[col].drop_duplicates()
    dt.to_excel(path+file_name, index = False)

def dt_time_sts(df,colm,dictm,colv,dictv,path):
    file_name = 'dt_time_sts.xlsx'
    dt_m = df[colm].rename(columns = dictm )
    dt_v = df[colv].rename(columns = dictv )
    dt = dt_m.append(dt_v)
    dt.to_excel(path+file_name, index = False)

def dt_time(dict_id,path):
    file_name = 'dt_time.xlsx'
    dt = pd.DataFrame(list(dict_id.items()), columns = ['Equipe','Equipe_ID'])
    dt.to_excel(path+file_name, index = False)

df = pd.read_excel(path+file)

df = create_partida_id(df)
df = create_equipe_id(df,time_id)

dt_info(df,col_dt_info, dict_dt_info, path_DB)
dt_partida(df,col_dt_partida,path_DB)
dt_time_sts(df,col_dt_mandante,dict_dt_mandante,col_dt_visitante,dict_dt_visitante,path_DB)
dt_time(time_id,path_DB)