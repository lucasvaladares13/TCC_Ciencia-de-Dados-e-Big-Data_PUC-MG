import pandas as pd
import numpy as np
from _dictionary import sts_parameters, predict_parameters

path_DB = 'documentos\\DB_FUT\\'

dt_info = pd.read_excel(path_DB + 'dt_info.xlsx')
dt_partida = pd.read_excel(path_DB + 'dt_partida.xlsx')
dt_temporada = pd.read_excel(path_DB + 'dt_temporada.xlsx', dtype = {'Equipe_ID':np.int32})
dt_time_sts = pd.read_excel(path_DB + 'dt_time_sts.xlsx')
dt_time = pd.read_excel(path_DB + 'dt_time.xlsx')

def calculate_winner(dt_partida):
    dt_partida['Diferenca'] = dt_partida['m_placar'] - dt_partida['v_placar']
    dt_partida.loc[dt_partida['Diferenca'] == 0, 'winner'] = 2
    dt_partida.loc[dt_partida['Diferenca'] > 0, 'winner'] = 1
    dt_partida.loc[dt_partida['Diferenca'] < 0, 'winner'] = 3
    dt_partida = dt_partida.drop(['Diferenca'], axis = 'columns')

    return dt_partida

def calculate_score(dt_time_sts):
    dt_time_sts['Diferenca'] = dt_time_sts['GolsFeitos'] - dt_time_sts['GolsSofridos']
    dt_time_sts.loc[dt_time_sts['Diferenca'] == 0, 'score'] = 1
    dt_time_sts.loc[dt_time_sts['Diferenca'] > 0, 'score'] = 3
    dt_time_sts.loc[dt_time_sts['Diferenca'] < 0, 'score'] = 0
    dt_time_sts = dt_time_sts.drop(['Diferenca'], axis = 'columns')

    return dt_time_sts

def calculate_last_score(dt_time_sts,dt_time, dt_info,parameters,mdmv):
    dt_info = dt_info[['Partida_ID', 'Data']]
    dt_time_sts = pd.merge(dt_time_sts, dt_info, on= 'Partida_ID', how='inner')
    dt_time_sts['Data'] = pd.to_datetime(dt_time_sts['Data'],)

    df_sts = pd.DataFrame()
    for idx in dt_time.itertuples():
        try:
            nColumns = []
            df = dt_time_sts[dt_time_sts['Equipe_ID'] == idx.Equipe_ID ]
            df = df.sort_values(by=['Data'])
            for param in parameters:
                nColum = param+'_mdmv_'+str(mdmv)
                nColumns.append(nColum)
                df[nColum] = df[param].rolling(mdmv).sum()
                df = df.reset_index(drop=True)
                
                df[nColum] = df[nColum].fillna(df.loc[mdmv-1,nColum])
            #cols = ['Equipe_ID','Partida_ID','Data']
            #cols.extend(nColumns)
            #df = df[cols]
            df_sts = df_sts.append(df)
            #print(df)
        except:
            print(idx.Equipe_ID)
    df_sts = df_sts.drop(['Data'], axis = 'columns')
    df_sts = df_sts.reset_index(drop=True)
    return df_sts

def calculate_mdmv(mdmv, parameters,dt_time_sts,dt_info,dt_time):
    dt_info = dt_info[['Partida_ID', 'Data']]
    dt_time_sts = pd.merge(dt_time_sts, dt_info, on= 'Partida_ID', how='inner')
    dt_time_sts['Data'] = pd.to_datetime(dt_time_sts['Data'],)

    df_sts = pd.DataFrame()
    for idx in dt_time.itertuples():
        try:
            nColumns = ['Data']
            df = dt_time_sts[dt_time_sts['Equipe_ID'] == idx.Equipe_ID ]
            df = df.sort_values(by=['Data'])
            for param in parameters:
                nColum = param+'_mdmv_'+str(mdmv)
                nColumns.append(nColum)
                df[nColum] = df[param].rolling(mdmv).mean()
                df = df.reset_index(drop=True)
                
                df[nColum] = df[nColum].fillna(df.loc[mdmv-1,nColum])
            #cols = ['Equipe_ID','Partida_ID','Data']
            #cols.extend(nColumns)
            #df = df[cols]
            df_sts = df_sts.append(df)
            #print(df)
        except:
            print(idx.Equipe_ID)
    df_sts = df_sts.drop(['Data'], axis = 'columns')
    df_sts = df_sts.reset_index(drop=True)
    return df_sts   

def merge_time(df_partida,df_time,sufix):
    
    
    df_partida['Partida_Equipe_ID'] = df_partida[sufix+'LastPartida_ID'].astype(str) + "_" + df_partida[sufix+'Equipe_ID'].astype(str)
    df_time['Partida_Equipe_ID'] = df_time['Partida_ID'].astype(str) + "_" + df_time['Equipe_ID'].astype(str)
    df_time = df_time.drop(['Partida_ID','Equipe_ID'], axis = 'columns')
    dict_columns = {}
    for colum in df_time.columns:
        if colum not in ['Partida_Equipe_ID']:
            dict_columns[colum] = sufix+colum 
    
    df_partida = pd.merge(df_partida,df_time, how ='inner', on = 'Partida_Equipe_ID')
    df_partida = df_partida.rename(columns = dict_columns)
    df_partida = df_partida.drop(['Partida_Equipe_ID'], axis = 'columns')
    #print(df_partida.columns)
    return df_partida

def merge_time_atual(df_partida,df_time,sufix,Param):
    
    
    df_partida['Partida_Equipe_ID'] = df_partida['Partida_ID'].astype(str) + "_" + df_partida[sufix+'Equipe_ID'].astype(str)
    df_time['Partida_Equipe_ID'] = df_time['Partida_ID'].astype(str) + "_" + df_time['Equipe_ID'].astype(str)
    df_time = df_time.drop(['Partida_ID','Equipe_ID'], axis = 'columns')
    Param.append('Partida_Equipe_ID')
    print(Param)
    #print(df_time.columns)
    df_time = df_time[Param]
    Param.remove('Partida_Equipe_ID')
    dict_columns = {}
    for colum in df_time.columns:
        if colum not in ['Partida_Equipe_ID','Data']:
            dict_columns[colum] = sufix+'Atual'+colum 
    
    df_partida = pd.merge(df_partida,df_time, how ='left', on = 'Partida_Equipe_ID')
    df_partida = df_partida.rename(columns = dict_columns)
    #df_partida = df_partida.drop(['Data'], axis = 'columns')
    #print(df_partida.columns)
    return df_partida

def merge_aux(df_partida,df_temporada,sufix):
    
    if sufix == 'm_':
        type_info = '1'
    elif sufix == 'v_':
        type_info = '2'
    else:
        type_info = ''

    list_col = ['QtdJogadores', 'MdIdade', 'MdPosse', 'QtdJogosDisputados',
       'QtdGols', 'QtdAssistencias', 'QtdGolsNormais', 'QtdGolPenaltis',
       'QtdPenaltisBatidos', 'QtdCartoesAmarelos', 'QtdCartoesVermelhos',
       'MdGolsPorPartida', 'MdAssistenciasPorPartida',
       'MdGolsAssistenciasPorPartida', 'MdGolsNormaisPorPartida',
       'MdGolsNormaisAssistenciasPorPartida','key']

    df_temporada['key'] = df_temporada['Equipe_ID'].astype(str)+"_"+df_temporada['ano'].astype(str)+"_"+df_temporada['info_type'].astype(str)
    df_temporada = df_temporada[list_col]
    df_partida['ano_anterior'] = df_partida['ano_ref'] - 1
    df_partida[sufix+'key'] = df_partida[sufix+'Equipe_ID'].astype(str)+"_"+df_partida['ano_anterior'].astype(str)+"_"+type_info
    nCol = sufix+'key'
    df_partida = pd.merge(df_partida,df_temporada,how ='inner',left_on=nCol, right_on='key' )
    
    for col in list_col:
    
        df_partida = df_partida.rename(columns = {col:sufix+col})
    df_partida = df_partida.drop([nCol], axis = 'columns')
    return df_partida

def merge_ID(df_partida, df_LastPartida, sufix):
    df_partida['key'] = df_partida['Partida_ID'].astype(str) + "_" + df_partida[sufix+'Equipe_ID'].astype(str)
    df_partida = pd.merge(df_partida,df_LastPartida, on ='key', how = 'inner')
    nCol = sufix + 'LastPartida_ID'
    df_partida = df_partida.rename(columns = {'LastPartida_ID':nCol})
    df_partida = df_partida.drop(['key'], axis = 'columns')
    return df_partida

def get_last_partidaID(df_partida, df_info):
    df_partida = df_partida.sort_values(by = ['Partida_ID'])
    df_partida = df_partida.reset_index(drop=True)
    df_info = df_info[['Partida_ID','Data']]
    df_partida = df_partida[['Partida_ID','m_Equipe_ID','v_Equipe_ID']]
    df_partida = pd.merge(df_partida,df_info, on='Partida_ID',how='inner')

    df = df_partida[['Partida_ID','m_Equipe_ID','Data']].rename(columns = {'m_Equipe_ID':'Equipe_ID'})
    df = df.append(df_partida[['Partida_ID','v_Equipe_ID','Data']].rename(columns = {'v_Equipe_ID':'Equipe_ID'}))
    #df = df.sort_values(by = ['Partida_ID'])
    print(len(df))
    df_PartidaID = pd.DataFrame()
    for ID in list(set(df.Equipe_ID.tolist())):
        df_Equipe = df.loc[df.Equipe_ID == ID,]
        df_Equipe = df_Equipe.sort_values(by = ['Data'])
        df_Equipe['LastPartida_ID'] = df_Equipe.Partida_ID.shift(periods=1)

        df_PartidaID = df_PartidaID.append(df_Equipe)
        #print(df_Equipe)
    df_PartidaID['key'] = df_PartidaID['Partida_ID'].astype(str) + "_" + df_PartidaID['Equipe_ID'].astype(str)
    df_PartidaID = df_PartidaID[['key','LastPartida_ID']]
    df_PartidaID['LastPartida_ID'] = df_PartidaID['LastPartida_ID'].astype(str).str.replace('.0','',regex = False)
    return df_PartidaID


df_LastPartida = get_last_partidaID(dt_partida,dt_info)

df_sts = calculate_mdmv(10,['GolsFeitos','GolsSofridos','Faltas', 'Escanteios', 'Cruzamentos','QtdDefesas', 'Impedimentos','posse', 'chutesaogol'],dt_time_sts,dt_info,dt_time)
df_sts = calculate_score(df_sts)
df_sts = calculate_last_score(df_sts,dt_time,dt_info,['score'],10)
dt_partida = calculate_winner(dt_partida)
dt_partida = merge_ID(dt_partida, df_LastPartida,'m_')
dt_partida = merge_ID(dt_partida,df_LastPartida,'v_')
dt_partida = merge_time(dt_partida, df_sts ,"m_")
dt_partida = merge_time(dt_partida, df_sts ,"v_")

dt_partida = merge_aux(dt_partida,dt_temporada,'m_')
dt_partida = merge_aux(dt_partida,dt_temporada,'v_')
dt_partida = dt_partida.dropna()

dt_partida.to_excel(path_DB+'DadosTreinamento.xlsx', index = False)











