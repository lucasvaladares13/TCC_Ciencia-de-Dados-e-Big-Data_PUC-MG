# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup


class scrap_html():
    def __init__(self, url) -> None:
        super().__init__()
        self.flag = 0
        self.url = url
        self.r = requests.get(self.url)
        self.soup = BeautifulSoup(self.r.content, 'html5lib')

    def main_statistics(self):

        table = self.soup.find('div', attrs = {'id':'team_stats_extra'})
        
        quotes = []
        for row in table.findAll('div', ):
            quote = {}
            try: 
                var = str(row)
                var = (var.replace('<div>',"")
                .replace('div',"")
                .replace('>',"")
                .replace('<',"")
                .replace('\n',";")
                .replace(' ',"")
                .replace('\t',"")
                .replace('class="th"',"")
                .replace(' ',"")
                .replace(';;',";")
                .replace(';;',";")
                .replace('/;',";")
                .replace(';/',";")
                )
            
                quote['item'] = var
                quotes.append(quote)
            except TypeError:
                continue
        df = pd.DataFrame(quotes ,columns = ['item'])
        df = df.loc[df['item'].str.startswith(';'),].reset_index(drop = True)

        feature_list = ['Faltas','Escanteios','Cruzamentos','Contatos','Botedefensivo','Cortes','Jogadasaéreas','Defesas','Impedimentos','Tirodemeta','Cobrançadelateral','Bolaslongas']
        dict_sts = {}
        for idx, row in df.itertuples():
            list_item = str(row).split(';')
            #print(var)

            for item in list_item:
                if '//' in str(item):
                    item_var = str(item).split('/')
                    dict_sts['mandante'] = item_var[0]
                    dict_sts['visitante'] = item_var[-1]

                for feature in feature_list:
                    if str(feature) in str(item):
                        item_var = str(item).split('/')
                        dict_sts['mandante.' + str(feature)] = item_var[0]
                        dict_sts['visitante.' + str(feature)] = item_var[-1]
                           
        columns_dict = ['mandante', 'visitante', 'mandante.Faltas', 'visitante.Faltas', 'mandante.Escanteios', 'visitante.Escanteios', 'mandante.Cruzamentos', 'visitante.Cruzamentos', 'mandante.Contatos', 'visitante.Contatos', 'mandante.Botedefensivo', 'visitante.Botedefensivo', 'mandante.Cortes', 'visitante.Cortes', 'mandante.Defesas', 'visitante.Defesas', 'mandante.Impedimentos', 'visitante.Impedimentos', 'mandante.Tirodemeta', 'visitante.Tirodemeta', 'mandante.Bolaslongas', 'visitante.Bolaslongas']
        df_sts = pd.DataFrame([dict_sts])
        df_sts = df_sts.rename(columns={'mandante.Defesas':'mandante.QtdDefesas', 'visitante.Defesas':'visitante.QtdDefesas'})

        return df_sts
        
    def team_statistics(self):

        table = self.soup.find('div', attrs = {'id':'team_stats'})
        dict_column = [ {'column.statistics': 'mandante.posse'},
                {'column.statistics': 'visitante.posse'},
                {'column.statistics': 'mandante.acertodepasses'},
                {'column.statistics': 'visitante.acertodepasses'},
                {'column.statistics': 'mandante.chutesaogol'},
                {'column.statistics': 'visitante.chutesaogol'},
                {'column.statistics': 'mandante.defesasPorcentagem'},
                {'column.statistics': 'visitante.defesasPorcentagem'}]
        df_columns = pd.DataFrame(dict_column, columns = ['column.statistics'])
        #print(df_columns)
        quotes = []
        for row in table.findAll('strong'):
            quote = {}
            
            quote['item'] = row
            quotes.append(quote)

        df = pd.DataFrame(quotes ,columns = ['item'])
        df = df.reset_index(drop = True)
        df['column.statistics'] = df_columns['column.statistics']
        df = df.set_index('column.statistics')
        df = df.T
        df = df.reset_index(drop=True)
        return df
    
    def game_info(self):

        table = self.soup.find('div', attrs = {'id':'content'})
        quotes = []
        for row in table.findAll('h1'):
            quote = {}
            
            list_item = (str(row).replace('<br/>Clássico Majestoso','')
                                    .replace('<br/>Clássico dos Milhões','')
                                    .replace('<br/>Cl�ssico dos Milh�es','')
                                    .replace('<br/>Cl�ssico Majestoso','')
                                    .split('–')[-1]
                                    .replace('</h1>','')
                                    .replace(',','')
                                    .split(' ')
                            )
            quote['diadasemana'] = list_item[-4]
            quote['mes'] = list_item[-3]
            quote['dia'] = list_item[-2]
            quote['ano'] = list_item[-1]
            quotes.append(quote)
        df = pd.DataFrame(quotes ,columns = ['diadasemana','mes','dia','ano'])

        return df

    def score_game(self):

        table = self.soup.find('div', attrs = {'id':'content'})
        dict_column = [ {'column.': 'mandante.placar'},
                {'column.': 'visitante.placar'}]
        df_columns = pd.DataFrame(dict_column, columns = ['column.'])
        quotes = []
        for row in table.findAll('div',attrs= {'class':"score"}):
            quote = {}
            
            quote['placar'] = (str(row).replace('<div class="score">','')
                                    .replace('</div>',''))
            quotes.append(quote)


        quotes

        df = pd.DataFrame(quotes ,columns = ['placar'])
        df = df.reset_index(drop = True)

        df['column.'] = df_columns['column.']
        df = df.set_index('column.')
        df = df.T
        df = df.reset_index(drop=True)
        return df

    def champ_round(self):
 
        table = self.soup.find('div', attrs= {'class':'scorebox_meta'})
        quotes = []
        for row in table.findAll('div'):
            if ( 'a href' in str(row) and
                'strong' not in str(row) and
                '(' in str(row)

            ):
                quote = {}
                quote['item'] = (str(row).replace('</a>','')
                                        .replace('</div>','')
                                        .split('>')[-1]
                                        )
                quotes.append(quote)
        df = pd.DataFrame(quotes ,columns = ['item'])
        return df
        
    def get_resume(self):
        try:
            df_sts = self.main_statistics()
            df_stt = self.team_statistics()
            df_game = self.game_info()
            df_score = self.score_game()
            df_round = self.champ_round()


            df = pd.merge(df_sts, df_stt, left_index=True, right_index=True)
            df = pd.merge(df, df_game, left_index=True, right_index=True)
            df = pd.merge(df, df_score, left_index=True, right_index=True)
            df = pd.merge(df, df_round, left_index=True, right_index=True)
            return df
        except TypeError:
            print("erro")

        

class search_link():
    def __init__(self, url) -> None:
        super().__init__()
        self.flag = 0
        self.url = url

    def get_link(self):
        link_id = self.get_link_id()
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, 'html5lib')
        table = soup.find('div', attrs = {'id':'div_sched_' + link_id + '_1'})
        quotes = []
        for row in table.findAll('td', attrs = {'data-stat':'match_report'}):
            quote = {}
            try: 
                quote['link'] = row.a['href']
                quotes.append(quote)

            except TypeError:
                continue

        df = pd.DataFrame(quotes ,columns = ['link'])
        df['link'] = 'https://fbref.com' + df['link'].astype(str)
        return df

    def get_link_id(self):
        link_id = str(self.url).split('/')[6]
        #print(link_id)
        return link_id