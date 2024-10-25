# AUTHOR: Bhushan Oza
import pandas as pd
import datetime                            
import os
from datetime import datetime
import numpy as np
from math import log

def populate(input_file_path:str='data/input.csv'):
    path = '.'
    filepaths = [f for f in os.listdir(path) if f.startswith(input_file_path)]
    df = pd.concat(map(pd.read_csv, filepaths))
    # process
    df = df[['Date-Time','Type','Price','Volume','Qualifiers','#RIC', 'Bid Price', 'Ask Price', 'Bid Size', 'Ask Size']]
    df = df.drop(df[df.Type != 'Trade'].index)
    df = df.drop(df[pd.isna(df.Price)].index)
    df['PV'] = df.Price * df.Volume
    for i in range(len(df['Date-Time'])):
        df.iat[i, df.columns.get_loc('Date-Time')] = (df.iat[i, df.columns.get_loc('Date-Time')])[0:10]+' '+(df.iat[i, df.columns.get_loc('Date-Time')])[11:19]
    # scan
    period = input('Enter aggregation period in minutes: ')
    print('Selected period is '+str(period)+' minutes.')  
    # calculate
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    newDf = pd.DataFrame()
    newDf['Sec'] = df['#RIC'][1].split('.')[0]
    newDf = newDf.fillna(0)
    newDf = newDf.drop(newDf[newDf.TradeCount <= 0].index)   
    return df, newDf

def dollarVolumeTraded(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_dollarVolumeTraded.csv'):
    df, newDf = populate(input_file_path)
    summed = df.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_b = df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_s = df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    newDf['DollarVolumeTraded'] = summed.iloc[:,2]
    cats = ['b','s']
    for i in cats:
        if i=='b':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_b.iloc[:,2]
        elif i=='s':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_s.iloc[:,2]
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf
    
def shareVolumeTraded(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_shareVolumeTraded.csv'):
    df, newDf = populate(input_file_path)
    summed = df.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_b = df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_s = df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    cats = ['b','s']
    newDf['ShareVolumeTraded'] = summed.iloc[:,1]
    for i in cats:
        if i=='b':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_b.iloc[:,1]
        elif i=='s':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_s.iloc[:,1]
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf  

def vWAP(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_vWAP.csv'):
    df, newDf = populate(input_file_path)
    summed = df.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_b = df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_s = df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    newDf['DollarVolumeTraded'] = summed.iloc[:,2]
    cats = ['b','s']
    for i in cats:
        if i=='b':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_b.iloc[:,2]
        elif i=='s':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_s.iloc[:,2]
    newDf['ShareVolumeTraded'] = summed.iloc[:,1]
    for i in cats:
        if i=='b':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_b.iloc[:,1]
        elif i=='s':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_s.iloc[:,1]
    newDf['VWAP'] = newDf.DollarVolumeTraded / newDf.ShareVolumeTraded
    for i in cats:
        if i=='b':
            newDf['VWAP_<'+i+'>'] = newDf['DollarVolumeTraded_<b>'] / newDf['ShareVolumeTraded_<b>']
        elif i=='s':
            newDf['VWAP_<'+i+'>'] = newDf['DollarVolumeTraded_<s>'] / newDf['ShareVolumeTraded_<s>']  
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf

def arithmeticReturn(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_arithmeticReturn.csv'):
    df, newDf = populate(input_file_path)
    summed = df.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_b = df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_s = df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    newDf['DollarVolumeTraded'] = summed.iloc[:,2]
    cats = ['b','s']
    for i in cats:
        if i=='b':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_b.iloc[:,2]
        elif i=='s':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_s.iloc[:,2]
    newDf['ShareVolumeTraded'] = summed.iloc[:,1]
    for i in cats:
        if i=='b':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_b.iloc[:,1]
        elif i=='s':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_s.iloc[:,1]
    newDf['VWAP'] = newDf.DollarVolumeTraded / newDf.ShareVolumeTraded
    for i in cats:
        if i=='b':
            newDf['VWAP_<'+i+'>'] = newDf['DollarVolumeTraded_<b>'] / newDf['ShareVolumeTraded_<b>']
        elif i=='s':
            newDf['VWAP_<'+i+'>'] = newDf['DollarVolumeTraded_<s>'] / newDf['ShareVolumeTraded_<s>']  
    newDf['AReturn'] = newDf.VWAP.pct_change()
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf

def logReturn(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_logReturn.csv'):
    df, newDf = populate(input_file_path)
    summed = df.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_b = df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    summed_s = df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).sum(numeric_only=True)
    newDf['DollarVolumeTraded'] = summed.iloc[:,2]
    cats = ['b','s']
    for i in cats:
        if i=='b':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_b.iloc[:,2]
        elif i=='s':
            newDf['DollarVolumeTraded_<'+i+'>'] = summed_s.iloc[:,2]
    newDf['ShareVolumeTraded'] = summed.iloc[:,1]
    for i in cats:
        if i=='b':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_b.iloc[:,1]
        elif i=='s':
            newDf['ShareVolumeTraded_<'+i+'>'] = summed_s.iloc[:,1]
    newDf['VWAP'] = newDf.DollarVolumeTraded / newDf.ShareVolumeTraded
    for i in cats:
        if i=='b':
            newDf['VWAP_<'+i+'>'] = newDf['DollarVolumeTraded_<b>'] / newDf['ShareVolumeTraded_<b>']
        elif i=='s':
            newDf['VWAP_<'+i+'>'] = newDf['DollarVolumeTraded_<s>'] / newDf['ShareVolumeTraded_<s>']  
    newDf['LogReturn'] = np.log(newDf.VWAP) - np.log(newDf.VWAP.shift(1)) 
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf    

def tradeCount(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_tradeCount.csv'):            
    df, newDf = populate(input_file_path)
    newDf['TradeCount'] = df.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).size()
    df_b = df.drop(df[df.Qualifiers[0] != 'B'].index)
    df_s = df.drop(df[df.Qualifiers[0] != 'S'].index)
    cats = ['b','s']
    for i in cats:
        if i=='b':
            newDf['TradeCount_<'+i+'>'] = df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).size()
        elif i=='s':
            newDf['TradeCount_<'+i+'>'] = df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).size()
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf

def effectiveSpread(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_effectiveSpread.csv'):
    df, newDf = populate(input_file_path)
    cats = ['b','s']
    exchanges = df['#RIC'].split('.')[1].unique().tolist() 
    for m in exchanges:
        df_m = df.drop(df[df['#RIC'].split('.')[1] != m].index)
        df_m = df_m[['Bid Price','Ask Price']]
        averaged_m = df_m.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).mean(numeric_only=True)
        newDf['QuotedSpread_<'+m+'>'] = averaged_m.iloc[:,1] - averaged_m.iloc[:,0]
        newDf['PercentageSpread_<'+m+'>'] = (averaged_m.iloc[:,1] - averaged_m.iloc[:,0])/((averaged_m.iloc[:,1] + averaged_m.iloc[:,0])/2)
        df_v = df.drop(df[df['#RIC'].split('.')[1] != m].index)
        df_v = df_v[['Bid Size','Ask Size']]
        max_v = df_v.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).max(numeric_only=True)
        newDf['QuotedDollarDepth_<'+m+'>'] = ((max_v.iloc[:,1] * averaged_m.iloc[:,0]) + (averaged_m.iloc[:,1] * max_v.iloc[:,0]))/2
        newDf['QuotedShareDepth_<'+m+'>'] = (max_v.iloc[:,1] + max_v.iloc[:,0])/2
        for i in cats:
            if i=='b':
                newDf['EffectiveSpread_<k,'+m+','+i+',b>'] = np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).iloc[:,-1]) - np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).mean(numeric_only=True).iloc[:,0])
            elif i=='s':
                newDf['EffectiveSpread_<k,'+m+','+i+',b>'] = np.log(df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).iloc[:,-1]) - np.log(df_s.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).mean(numeric_only=True).iloc[:,0])
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf
    
def realisedSpread(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_realisedSpread.csv'):
    df, newDf = populate(input_file_path)
    cats = ['b','s']
    exchanges = df['#RIC'].split('.')[1].unique().tolist() 
    for m in exchanges:
        df_m = df.drop(df[df['#RIC'].split('.')[1] != m].index)
        df_m = df_m[['Bid Price','Ask Price']]
        averaged_m = df_m.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).mean(numeric_only=True)
        newDf['QuotedSpread_<'+m+'>'] = averaged_m.iloc[:,1] - averaged_m.iloc[:,0]
        newDf['PercentageSpread_<'+m+'>'] = (averaged_m.iloc[:,1] - averaged_m.iloc[:,0])/((averaged_m.iloc[:,1] + averaged_m.iloc[:,0])/2)
        df_v = df.drop(df[df['#RIC'].split('.')[1] != m].index)
        df_v = df_v[['Bid Size','Ask Size']]
        max_v = df_v.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).max(numeric_only=True)
        newDf['QuotedDollarDepth_<'+m+'>'] = ((max_v.iloc[:,1] * averaged_m.iloc[:,0]) + (averaged_m.iloc[:,1] * max_v.iloc[:,0]))/2
        newDf['QuotedShareDepth_<'+m+'>'] = (max_v.iloc[:,1] + max_v.iloc[:,0])/2
        for i in cats:
            if i=='b':
                newDf['RealisedSpread_<k,'+m+','+i+',b>'] = 2 * (np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).iloc[:,-1]) - np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period+period)+'T'))).mean(numeric_only=True).iloc[:,0]))
            elif i=='s':
                newDf['RealisedSpread_<k,'+m+','+i+',b>'] = 2 * (np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period+period)+'T'))).mean(numeric_only=True).iloc[:,0]) - np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).iloc[:,-1]))
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf
    
def priceImpact(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_priceImpact.csv'):
    df, newDf = populate(input_file_path)
    cats = ['b','s']
    exchanges = df['#RIC'].split('.')[1].unique().tolist() 
    for m in exchanges:
        df_m = df.drop(df[df['#RIC'].split('.')[1] != m].index)
        df_m = df_m[['Bid Price','Ask Price']]
        averaged_m = df_m.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).mean(numeric_only=True)
        newDf['QuotedSpread_<'+m+'>'] = averaged_m.iloc[:,1] - averaged_m.iloc[:,0]
        newDf['PercentageSpread_<'+m+'>'] = (averaged_m.iloc[:,1] - averaged_m.iloc[:,0])/((averaged_m.iloc[:,1] + averaged_m.iloc[:,0])/2)
        df_v = df.drop(df[df['#RIC'].split('.')[1] != m].index)
        df_v = df_v[['Bid Size','Ask Size']]
        max_v = df_v.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).max(numeric_only=True)
        newDf['QuotedDollarDepth_<'+m+'>'] = ((max_v.iloc[:,1] * averaged_m.iloc[:,0]) + (averaged_m.iloc[:,1] * max_v.iloc[:,0]))/2
        newDf['QuotedShareDepth_<'+m+'>'] = (max_v.iloc[:,1] + max_v.iloc[:,0])/2
        for i in cats:
            if i=='b':
                newDf['PriceImpact_<k,'+m+','+i+',b>'] = 2 * (np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period+period)+'T'))).iloc[:,0]) - np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).mean(numeric_only=True).iloc[:,0]))
            elif i=='s':
                newDf['PriceImpact_<k,'+m+','+i+',b>'] = 2 * (np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).iloc[:,0]) - np.log(df_b.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period+period)+'T'))).mean(numeric_only=True).iloc[:,0]))
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf

def averagePrice(input_file_path:str='data/input.csv', output_file_path:str= 'data/data_averagePrice.csv'):    
    df, newDf = populate(input_file_path)
    averaged = df.groupby(pd.Grouper(key='Date-Time', axis=0, freq=(str(period)+'T'))).mean(numeric_only=True)
    newDf['AveragePrice'] = averaged.iloc[:,0]
    newDf = newDf.fillna('Not Applicable')
    newDf.to_csv(output_file_path)
    return newDf
    
def main():
    dollarVolumeTraded('BHPAX')
    shareVolumeTraded('BHPAX')
    vWAP('BHPAX')
    arithmeticReturn('BHPAX')
    logReturn('BHPAX')
    tradeCount('BHPAX')
    effectiveSpread('BHPAX')
    realisedSpread('BHPAX')
    priceImpact('BHPAX')
    averagePrice('BHPAX')

if __name__ == "__main__":
    main()