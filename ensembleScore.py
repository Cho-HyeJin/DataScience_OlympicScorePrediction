import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

df=pd.read_csv('athlete_events.csv', encoding='utf-8')

df_score=0
# index가 NOC
df_score=pd.DataFrame(columns=('NOC','Score'))

# df=df.dropna(axis='rows')

# 분석에 필요한 값만 남김
df=df.drop('Sex', axis=1)
df=df.drop('Age', axis=1)
df=df.drop('Height', axis=1)
df=df.drop('Weight', axis=1)
df=df.drop('Team', axis=1)
df=df.drop('City', axis=1)

df=df.drop('Games', axis=1)
df=df.drop('ID', axis=1)
df=df.drop('Name', axis=1)

# 분석에 필요한 값만 남겼으므로 중복이 있으면 같은 경기, 같은 선수단 인 것(팀전) -> 삭제
df.drop_duplicates()

gold=0
silver=0
bronze=0

# 시즌 별로 (summer, winter) 분류
df_summer=df[df['Season'].isin(['Summer'])]
df_winter=df[df['Season'].isin(['Winter'])]

value_list = df_summer['NOC'].unique()
value_list_2=df_summer['Year'].unique()

gold=0
silver=0
bronze=0

# 연도 별로 분류
df_summer_2000=df_summer[df_summer['Year'].isin(['2000'])]
df_summer_2004=df_summer[df_summer['Year'].isin(['2004'])]
df_summer_2008=df_summer[df_summer['Year'].isin(['2008'])]
df_summer_2012=df_summer[df_summer['Year'].isin(['2012'])]
df_summer_2016=df_summer[df_summer['Year'].isin(['2016'])]

df_winter_2002=df_winter[df_winter['Year'].isin(['2002'])]
df_winter_2006=df_winter[df_winter['Year'].isin(['2006'])]
df_winter_2010=df_winter[df_winter['Year'].isin(['2010'])]
df_winter_2014=df_winter[df_winter['Year'].isin(['2014'])]

# 해당 년도의 각 나라의 score 계산하는 함수
def calScores(df):
    for i in value_list:
        # 나라별로 분류한 DataFrame
        global df_score
        global gold, silver, bronze
        noc_df = df.loc[df['NOC'] == i]
        a=pd.DataFrame(noc_df['Medal'].value_counts())
        if a.index.contains('Gold'):
            gold=a.loc['Gold','Medal']
        if a.index.contains('Silver'):
            silver=a.loc['Silver','Medal']
        if a.index.contains('Bronze'):
            bronze=a.loc['Bronze','Medal']
        score=gold*100+silver+bronze*0.01
        df_score.loc[i]=[i, score]
        df_score=df_score.sort_values(by=['Score'], ascending=False)

def changeRank(df):
    count=1
    for i in range(len(df)):
        df.iloc[i,1]=int(count)
        count+=1
    return df


calScores(df_summer_2000)
df_score_2000Summer=df_score
df_score_2000SummerRank=changeRank(df_score_2000Summer)

calScores(df_summer_2004)
df_score_2004Summer=df_score
df_score_2004SummerRank=changeRank(df_score_2004Summer)

calScores(df_summer_2008)
df_score_2008Summer=df_score
df_score_2008SummerRank=changeRank(df_score_2008Summer)

calScores(df_summer_2012)
df_score_2012Summer=df_score
df_score_2012SummerRank=changeRank(df_score_2012Summer)

calScores(df_summer_2016)
df_score_2016Summer=df_score
df_score_2016SummerRank=changeRank(df_score_2016Summer)

result1= pd.merge(df_score_2000SummerRank,df_score_2004SummerRank, on='NOC')
result1= pd.merge(result1,df_score_2008SummerRank, on='NOC')
result1= pd.merge(result1,df_score_2012SummerRank, on='NOC')
result1= pd.merge(result1,df_score_2016SummerRank, on='NOC')
result1.columns=["NOC","2000","2004","2008","2012","2016"]
result1['2020']=0
for i in range(len(result1)):
    result1.iloc[i,6]=int((result1.iloc[i,1]+result1.iloc[i,2]+result1.iloc[i,3]+result1.iloc[i,4]+result1.iloc[i,5])/5)
result1.sort_values(by='2020',inplace=True)
result1['2020']=result1['2020'].rank()
print(result1)


#Winter(2002,2006,2010,2014)
calScores(df_winter_2002)
df_score_2002Winter=df_score
df_score_2002WinterRank=changeRank(df_score_2002Winter)
#print(df_score_2002Winter.head())

calScores(df_winter_2006)
df_score_2006Winter=df_score
df_score_2006WinterRank=changeRank(df_score_2006Winter)
#print(df_score_2006Winter.head())

calScores(df_winter_2010)
df_score_2010Winter=df_score
df_score_2010WinterRank=changeRank(df_score_2010Winter)
#print(df_score_2010Winter.head())

calScores(df_winter_2014)
df_score_2014Winter=df_score
df_score_2014WinterRank=changeRank(df_score_2014Winter)
#print(df_score_2014Winter)

result2= pd.merge(df_score_2002WinterRank,df_score_2006WinterRank, on='NOC')
result2= pd.merge(result2,df_score_2010WinterRank, on='NOC')
result2= pd.merge(result2,df_score_2014WinterRank, on='NOC')
result2['2018']=0
for i in range(len(result2)):
    result2.iloc[i,5]=int((result2.iloc[i,1]+result2.iloc[i,2]+result2.iloc[i,3]+result2.iloc[i,4])/4)
result2.columns=["NOC","2002","2006","2010","2014","2018"]
result2.sort_values(by='2018',inplace=True)
result2['2018']=result2['2018'].rank()
print(result2)

#--------Accuracy
df_2018=pd.read_csv('2018.csv', header=0)
score = accuracy_score(result2.iloc[:10,5:6].astype(int),df_2018.iloc[:,1])
print(result2.iloc[:10,5:6],df_2018.iloc[:,1])
print('\n\nAccuracy is', score)





# --------ensemble-------- 4개 값 모두 있는 것만 df_summer_ensemble=pd.DataFrame(columns=('NOC','Score'))

df_score_2000Summer.rename(columns = {"Score":"Score2000"}, inplace=True)

df_score_2004Summer.rename(columns = {"Score":"Score2004"}, inplace=True)
#print(df_score_2004Summer)

df_score_2008Summer.rename(columns = {"Score":"Score2008"}, inplace=True)
#print(df_score_2008Summer)

df_score_2012Summer.rename(columns = {"Score":"Score2012"}, inplace=True)
#print(df_score_2012Summer)

df_score_2016Summer.rename(columns = {"Score":"Score2016"}, inplace=True)

'''
df_score_summerAll=pd.merge(df_score_2000Summer, df_score_2004Summer)
df_score_summerAll=pd.merge(df_score_summerAll, df_score_2008Summer)
df_score_summerAll=pd.merge(df_score_summerAll, df_score_2012Summer)
df_score_summerAll=pd.merge(df_score_summerAll, df_score_2016Summer)
df_score_summerAll['Avg']=(df_score_summerAll['Score2000']+df_score_summerAll['Score2004']+df_score_summerAll['Score2008']+df_score_summerAll['Score2012']+df_score_summerAll['Score2016'])/5
print(df_score_summerAll)
'''

df_score_2002Winter.rename(columns = {"Score":"Score2002"}, inplace=True)

df_score_2006Winter.rename(columns = {"Score":"Score2006"}, inplace=True)
#print(df_score_2004Summer)

df_score_2010Winter.rename(columns = {"Score":"Score2010"}, inplace=True)
#print(df_score_2008Summer)

df_score_2014Winter.rename(columns = {"Score":"Score2014"}, inplace=True)
#print(df_score_2012Summer)

'''
df_score_winterAll=pd.merge(df_score_2002Winter, df_score_2006Winter)
df_score_winterAll=pd.merge(df_score_winterAll, df_score_2010Winter)
df_score_winterAll=pd.merge(df_score_winterAll, df_score_2014Winter)
df_score_winterAll['Avg']=(df_score_winterAll['Score2002']+df_score_winterAll['Score2006']+df_score_winterAll['Score2010']+df_score_winterAll['Score2014'])/4
print(df_score_winterAll)
'''
