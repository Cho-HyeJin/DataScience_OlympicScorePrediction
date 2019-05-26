import pandas as pd

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

calScores(df_summer_2000)
df_score_2000Summer=df_score
print(df_score_2000Summer)

calScores(df_summer_2004)
df_score_2004Summer=df_score
print(df_score_2004Summer)

calScores(df_summer_2008)
df_score_2008Summer=df_score
print(df_score_2008Summer)

calScores(df_summer_2012)
df_score_2012Summer=df_score
print(df_score_2012Summer)

calScores(df_summer_2016)
df_score_2016Summer=df_score
print(df_score_2016Summer)

calScores(df_winter_2002)
df_score_2002Winter=df_score
print(df_score_2002Winter)

calScores(df_winter_2006)
df_score_2006Winter=df_score
print(df_score_2006Winter)

calScores(df_winter_2010)
df_score_2010Winter=df_score
print(df_score_2010Winter)

calScores(df_winter_2014)
df_score_2014Winter=df_score
print(df_score_2014Winter)
