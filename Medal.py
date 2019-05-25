import pandas as pd

df=pd.read_csv('athlete_events.csv', encoding='utf-8')


df=df.dropna(axis='rows')


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

# summer, winter 분류
df_summer=df[df['Season'].isin(['Summer'])]
df_winter=df[df['Season'].isin(['Winter'])]

counts=pd.DataFrame(df['NOC'].value_counts())

gold=0
silver=0
bronze=0

df_score=0
# index가 NOC
df_score=pd.DataFrame(columns=('Year','Score'))

value_list = df_summer['NOC'].unique()
value_list_2=df_summer['Year'].unique()

for i in value_list:
    # 나라별로 분류한 DataFrame
    noc_df = df.loc[df['NOC'] == i]
    # 나라에서 연도로 분류한 DataFrame
    for j in value_list_2:
        noc_df_2=noc_df.loc[df['Year']==j]
        a=pd.DataFrame(noc_df_2['Medal'].value_counts())
        if a.index.contains('Gold'):
            gold=a.loc['Gold','Medal']
        if a.index.contains('Silver'):
            silver=a.loc['Silver','Medal']
        if a.index.contains('Bronze'):
            bronze=a.loc['Bronze','Medal']
        score=gold*100+silver+bronze*0.01
        df_score.loc[i]=[j,score]
print(df_score)


        
