import pandas as pd

df=pd.read_csv('C:\\Users\parks\\Desktop\\athlete_events.csv', encoding='utf-8')

# unique(팀전일 때 하나만 남기기)


df=df.dropna(axis='rows')

counts=pd.DataFrame(df['NOC'].value_counts())

gold=0
silver=0
bronze=0

df_score=0
# index가 NOC
df_score=pd.DataFrame(columns=('Year','Score'))

value_list = df['NOC'].unique()
value_list_2=df['Year'].unique()

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
            gold=a.loc['Silver','Medal']
        if a.index.contains('Bronze'):
            gold=a.loc['Bronze','Medal']
        score=gold*100+silver+bronze*0.01
        df_score.loc[i]=[j,score]

print(df_score)
        
