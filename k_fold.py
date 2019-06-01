import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import statistics as st
from collections import Counter

# read file
athlete_df = pd.read_csv("athlete_events.csv")

# ===========================SCORE============================

# index가 NOC
df_score = pd.DataFrame(columns=('NOC', 'Score'))

# df=df.dropna(axis='rows')

# 분석에 필요한 값만 남김
athlete_df = athlete_df.drop('Sex', axis=1)
athlete_df = athlete_df.drop('Age', axis=1)
athlete_df = athlete_df.drop('Height', axis=1)
athlete_df = athlete_df.drop('Weight', axis=1)
athlete_df = athlete_df.drop('Team', axis=1)
athlete_df = athlete_df.drop('City', axis=1)

athlete_df = athlete_df.drop('Games', axis=1)
athlete_df = athlete_df.drop('ID', axis=1)
athlete_df = athlete_df.drop('Name', axis=1)

# 분석에 필요한 값만 남겼으므로 중복이 있으면 같은 경기, 같은 선수단 인 것(팀전) -> 삭제
athlete_df.drop_duplicates()

gdp_df = athlete_df.drop('Sport', axis=1)
gdp_df = gdp_df.drop('Event', axis=1)
gdp_df = gdp_df.drop('Medal', axis=1)
gdp_df = gdp_df.drop('Area', axis=1)

gold = 0
silver = 0
bronze = 0

# 시즌 별로 (summer, winter) 분류
df_summer = athlete_df[athlete_df['Season'].isin(['Summer'])]
df_winter = athlete_df[athlete_df['Season'].isin(['Winter'])]

value_list = df_summer['NOC'].unique()

gold = 0
silver = 0
bronze = 0

# 연도 별로 분류
df_summer_2000 = df_summer[df_summer['Year'].isin(['2000'])]
df_summer_2004 = df_summer[df_summer['Year'].isin(['2004'])]
df_summer_2008 = df_summer[df_summer['Year'].isin(['2008'])]
df_summer_2012 = df_summer[df_summer['Year'].isin(['2012'])]
df_summer_2016 = df_summer[df_summer['Year'].isin(['2016'])]

df_winter_2002 = df_winter[df_winter['Year'].isin(['2002'])]
df_winter_2006 = df_winter[df_winter['Year'].isin(['2006'])]
df_winter_2010 = df_winter[df_winter['Year'].isin(['2010'])]
df_winter_2014 = df_winter[df_winter['Year'].isin(['2014'])]


# 해당 년도의 각 나라의 score 계산하는 함수
def calScores(df):
    for i in value_list:
        # 나라별로 분류한 DataFrame
        global df_score
        global gold, silver, bronze
        noc_df = df.loc[df['NOC'] == i]
        a = pd.DataFrame(noc_df['Medal'].value_counts())
        if a.index.contains('Gold'):
            gold = a.loc['Gold', 'Medal']
        if a.index.contains('Silver'):
            silver = a.loc['Silver', 'Medal']
        if a.index.contains('Bronze'):
            bronze = a.loc['Bronze', 'Medal']
        score = gold*100+silver+bronze*0.01
        df_score.loc[i] = [i, score]
        df_score = df_score.sort_values(by=['Score'], ascending=False)


calScores(df_summer_2000)
df_summer_score_2000 = df_score
athlete_df_2000 = athlete_df.loc[athlete_df['Year'] == 2000]
df_2000 = pd.merge(df_summer_score_2000, athlete_df_2000[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2000 = df_2000.drop_duplicates(subset='NOC', keep='first')

calScores(df_summer_2004)
df_summer_score_2004 = df_score
athlete_df_2004 = athlete_df.loc[athlete_df['Year'] == 2004]
df_2004 = pd.merge(df_summer_score_2004, athlete_df_2004[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2004 = df_2004.drop_duplicates(subset='NOC', keep='first')

calScores(df_summer_2008)
df_summer_score_2008 = df_score
athlete_df_2008 = athlete_df.loc[athlete_df['Year'] == 2008]
df_2008 = pd.merge(df_summer_score_2008, athlete_df_2008[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2008 = df_2008.drop_duplicates(subset='NOC', keep='first')

calScores(df_summer_2012)
df_summer_score_2012 = df_score
athlete_df_2012 = athlete_df.loc[athlete_df['Year'] == 2012]
df_2012 = pd.merge(df_summer_score_2012, athlete_df_2012[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2012 = df_2008.drop_duplicates(subset='NOC', keep='first')

calScores(df_summer_2016)
df_summer_score_2016 = df_score
athlete_df_2016 = athlete_df.loc[athlete_df['Year'] == 2016]
df_2016 = pd.merge(df_summer_score_2016, athlete_df_2016[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2016 = df_2016.drop_duplicates(subset='NOC', keep='first')

calScores(df_winter_2002)
df_winter_score_2002 = df_score
athlete_df_2002 = athlete_df.loc[athlete_df['Year'] == 2002]
df_2002 = pd.merge(df_winter_score_2002, athlete_df_2002[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2002 = df_2002.drop_duplicates(subset='NOC', keep='first')

calScores(df_winter_2006)
df_winter_score_2006 = df_score
athlete_df_2006 = athlete_df.loc[athlete_df['Year'] == 2006]
df_2006 = pd.merge(df_winter_score_2006, athlete_df_2006[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2006 = df_2006.drop_duplicates(subset='NOC', keep='first')

calScores(df_winter_2010)
df_winter_score_2010 = df_score
athlete_df_2010 = athlete_df.loc[athlete_df['Year'] == 2010]
df_2010 = pd.merge(df_winter_score_2010, athlete_df_2010[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2010 = df_2010.drop_duplicates(subset='NOC', keep='first')

calScores(df_winter_2014)
df_winter_score_2014 = df_score
athlete_df_2014 = athlete_df.loc[athlete_df['Year'] == 2014]
df_2014 = pd.merge(df_winter_score_2014, athlete_df_2014[['NOC', 'GDP', 'Area', 'Year']], on='NOC', how='outer')
df_2014 = df_2014.drop_duplicates(subset='NOC', keep='first')

df_2000 = df_2000[df_2000['GDP'].notnull()]
df_2004 = df_2004[df_2004['GDP'].notnull()]
df_2008 = df_2008[df_2008['GDP'].notnull()]
df_2012 = df_2012[df_2012['GDP'].notnull()]
df_2016 = df_2016[df_2016['GDP'].notnull()]

df_2002 = df_2002[df_2002['GDP'].notnull()]
df_2006 = df_2006[df_2006['GDP'].notnull()]
df_2010 = df_2010[df_2010['GDP'].notnull()]
df_2014 = df_2014[df_2014['GDP'].notnull()]


frames_summer = [df_2000, df_2004, df_2008, df_2012, df_2016]
frames_winter = [df_2002, df_2006, df_2010, df_2014]

summer_df = pd.concat(frames_summer)
winter_df = pd.concat(frames_winter)

summer_k_fold = KFold(n_splits=5, shuffle=False)

pre_summer_score_set = []
groupTest=[]
groupSummer=[]

i=0

pred_scoreAll=[]
for train_index, test_index in summer_k_fold.split(summer_df):
    train_df = summer_df.iloc[train_index, :]
    test_df = summer_df.iloc[test_index, :]

    GDP = train_df['GDP']
    score = train_df['Score']

    reg = linear_model.LinearRegression()
    reg.fit(GDP[:, np.newaxis], score)
    px = np.array([GDP.min(), GDP.max()])
    py = reg.predict(px[:, np.newaxis])

    plt.scatter(GDP, score)
    plt.plot(px, py, color="r")
    plt.xlabel("GDP")
    plt.ylabel("score")
#    plt.show()
#    print(test_df)
    test_df_GDP=test_df['GDP'][:, np.newaxis]
    pred_score=reg.predict(test_df_GDP)
    test_score=((test_df['Score'].values-(test_df['Score'].values % 1300))/1300)
    pred_score=((pred_score-(pred_score % 1300))/1300)
    pred_scoreAll.append(pred_score)
    i+=1
#    print(test_score)
#    print(pred_score)
#    print(accuracy_score(test_score, pred_score))

print(pred_scoreAll)
voted_scoreGrade=[]
for i in range(len(pred_scoreAll[0])-1):
    pred_i=[pred_scoreAll[0][i], pred_scoreAll[1][i], pred_scoreAll[2][i], pred_scoreAll[3][i], pred_scoreAll[4][i]]
#    print(pred_i)
    cnt= Counter(pred_i)
#    print(cnt)
    cnt_mode=cnt.most_common(1)
#    print(cnt_mode)
#    print(cnt_mode[0][0])
    voted_scoreGrade.append(cnt_mode[0][0])

print(accuracy_score(test_score, voted_scoreGrade))

pred_scoreAll=[]
for train_index, test_index in summer_k_fold.split(winter_df):
    train_df = winter_df.iloc[train_index, :]
    test_df = winter_df.iloc[test_index, :]

    GDP = train_df['GDP']
    score = train_df['Score']

    reg = linear_model.LinearRegression()
    reg.fit(GDP[:, np.newaxis], score)
    px = np.array([GDP.min(), GDP.max()])
    py = reg.predict(px[:, np.newaxis])

    plt.scatter(GDP, score)
    plt.plot(px, py, color="r")
    plt.xlabel("GDP")
    plt.ylabel("score")
#    plt.show()
#    print(test_df)
    test_df_GDP=test_df['GDP'][:, np.newaxis]
    pred_score=reg.predict(test_df_GDP)
    test_score=((test_df['Score'].values-(test_df['Score'].values % 700))/700)
    pred_score=((pred_score-(pred_score % 700))/700)
    pred_scoreAll.append(pred_score)
    i+=1
#    print(test_score)
#    print(pred_score)
#    print(accuracy_score(test_score, pred_score))

print(pred_scoreAll)
voted_scoreGrade=[]
for i in range(len(pred_scoreAll[0])-1):
    pred_i=[pred_scoreAll[0][i], pred_scoreAll[1][i], pred_scoreAll[2][i], pred_scoreAll[3][i], pred_scoreAll[4][i]]
#    print(pred_i)
    cnt= Counter(pred_i)
#    print(cnt)
    cnt_mode=cnt.most_common(1)
#    print(cnt_mode)
#    print(cnt_mode[0][0])
    voted_scoreGrade.append(cnt_mode[0][0])

print(accuracy_score(test_score, voted_scoreGrade))
