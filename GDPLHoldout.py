import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
from sklearn.metrics import confusion_matrix
from sklearn import tree

# read file
athlete_df = pd.read_csv("athlete_events.csv")


#===========================SCORE============================

df_score=0
# index가 NOC
df_score=pd.DataFrame(columns=('NOC','Score'))

# df=df.dropna(axis='rows')

# 분석에 필요한 값만 남김
athlete_df=athlete_df.drop('Sex', axis=1)
athlete_df=athlete_df.drop('Age', axis=1)
athlete_df=athlete_df.drop('Height', axis=1)
athlete_df=athlete_df.drop('Weight', axis=1)
athlete_df=athlete_df.drop('Team', axis=1)
athlete_df=athlete_df.drop('City', axis=1)

athlete_df=athlete_df.drop('Games', axis=1)
athlete_df=athlete_df.drop('ID', axis=1)
athlete_df=athlete_df.drop('Name', axis=1)

# 분석에 필요한 값만 남겼으므로 중복이 있으면 같은 경기, 같은 선수단 인 것(팀전) -> 삭제
athlete_df.drop_duplicates()

gdp_df=athlete_df.drop('Sport', axis=1)
gdp_df=gdp_df.drop('Event', axis=1)
gdp_df=gdp_df.drop('Medal', axis=1)
gdp_df=gdp_df.drop('Area', axis=1)

gold=0
silver=0
bronze=0

# 시즌 별로 (summer, winter) 분류
df_summer=athlete_df[athlete_df['Season'].isin(['Summer'])]
df_winter=athlete_df[athlete_df['Season'].isin(['Winter'])]

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

# linear regreission
def linRegGDP(df):
    df= df.dropna()
#    print(df)
    X=df['GDP'].values
    y=df['Score'].values  
    X=X.reshape(len(X),1)
    y=y.reshape(len(y),1)
    regr=linear_model.LinearRegression()
    regr.fit(X,y)
    plt.plot(X, regr.predict(X), color='blue', linewidth=3)
    plt.scatter(X,y)
    plt.show()

# hold out
def holdOut(df):
    # split training dataset and testing dataset
    df= df.dropna()
    X=df['GDP']
    y=df['Score']
    X=X[:,np.newaxis]
    y=y[:,np.newaxis]
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1)
    print(X_train, X_test, y_train, y_test)
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    y_pred=regr.predict(X_test)
    y_test=y_test[:,-1].astype(int)
    y_pred= y_pred[:,-1].astype(int)
    groupTest=[]
    groupPred=[]
    for i in range(len(y_test)):
        groupTest.append((y_test[i]-(y_test[i]%1300))/1300)
    for i in range(len(y_pred)):
        groupPred.append((y_pred[i]-(y_pred[i]%1300))/1300)
    print(groupPred, groupTest)
    print("accuracy: ", accuracy_score(groupPred, groupTest))
    
#Summer : 2000~2016 -> 총 5개 ----------------------------------------

calScores(df_summer_2000)
df_score_2000Summer=df_score
gdp_df2000= gdp_df[gdp_df['Year'].isin(['2000'])]
#print(df_score_2000Summer)
df_score_2000Summer=pd.merge(df_score_2000Summer, gdp_df2000, on='NOC')
df_score_2000Summer=df_score_2000Summer.drop_duplicates()
#print(df_score_2000Summer)

calScores(df_summer_2004)
df_score_2004Summer=df_score
gdp_df2004= gdp_df[gdp_df['Year'].isin(['2004'])]
df_score_2004Summer=pd.merge(df_score_2004Summer, gdp_df2004, on='NOC')
df_score_2004Summer=df_score_2004Summer.drop_duplicates()
#print(df_score_2004Summer)

calScores(df_summer_2008)
df_score_2008Summer=df_score
gdp_df2008= gdp_df[gdp_df['Year'].isin(['2008'])]
df_score_2008Summer=pd.merge(df_score_2008Summer, gdp_df2008, on='NOC')
df_score_2008Summer=df_score_2008Summer.drop_duplicates()
#print(df_score_2008Summer)

calScores(df_summer_2012)
df_score_2012Summer=df_score
gdp_df2012= gdp_df[gdp_df['Year'].isin(['2012'])]
df_score_2012Summer=pd.merge(df_score_2012Summer, gdp_df2012, on='NOC')
df_score_2012Summer=df_score_2012Summer.drop_duplicates()
#print(df_score_2012Summer)

calScores(df_summer_2016)
df_score_2016Summer=df_score
gdp_df2016= gdp_df[gdp_df['Year'].isin(['2016'])]
df_score_2016Summer=pd.merge(df_score_2016Summer, gdp_df2016, on='NOC')
df_score_2016Summer=df_score_2016Summer.drop_duplicates()
#print(df_score_2016Summer)

#Winter : 2002~2014 -> 총 4개 ----------------------------------------------
calScores(df_winter_2002)
df_score_2002Winter=df_score
gdp_df2002= gdp_df[gdp_df['Year'].isin(['2002'])]
df_score_2002Winter=pd.merge(df_score_2002Winter, gdp_df2002, on='NOC')
df_score_2002Winter=df_score_2002Winter.drop_duplicates()
#print(df_score_2002Winter)

calScores(df_winter_2006)
df_score_2006Winter=df_score
gdp_df2006= gdp_df[gdp_df['Year'].isin(['2006'])]
df_score_2006Winter=pd.merge(df_score_2006Winter, gdp_df2006, on ='NOC')
df_score_2006Winter=df_score_2006Winter.drop_duplicates()
#print(df_score_2006Winter)

calScores(df_winter_2010)
df_score_2010Winter=df_score
gdp_df2010= gdp_df[gdp_df['Year'].isin(['2010'])]
df_score_2010Winter=pd.merge(df_score_2010Winter, gdp_df2010, on='NOC')
df_score_2010Winter=df_score_2010Winter.drop_duplicates()
#print(df_score_2010Winter)

calScores(df_winter_2014)
df_score_2014Winter=df_score
gdp_df2014= gdp_df[gdp_df['Year'].isin(['2014'])]
df_score_2014Winter=pd.merge(df_score_2014Winter, gdp_df2014, on='NOC')
df_score_2014Winter=df_score_2014Winter.drop_duplicates()
#print(df_score_2014Winter)

df_score_summer=pd.concat([df_score_2000Summer,df_score_2004Summer,df_score_2008Summer,df_score_2012Summer,df_score_2016Summer])
df_score_winter=pd.concat([df_score_2002Winter, df_score_2006Winter, df_score_2010Winter, df_score_2014Winter])
# GDP Regression
#linRegGDP(df_score_summer)
#linRegGDP(df_score_winter)

holdOut(df_score_summer)
