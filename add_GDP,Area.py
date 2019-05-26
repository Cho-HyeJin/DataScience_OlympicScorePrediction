import numpy as np
import pandas as pd

# read file
athlete_df = pd.read_csv("athlete_events.csv")
noc_df = pd.read_csv("noc_regions.csv")
country_df = pd.read_csv("countries of the world.csv")
gdp_df = pd.read_excel("w_gdp.xls")

# array for contain GDP, this array is use for add column to dataframe
gdp_list = np.array([])

# Find GDP by using NOC and Year
for i in range(len(athlete_df)):
    noc_index = gdp_df[gdp_df['Country Code'] == athlete_df.iloc[i, 7]].index.values.astype(int)
    year = str(athlete_df.iloc[i, 9])
    year_index = gdp_df.columns.get_loc(year)
    # append to gdp_list if value is nan add nan
    if len(gdp_df.iloc[noc_index, year_index].values) == 0:
        gdp_list = np.append(gdp_list, np.nan)
    else:
        gdp_list = np.append(gdp_list, gdp_df.iloc[noc_index, year_index].values)

# add column 'GDP' to athlete_df file
athlete_df['GDP'] = gdp_list

# array for contain NOC, this array is use for add column to dataframe
# array for contain Area, this array is use for add column to dataframe
noc=np.array([])
area=np.array([])

# Find NOC by using Country and region
for i in range(len(country_df)):
    noc_index=noc_df[noc_df['region']==country_df.iloc[i,0]].index.values.astype(int)

    if len(noc_index)==0:
        noc=np.append(noc,np.nan)
    else:
        noc=np.append(noc,noc_df.iloc[noc_index,0].values)
        
# add column 'NOC' to country_df file
country_df['NOC']=noc

# Find Area by using NOC
for i in range(len(athlete_df)):
    area_index=country_df[country_df['NOC']==athlete_df.iloc[i,7]].index.values.astype(int)

    
    if len(area_index)==0:
        area=np.append(area,np.nan)
    else:
        area=np.append(area,country_df.iloc[area_index,2].values)

# add column 'Area' to athlete_df file
athlete_df['Area']=area



print(athlete_df)

