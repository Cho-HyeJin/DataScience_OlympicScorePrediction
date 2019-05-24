import numpy as np
import pandas as pd

# read data
athlete_df = pd.read_csv("athlete_events.csv")
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

# add column 'GDP'
athlete_df['GDP'] = gdp_list

print(athlete_df)
