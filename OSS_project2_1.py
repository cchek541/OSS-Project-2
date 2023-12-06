import pandas as pd
from pandas import Series, DataFrame

df = pd.read_csv(r"C:\Users\cchek\OneDrive\Desktop\subject\2-2\Intro to open source sw\oss project 2\2019_kbo_for_kaggle_v2.csv")

for year in range(2015,2019):
    print(df[df['year'] == year].nlargest(10, 'H')[['batter_name', 'H']])
    print(df[df['year'] == year].nlargest(10, 'avg')[['batter_name', 'avg']])
    print(df[df['year'] == year].nlargest(10, 'HR')[['batter_name', 'HR']])
    print(df[df['year'] == year].nlargest(10, 'OBP')[['batter_name', 'OBP']])

df_2018 = df[df['year'] == 2018]

positions = ['포수','1루수','2루수','3루수','유격수','좌익수','중견수','우익수','지명타자']
highest_wars = {}

for position in positions:
    position_data = df_2018[df_2018['cp'] == position]

    highest_war = position_data.loc[position_data['war'].astype(float).idxmax()]

    highest_wars[position] = highest_war[['batter_name','war']]

print("")
for position, player_info in highest_wars.items():
    print(f"{position}: {player_info['batter_name']} (war: {player_info['war']})")

correlations = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()['salary']

max_corr_statistic = correlations.iloc[:-1].idxmax()

print("")
print(f"highest correlation is : {max_corr_statistic}")