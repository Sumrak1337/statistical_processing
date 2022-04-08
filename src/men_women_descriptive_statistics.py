import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt


def write_to_file(df, file, distrs):
    # TODO: change encoding
    nan_policy = 'omit'
    for ind, row in df.iterrows():
        file.write(f'{distrs[ind]} \n')
        file.write(f'min: {int(np.nanmin(row))}, ({df.columns[np.argmin(row)]}) \n')
        file.write(f'max: {int(np.nanmax(row))}, ({df.columns[np.argmax(row)]}) \n')
        file.write(f'mean: {np.nanmean(row)} \n')
        file.write(f'median: {np.nanmedian(row)} \n')
        file.write(f'sd: {np.nanstd(row, ddof=1)} \n')
        file.write(f'interquartile range: {stat.iqr(row, nan_policy=nan_policy)} \n')
        file.write(f'range: {int(np.nanmax(row) - np.nanmin(row))} \n')
        file.write(f'skewness: {stat.skew(row, nan_policy=nan_policy)} \n')
        file.write(f'kurtosis: {stat.kurtosis(row, nan_policy=nan_policy)} \n')
        file.write('\n')
    file.close()


def plotting(df, title, distrs):
    plt.figure(figsize=(40, 30))
    for ind, column in enumerate(df.columns):
        plt.subplot(3, 3, ind + 1)
        plt.title(f'{title} {column}')
        plt.pie(df[column][1:].dropna(), labels=distrs[~np.isnan(df[column].to_numpy())][1:], autopct='%1.3f%%')
        plt.legend()
    plt.savefig(Path(results_path, f'{title}_subplots.png'))


"""
Центральный
Северо-западный
Южный
Южный
Северо-кавказский
Приволжский
Уральский
Сибирский
Сибирский
Дальневосточный
Дальневосточный
Крымский
"""

rep_path = pathlib.Path.cwd().resolve().parents[0]
data_path = Path(rep_path, 'data')
results_path = Path(rep_path, 'results')
clear_data_path = Path(rep_path, 'clear_data')

tmp_df_men = pd.read_excel(Path(data_path, 'men_2015_2022.xlsx'))
tmp_df_women = pd.read_excel(Path(data_path, 'women_2015_2022.xlsx'))

df_men_agged = pd.DataFrame()
df_women_agged = pd.DataFrame()

columns_names = ['age', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
for i, r in tmp_df_men.iterrows():
    if i < 2:
        continue
    if (i - 2) % 20 != 0:
        df_men_agged = df_men_agged.append(r, ignore_index=True).astype(int, errors='ignore')

for i, r in tmp_df_women.iterrows():
    if i < 2:
        continue
    if (i - 2) % 20 != 0:
        df_women_agged = df_women_agged.append(r, ignore_index=True)

df_men_agged.columns = columns_names
df_women_agged.columns = columns_names

# TODO: write to 'clear data' (.xls)

del df_men_agged['age']
del df_women_agged['age']

df_men = pd.DataFrame()
df_women = pd.DataFrame()
for i in range(13):
    sum_row_men = 0
    sum_row_women = 0
    for j in range(19):
        sum_row_men += df_men_agged.loc[j + 19 * i]
        sum_row_women += df_women_agged.loc[j + 19 * i]
    df_men = df_men.append(sum_row_men, ignore_index=True)
    df_women = df_women.append(sum_row_women, ignore_index=True)

df_sum = df_men + df_women
file_men = open(Path(results_path, 'describing_men.txt'), 'w')
file_women = open(Path(results_path, 'describing_women.txt'), 'w')
districts = np.array(['Российская Федерация',
                      'Центральный ФО',
                      'Северо-Западный ФО',
                      'Южный ФО (до 2016)',
                      'Южный ФО (с 2017)',
                      'Северо-Кавказский ФО',
                      'Приволжский ФО',
                      'Уральский ФО',
                      'Сибирский ФО (до 2018)',
                      'Сибирский ФО (с 2019)',
                      'Дальневосточный ФО (до 2018)',
                      'Дальневосточный ФО (с 2019)',
                      'Крымский ФО'])

write_to_file(df_men, file_men, districts)
write_to_file(df_women, file_women, districts)

plotting(df_men, 'Men', districts)
plotting(df_women, 'Women', districts)

# TODO: age distr (men and women)
