import pathlib

import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

"""
Центральный
Северо-западный
Южный
Северо-кавказский
Приволжский
Уральский
Сибирский
Дальневосточный
"""

rep_path = pathlib.Path.cwd().resolve().parents[0]
data_path = Path(rep_path, 'data')
results_path = Path(rep_path, 'results')
clear_data_path = Path(rep_path, 'clear_data')

tmp_df = pd.read_excel(Path(data_path, 'population_2014_2021.xlsx'))
df = pd.DataFrame()

for i, row in tmp_df.iterrows():
    if (i % 2 == 1) and (i != 1):
        df = df.append(row, ignore_index=True)

df = df.drop(df.columns[[0, 1]], axis=1)
columns_names = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
df.columns = columns_names

problems = [3, 8, 10]
clear_df = df.dropna()
for i in problems:
    s1 = df.loc[i]
    s2 = df.loc[i + 1]
    s = pd.concat([s1, s2]).dropna()
    clear_df = clear_df.append(s, ignore_index=True)

file = open(Path(results_path, 'describing.txt'), 'w')
districts = ['Российская Федерация',
             'Центральный ФО',
             'Северо-Западный ФО',
             'Северо-Кавказский ФО',
             'Приволжский ФО',
             'Уральский ФО',
             'Южный ФО',
             'Сибирский ФО',
             'Дальневосточный ФО']

plt.figure(figsize=(20, 15))
for i, row in clear_df.iterrows():
    file.write(f'{districts[i]}\n')
    file.write(f'min: {np.min(row)} \n')
    file.write(f'max: {np.max(row)} \n')
    file.write(f'mean: {np.mean(row)} \n')
    file.write(f'median: {np.median(row)} \n')
    file.write(f'sd: {np.std(row, ddof=1)} \n')
    file.write(f'interquartile range: {stat.iqr(row)} \n')
    file.write(f'range: {np.max(row) - np.min(row)} \n')
    file.write(f'confidence interval with equal areas around the mean: '
               f'{stat.t.interval(0.95, len(row) - 1, loc=np.mean(row), scale=stat.sem(row))} \n')
    file.write(f'skewness: {stat.skew(row)} \n')
    file.write(f'kurtosis: {stat.kurtosis(row)} \n')
    file.write('\n')

    plt.subplot(3, 3, i + 1)
    plt.title(districts[i])
    plt.xlabel('year')
    plt.ylabel('number of inhabitants')
    plt.plot(columns_names, row, color=list(mcolors.TABLEAU_COLORS)[i], marker='o')
    plt.axhline(y=np.mean(row), linestyle='--', color='red', label='mean')
    plt.fill_between(columns_names, row, np.min(row), color=list(mcolors.TABLEAU_COLORS)[i], alpha=0.7)
    plt.grid(True)
    plt.legend()

plt.savefig(Path(results_path, 'subplots.png'))
file.close()
