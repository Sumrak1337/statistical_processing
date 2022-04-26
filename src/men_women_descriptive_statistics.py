import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt


def write_to_file(df, file):
    nan_policy = 'omit'
    distrs = df['Unnamed: 0']
    del df['Unnamed: 0']
    for ind, row in df.iterrows():
        file.write(f'{distrs[ind]}\n')
        file.write(f'min: {int(np.nanmin(row))}, ({df.columns[np.argmin(row)]})\n')
        file.write(f'max: {int(np.nanmax(row))}, ({df.columns[np.argmax(row)]})\n')
        file.write(f'mean: {np.nanmean(row)}\n')
        file.write(f'median: {np.nanmedian(row)}\n')
        file.write(f'sd: {np.nanstd(row, ddof=1)}\n')
        file.write(f'interquartile range: {stat.iqr(row, nan_policy=nan_policy)}\n')
        file.write(f'range: {int(np.nanmax(row) - np.nanmin(row))}\n')
        file.write(f'skewness: {stat.skew(row, nan_policy=nan_policy)}\n')
        file.write(f'kurtosis: {stat.kurtosis(row, nan_policy=nan_policy)}\n')
        file.write('\n')
    file.close()


def plotting(df, title):
    distrs = df['Unnamed: 0']
    del df['Unnamed: 0']
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
results_path = Path(rep_path, 'results')
clear_data_path = Path(rep_path, 'clear_data')

df_men = pd.read_excel(Path(clear_data_path, 'men_2015_2022.xlsx'))
df_women = pd.read_excel(Path(clear_data_path, 'women_2015_2022.xlsx'))

file_men = open(Path(results_path, 'describing_men.txt'), 'w', encoding='utf-8')
file_women = open(Path(results_path, 'describing_women.txt'), 'w', encoding='utf-8')

write_to_file(df_men.copy(), file_men)
write_to_file(df_women.copy(), file_women)

plotting(df_men.copy(), 'Men')
plotting(df_women.copy(), 'Women')
