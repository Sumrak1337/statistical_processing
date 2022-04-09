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
results_path = Path(rep_path, 'results')
clear_data_path = Path(rep_path, 'clear_data')

df = pd.read_excel(Path(clear_data_path, 'population_2015_2021.xlsx'))
file = open(Path(results_path, 'describing.txt'), 'w', encoding='utf-8')

districts = df['Unnamed: 0']
del df['Unnamed: 0']

plt.figure(figsize=(30, 25))

for i, row in df.iterrows():
    nan_policy = 'omit'
    file.write(f'{districts[i]} \n')
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

    row = row / 1e6
    plt.subplot(5, 3, i + 1)
    plt.title(districts[i])
    plt.xlabel('year')
    plt.ylabel('number of inhabitants (million people)')
    plt.plot(df.columns[np.where(~np.isnan(row))[0]], row.iloc[np.where(~np.isnan(row))], color=list(mcolors.TABLEAU_COLORS)[int(i % 10)], marker='o')
    plt.axhline(y=np.nanmean(row), linestyle='--', color='red', label='mean')
    x = df.columns[np.where(~np.isnan(row))[0]]
    x = np.array(x, dtype=float)
    plt.fill_between(x, row.iloc[np.where(~np.isnan(row))], np.nanmin(row.to_numpy()), color=list(mcolors.TABLEAU_COLORS)[int(i % 10)], alpha=0.7)
    plt.grid(True)
    plt.legend()

plt.savefig(Path(results_path, 'general_subplots.png'))
file.close()
