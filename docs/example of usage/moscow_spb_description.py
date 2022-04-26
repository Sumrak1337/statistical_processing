import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
from pathlib import Path

rep_path = Path.cwd()
data_path = Path(rep_path, 'data')

df = pd.read_excel(Path(data_path, 'moscow_spb.xlsx'))

main_df = pd.DataFrame()
main_df = main_df.append(df.iloc[-3])
main_df = main_df.append(df.iloc[-1])

del main_df["Unnamed: 0"]
del main_df["Unnamed: 1"]

main_df = main_df.reset_index(drop=True)
main_df.columns = np.array([year for year in range(2000, 2022)])

cities = np.array(["Moscow", "Saint-Petersburg"])

plt.figure()
plt.xlabel("years")
plt.ylabel("mln people")

for i, row in main_df.iterrows():
    print(cities[i])
    print('min: ', np.min(row), main_df.columns[np.argmin(row)])
    print('max: ', np.max(row), main_df.columns[np.argmax(row)])
    print('mean: ', np.mean(row))
    print('median: ', np.median(row))
    print('sd: ', np.std(row, ddof=1))
    print('interquartile range: ', stat.iqr(row))
    print('range: ', np.max(row) - np.min(row))
    print('skewness: ', stat.skew(row))
    print('kurtosis: ', stat.kurtosis(row))
    print()

    x = main_df.columns
    y = row / 1e6

    plt.plot(x, y, label=f'{cities[i]}', marker='o')
    plt.axhline(y=np.mean(y), linestyle='--', color='red', label=f'{cities[i]}_mean')
    plt.fill_between(x, y, alpha=0.5)
    plt.grid(True)
    plt.legend()

plt.savefig('moscow_spb.png')
plt.show()
