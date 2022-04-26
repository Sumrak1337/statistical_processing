import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
from pathlib import Path

rep_path = pathlib.Path.cwd().resolve().parents[0]
results_path = Path(rep_path, 'results')
clear_data_path = Path(rep_path, 'clear_data')
men_data_path = Path(clear_data_path, 'men')
women_data_path = Path(clear_data_path, 'women')
generals_bars_path = Path(results_path, 'general_bars')
men_results = Path(results_path, 'men')
women_results = Path(results_path, 'women')

districts_names = np.array(['Российская Федерация',
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

districts_number = len(districts_names)

# for men/women plots
for gender in ['men', 'women']:
    for ind in range(districts_number):
        df = pd.read_excel(Path(clear_data_path, f'{gender}', f'{gender}_2015_2022_{ind}.xlsx'))
        index_name = df.iloc[:, 0].to_numpy()
        del df["Unnamed: 0"]

        # TODO: add animation ???
        for i, column in enumerate(df.columns):
            if np.all(np.isnan(df[column])):
                continue

            fig, ax = plt.subplots(figsize=(15, 10))
            sns.barplot(x=index_name, y=df[column] / 1e6)
            plt.xlabel("Age")
            plt.ylabel("Number of inhabitants (million people)")
            plt.title(f"{column}, {districts_names[ind]}")
            plt.grid()
            plt.savefig(Path(str(results_path) + f'/{gender}', f'{gender}_{ind}_{column}.png'))
            plt.close('all')

# for general plots
# for gender in ['men', 'women']:
for i in range(districts_number):
    df_men = pd.read_excel(Path(men_data_path, f'men_2015_2022_{i}.xlsx'))
    df_women = pd.read_excel(Path(women_data_path, f'women_2015_2022_{i}.xlsx'))
    index_name = df_men.iloc[:, 0].to_numpy()
    del df_men["Unnamed: 0"]
    del df_women["Unnamed: 0"]

    for year in df_men.columns:
        if np.all(np.isnan(df_men[year])):
            continue

        plt.figure(figsize=(15, 10))
        plt.title(f'{districts_names[i]}, {year}')
        x_lbl = np.arange(len(index_name))
        plt.bar(x_lbl - 0.2, df_men[year] / 1e6, 0.4, label='men')
        plt.bar(x_lbl + 0.2, df_women[year] / 1e6, 0.4, label='women')
        plt.xticks(x_lbl, index_name, rotation=45)
        plt.xlabel("Age")
        plt.ylabel("Million people")
        plt.legend()
        plt.grid()
        plt.savefig(Path(generals_bars_path, f'general_{i}_{year}.png'))
        plt.close('all')
