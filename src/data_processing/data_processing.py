import pandas as pd
import numpy as np
import pathlib
from pathlib import Path


def population_processing(file_name, index):
    column_names = np.array([year for year in range(2015, 2022)])

    df = pd.read_excel(Path(data_path, file_name))
    df = df.reset_index(drop=True)

    clear_df = pd.DataFrame()

    for ind, row in df.iterrows():
        if (ind % 2 == 1) and (ind != 1):
            clear_df = clear_df.append(row, ignore_index=True)

    clear_df.index = index
    clear_df = clear_df.drop("Unnamed: 0", axis='columns')
    clear_df = clear_df.drop("Unnamed: 1", axis='columns')
    clear_df.columns = column_names
    clear_df.to_excel(Path(clear_data_path, file_name), encoding='utf-8')


def men_women_processing(file_name, index):
    column_names = np.array([year for year in range(2015, 2023)])
    districts_number = 13
    age_number = 21

    df = pd.read_excel(Path(data_path, file_name))
    df = df.reset_index(drop=True)

    tmp_df = pd.DataFrame()
    clear_df = pd.DataFrame()

    for i, r in df.iterrows():
        if i < 2:
            continue
        if (i - 2) % (age_number + 1) != 0:
            tmp_df = tmp_df.append(r, ignore_index=True).astype(int, errors='ignore')

    del tmp_df['Unnamed: 0']
    del tmp_df['Unnamed: 1']

    for i in range(districts_number):
        sum_row = 0
        for j in range(age_number):
            sum_row += tmp_df.loc[j + age_number * i]
        clear_df = clear_df.append(sum_row, ignore_index=True)

    clear_df.index = index
    clear_df.columns = column_names
    clear_df.to_excel(Path(clear_data_path, file_name), encoding='utf-8')


def men_women_aged_processing(file_name, index):
    order = np.array([1, 2, 3, 4, 15, 16, 17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0])
    column_names = np.array([year for year in range(2015, 2023)])


rep_path = pathlib.Path.cwd().resolve().parents[1]
data_path = Path(rep_path, 'data')
clear_data_path = Path(rep_path, 'clear_data')

row_names = np.array(['Российская Федерация',
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

population_processing('population_2015_2021.xlsx', row_names)
men_women_processing('men_2015_2022.xlsx', row_names)
men_women_processing('women_2015_2022.xlsx', row_names)
men_women_aged_processing('men_2015_2022.xlsx', row_names)
men_women_aged_processing('women_2015_2022.xlsx', row_names)
