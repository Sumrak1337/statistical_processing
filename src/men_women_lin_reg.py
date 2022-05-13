import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path

rep_path = Path.cwd().resolve().parents[0]
results_path = Path(rep_path, 'results')
clear_data_path = Path(rep_path, 'clear_data')
graph_path = Path(results_path, 'rf')
colors = list(mcolors.TABLEAU_COLORS)


def descriptive_stats():
    for gender in ['men', 'women']:
        df = pd.read_excel(Path(clear_data_path, f'{gender}_rf_2015_2022.xlsx'))
        index_name = df.iloc[:, 0].to_numpy()
        x_lbl = np.arange(len(index_name))
        del df["Unnamed: 0"]

        for i, column in enumerate(df.columns):
            data = df[column] / 1e3
            descriptive_text = f'min: {np.min(data):.4}, ({index_name[np.argmin(data)]})\n' \
                               f'max: {np.max(data):.4}, ({index_name[np.argmax(data)]})\n' \
                               f'mean: {np.mean(data):.4}\n' \
                               f'median: {np.median(data):.4}\n' \
                               f'sd: {np.std(data, ddof=1):.4}\n' \
                               f'interquantile range: {stat.iqr(data):.4}\n' \
                               f'range: {np.max(data) - np.min(data):.4}\n' \
                               f'skewness: {stat.skew(data):.4}\n' \
                               f'kurtosis: {stat.kurtosis(data):.4}\n'

            plt.figure(figsize=(20, 10))
            plt.bar(x_lbl, data, 0.8)
            plt.xticks(x_lbl, index_name, rotation=90)
            plt.xlabel("Age")
            plt.ylabel("Thousand people")
            plt.title(f"{column}, {gender}")
            plt.text(89, np.mean(data), descriptive_text, fontsize='medium', bbox=dict(facecolor='none', edgecolor='black'))
            plt.savefig(Path(str(results_path) + '/rf', f'{column}_{gender}.png'))
            plt.close('all')


def general_plots():
    df_men = pd.read_excel(Path(clear_data_path, 'men_rf_2015_2022.xlsx'))
    df_women = pd.read_excel(Path(clear_data_path, 'women_rf_2015_2022.xlsx'))
    index_name = df_men.iloc[:, 0].to_numpy()
    del df_men["Unnamed: 0"]
    del df_women["Unnamed: 0"]

    for year in df_men.columns:
        plt.figure(figsize=(20, 10))
        plt.title(f'{year}')
        x_lbl = np.arange(len(index_name))
        plt.bar(x_lbl - 0.2, df_men[year] / 1e3, 0.4, label='men')
        plt.bar(x_lbl + 0.2, df_women[year] / 1e3, 0.4, label='women')
        plt.xticks(x_lbl, index_name, rotation=90)
        plt.xlabel("Age")
        plt.ylabel("Thousand people")
        plt.legend()
        plt.savefig(Path(str(results_path) + '/rf', f'general_{year}.png'))
        plt.close('all')


def corr_coeff():
    """
    Comparison of the correlation coefficient in 2015 and 2022
    """
    df_men = pd.read_excel(Path(clear_data_path, 'men_rf_2015_2022.xlsx'))
    df_women = pd.read_excel(Path(clear_data_path, 'women_rf_2015_2022.xlsx'))
    men2015 = df_men[2015]
    men2022 = df_men[2022]
    women2015 = df_women[2015]
    women2022 = df_women[2022]

    print(men2015.corr(women2015))
    print(men2022.corr(women2022))
    # TODO: ?


def lin_reg_for_mean():
    df_men = pd.read_excel(Path(clear_data_path, 'men_rf_2015_2022.xlsx'))
    df_women = pd.read_excel(Path(clear_data_path, 'women_rf_2015_2022.xlsx'))
    ages = df_men.iloc[:, 0]
    del df_men["Unnamed: 0"]
    del df_women["Unnamed: 0"]
    years = np.array(df_men.columns).reshape(-1, 1)

    men_mean_age = np.array([])
    women_mean_age = np.array([])

    for year in df_men.columns:
        men_mean_age = np.append(men_mean_age, (df_men[year] @ ages + df_men[year][0]) / df_men[year].sum())
        women_mean_age = np.append(women_mean_age, (df_women[year] @ ages + df_women[year][0]) / df_women[year].sum())

    reg_men = LinearRegression().fit(years, men_mean_age)
    reg_women = LinearRegression().fit(years, women_mean_age)

    plt.figure()
    plt.xlabel('years')
    plt.ylabel('age')
    plt.title('Linear Regression for mean age')
    plt.scatter(years, men_mean_age, color=colors[3])
    plt.scatter(years, women_mean_age, color=colors[3])
    y_pred_men = reg_men.predict(years)
    y_pred_women = reg_women.predict(years)
    r2_men = reg_men.score(years, men_mean_age)
    r2_women = reg_women.score(years, women_mean_age)
    plt.plot(years, y_pred_men, linestyle='--', label=f'men\n'
                                                      f'mce={mean_squared_error(men_mean_age, y_pred_men):.3}\n'
                                                      f'$R^2$={r2_men}')
    plt.plot(years, y_pred_women, linestyle='--', label=f'women\n'
                                                        f'mce={mean_squared_error(women_mean_age, y_pred_women):.3}\n'
                                                        f'$R^2$={r2_women}')
    plt.legend()
    plt.savefig(Path(str(results_path) + '/rf', 'linear_regression_for_mean_age.png'))


def lin_reg_75():
    df_men = pd.read_excel(Path(clear_data_path, 'men_rf_2015_2022.xlsx'))
    df_women = pd.read_excel(Path(clear_data_path, 'women_rf_2015_2022.xlsx'))
    df_men = df_men.iloc[75:].sum() / 1e6
    df_women = df_women.iloc[75:].sum() / 1e6
    del df_men["Unnamed: 0"]
    del df_women["Unnamed: 0"]
    years = np.array(df_men.index).reshape(-1, 1)

    reg_men = LinearRegression().fit(years, df_men)
    reg_women = LinearRegression().fit(years, df_women)

    plt.figure()
    plt.xlabel('years')
    plt.ylabel('million people')
    plt.title('Linear Regression for number of people aged 75+')
    plt.scatter(years, df_men, color=colors[3])
    plt.scatter(years, df_women, color=colors[3])
    y_pred_men = reg_men.predict(years)
    y_pred_women = reg_women.predict(years)
    r2_men = reg_men.score(years, df_men)
    r2_women = reg_women.score(years, df_women)
    plt.plot(years, y_pred_men, linestyle='--',
             label=f'men,\n'
                   f'mce={mean_squared_error(df_men, y_pred_men):.3}\n'
                   f'$R^2=${r2_men}')
    plt.plot(years, y_pred_women, linestyle='--',
             label=f'women,\n'
                   f'mce={mean_squared_error(df_women, y_pred_women):.3}\n'
                   f'$R^2=${r2_women}')
    plt.legend()
    plt.savefig(Path(str(results_path) + '/rf', 'linear_regression_for_75+.png'))


# descriptive_stats()
# general_plots()
# corr_coeff()
lin_reg_for_mean()
lin_reg_75()
