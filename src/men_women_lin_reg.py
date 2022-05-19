import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
    del df_men["Unnamed: 0"]
    del df_women["Unnamed: 0"]
    men2015 = df_men[2015]
    men2022 = df_men[2022]
    women2015 = df_women[2015]
    women2022 = df_women[2022]

    print('======relation=====')
    df_rel = df_men / df_women
    plt.figure()
    plt.title(r'$Y = \frac{men}{women}$')
    plt.xlabel('Age')
    plt.ylabel('Y')
    for year in df_rel.columns:
        plt.plot(df_rel.index, df_rel[year], label=f'{year}')
    plt.legend()
    plt.show()
    plt.close('all')
    print('=====correlation======')
    print('2015: ', men2015.corr(women2015))
    print('2022: ', men2022.corr(women2022))
    print('-----------')
    previous_row = None
    for i, row in df_men.iterrows():
        if i == 0:
            previous_row = row
            continue
        print(f'men aged {i - 1} - {i}: {row.corr(previous_row):.4}')
        previous_row = row

    for i, row in df_women.iterrows():
        if i == 0:
            previous_row = row
            continue
        print(f'women aged {i - 1} - {i}: {row.corr(previous_row):.4}')
        previous_row = row
    # TODO: find a way how to search in cycle for 2 frames


def lin_reg_for_mean():
    df_men = pd.read_excel(Path(clear_data_path, 'men_rf_2015_2022.xlsx'))
    df_women = pd.read_excel(Path(clear_data_path, 'women_rf_2015_2022.xlsx'))
    df = df_men + df_women
    ages = df_men.iloc[:, 0]
    del df_men["Unnamed: 0"]
    del df_women["Unnamed: 0"]
    del df["Unnamed: 0"]
    years = np.array(df_men.columns).reshape(-1, 1)

    men_mean_age = np.array([])
    women_mean_age = np.array([])
    general_mean_age = np.array([])

    for year in df_men.columns:
        men_mean_age = np.append(men_mean_age, (df_men[year] @ ages + df_men[year][0]) / df_men[year].sum())
        women_mean_age = np.append(women_mean_age, (df_women[year] @ ages + df_women[year][0]) / df_women[year].sum())
        general_mean_age = np.append(general_mean_age, (df[year] @ ages + df[year][0]) / df[year].sum())

    reg_men = LinearRegression()
    reg_men.fit(years, men_mean_age)
    reg_women = LinearRegression()
    reg_women.fit(years, women_mean_age)
    reg_gen = LinearRegression()
    reg_gen.fit(years, general_mean_age)

    plt.figure()
    plt.xlabel('years')
    plt.ylabel('age')
    plt.title('Linear Regression for mean age')
    plt.scatter(years, men_mean_age, color=colors[3])
    plt.scatter(years, women_mean_age, color=colors[3])
    plt.scatter(years, general_mean_age, color=colors[3])
    y_pred_men = reg_men.predict(years)
    y_pred_women = reg_women.predict(years)
    y_pred_general = reg_gen.predict(years)
    r2_men = reg_men.score(years, men_mean_age)
    r2_women = reg_women.score(years, women_mean_age)
    r2_general = reg_gen.score(years, general_mean_age)
    plt.plot(years, y_pred_men, linestyle='--', label=f'men\n'
                                                      f'mse={mean_squared_error(men_mean_age, y_pred_men):.3}\n'
                                                      f'coef={reg_men.coef_[0]:.3}\n'
                                                      f'$R^2$={r2_men:.6}')
    plt.plot(years, y_pred_women, linestyle='--', label=f'women\n'
                                                        f'mse={mean_squared_error(women_mean_age, y_pred_women):.3}\n'
                                                        f'coef={reg_women.coef_[0]:.3}\n'
                                                        f'$R^2$={r2_women:.6}')
    plt.plot(years, y_pred_general, linestyle='--', label=f'general\n'
                                                        f'mse={mean_squared_error(general_mean_age, y_pred_general):.3}\n'
                                                        f'coef={reg_gen.coef_[0]:.3}\n'
                                                        f'$R^2$={r2_general:.6}')
    plt.legend()
    plt.savefig(Path(str(results_path) + '/rf', 'linear_regression_for_mean_age.png'))
    plt.close('all')


def lin_reg_75():
    df_men = pd.read_excel(Path(clear_data_path, 'men_rf_2015_2022.xlsx'))
    df_women = pd.read_excel(Path(clear_data_path, 'women_rf_2015_2022.xlsx'))
    df_men = df_men.iloc[75:].sum() / 1e6
    df_women = df_women.iloc[75:].sum() / 1e6
    del df_men["Unnamed: 0"]
    del df_women["Unnamed: 0"]
    years = np.array(df_men.index).reshape(-1, 1)

    reg_men = LinearRegression()
    reg_men.fit(years, df_men)
    reg_women = LinearRegression()
    reg_women.fit(years, df_women)

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
             label=f'men\n'
                   f'mse={mean_squared_error(df_men, y_pred_men):.3}\n'
                   f'coef={reg_men.coef_[0]:.3}\n'
                   f'$R^2=${r2_men:.6}')
    plt.plot(years, y_pred_women, linestyle='--',
             label=f'women\n'
                   f'mse={mean_squared_error(df_women, y_pred_women):.3}\n'
                   f'coef={reg_women.coef_[0]:.3}\n'
                   f'$R^2=${r2_women:.6}')
    plt.legend()
    plt.savefig(Path(str(results_path) + '/rf', 'linear_regression_for_75+.png'))
    plt.close('all')


def general_lin_reg():
    df = pd.read_excel(Path(clear_data_path, 'population_2015_2021.xlsx'))
    del df["Unnamed: 0"]
    indices = df.columns
    df_rf = np.array(df.iloc[0] / 1e6)
    years = np.array(indices).astype(float)

    # polynomial regression
    fit2 = np.polyfit(years, df_rf, 2)
    quad_reg = np.poly1d(fit2)
    fit3 = np.polyfit(years, df_rf, 4)
    quat_reg = np.poly1d(fit3)

    # linear regression
    years = years.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(years, df_rf)

    # plotting
    plt.figure()
    plt.title('Linear and Polynomial Regression of Population RF')
    plt.xlabel('years')
    plt.ylabel('million people')
    plt.scatter(years, df_rf, color=colors[3])
    y_pred = lin_reg.predict(years)
    plt.plot(years, y_pred, linestyle='--',
             label=f'Linear Regression\n'
                   f'coef={lin_reg.coef_[0]:.3}\n'
                   f'mse={mean_squared_error(df_rf, y_pred):.3}\n'
                   f'$R^2$={lin_reg.score(years, df_rf):.6}')
    x = np.linspace(np.min(years), np.max(years))
    y2 = quad_reg(x)
    plt.plot(x, y2, linestyle='--',
             label=f'Polynomial Regression(deg=2)\n'
                   f'mse={mean_squared_error(df_rf, quad_reg(years)):.3}\n'
                   f'$R^2$={r2_score(df_rf, quad_reg(years)):.6}')
    y4 = quat_reg(x)
    plt.plot(x, y4, linestyle='--',
             label=f'Polynomial Regression(deg=4)\n'
                   f'mse={mean_squared_error(df_rf, quat_reg(years)):.3}\n'
                   f'$R^2$={r2_score(df_rf, quat_reg(years)):.6}')
    plt.legend()
    plt.savefig(Path(str(results_path) + '/rf', 'general_lin_reg.png'))
    plt.close('all')


def men_lin_reg():
    df = pd.read_excel(Path(clear_data_path, 'men_rf_2015_2022.xlsx'))
    df = df.sum() / 1e6
    del df["Unnamed: 0"]
    years = np.array(df.index).astype(float)
    df = np.array(df)

    # polynomial regression
    fit2 = np.polyfit(years, df, 2)
    quad_reg = np.poly1d(fit2)
    fit3 = np.polyfit(years, df, 4)
    quat_reg = np.poly1d(fit3)

    # linear regression
    years = years.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(years, df)

    # plotting
    plt.figure()
    plt.title('Linear and Polynomial Regression of Men RF')
    plt.xlabel('years')
    plt.ylabel('million people')
    plt.scatter(years, df, color=colors[3])
    y_pred = lin_reg.predict(years)
    plt.plot(years, y_pred, linestyle='--',
             label=f'Linear Regression\n'
                   f'coef={lin_reg.coef_[0]:.3}\n'
                   f'mse={mean_squared_error(df, y_pred):.3}\n'
                   f'$R^2$={lin_reg.score(years, df):.6}')
    x = np.linspace(np.min(years), np.max(years))
    y2 = quad_reg(x)
    plt.plot(x, y2, linestyle='--',
             label=f'Polynomial Regression(deg=2)\n'
                   f'mse={mean_squared_error(df, quad_reg(years)):.3}\n'
                   f'$R^2$={r2_score(df, quad_reg(years)):.6}')
    y4 = quat_reg(x)
    plt.plot(x, y4, linestyle='--',
             label=f'Polynomial Regression(deg=4)\n'
                   f'mse={mean_squared_error(df, quat_reg(years)):.3}\n'
                   f'$R^2$={r2_score(df, quat_reg(years)):.6}')
    plt.legend()
    plt.savefig(Path(str(results_path) + '/rf', 'men_lin_reg.png'))
    plt.close('all')


def women_lin_reg():
    df = pd.read_excel(Path(clear_data_path, 'women_rf_2015_2022.xlsx'))
    df = df.sum() / 1e6
    del df["Unnamed: 0"]
    years = np.array(df.index).astype(float)
    df = np.array(df)

    # polynomial regression
    fit2 = np.polyfit(years, df, 2)
    quad_reg = np.poly1d(fit2)
    fit3 = np.polyfit(years, df, 4)
    quat_reg = np.poly1d(fit3)

    # linear regression
    years = years.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(years, df)

    # plotting
    plt.figure()
    plt.title('Linear and Polynomial Regression of Women RF')
    plt.xlabel('years')
    plt.ylabel('million people')
    plt.scatter(years, df, color=colors[3])
    y_pred = lin_reg.predict(years)
    plt.plot(years, y_pred, linestyle='--',
             label=f'Linear Regression\n'
                   f'coef={lin_reg.coef_[0]:.3}\n'
                   f'mse={mean_squared_error(df, y_pred):.3}\n'
                   f'$R^2$={lin_reg.score(years, df):.6}')
    x = np.linspace(np.min(years), np.max(years))
    y2 = quad_reg(x)
    plt.plot(x, y2, linestyle='--',
             label=f'Polynomial Regression(deg=2)\n'
                   f'mse={mean_squared_error(df, quad_reg(years)):.3}\n'
                   f'$R^2$={r2_score(df, quad_reg(years)):.6}')
    y4 = quat_reg(x)
    plt.plot(x, y4, linestyle='--',
             label=f'Polynomial Regression(deg=4)\n'
                   f'mse={mean_squared_error(df, quat_reg(years)):.3}\n'
                   f'$R^2$={r2_score(df, quat_reg(years)):.6}')
    plt.legend()
    plt.savefig(Path(str(results_path) + '/rf', 'women_lin_reg.png'))
    plt.close('all')

# descriptive_stats()
# general_plots()
# corr_coeff()
# lin_reg_for_mean()
# lin_reg_75()
# general_lin_reg()
# men_lin_reg()
# women_lin_reg()
