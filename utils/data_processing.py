import os
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig

from defaults import RESULTS_ROOT, DATA_ROOT, CLEAR_DATA_ROOT


@hydra.main(version_base=None, config_path='../configs', config_name="config")
def main(cfg: DictConfig):
    # Read Excel file
    df = pd.read_excel(DATA_ROOT / f'{cfg.full_demography.name}.xlsx', sheet_name=0, index_col=0)
    # Drop the first column, because useless
    df.drop(columns=df.columns[0], inplace=True)
    # Rename columns from the first row and remove it
    df.columns = df.iloc[0]
    df.drop(index=df.index[0], inplace=True)
    df.columns = [col.split()[1] for col in df.columns]
    # Save useful indices
    indices = [idx for idx in df.index if "округ" in idx]
    # Remove rows, where all values are nan
    df.dropna(how='all', inplace=True)
    # Reset useless index names
    df.reset_index(inplace=True, drop=True)
    # Set useful index names
    df.rename(index={i: idx for i, idx in enumerate(indices)}, inplace=True)
    # Merge rows, where there exists nans. Fix special nan value
    # Problem rows: 2-4, 8-9, 10-11
    ufo_title = " ".join(df.iloc[2].name.split()[:3])
    ufo_values = df.iloc[2].add(df.iloc[3].add(df.iloc[4], fill_value=0), fill_value=0)
    # Rosstat has no info about UFO in 2003. Probably, it's a bug.
    ufo_values.fillna(value=22891859, inplace=True)
    sfo_title = " ".join(df.iloc[8].name.split())
    sfo_values = df.iloc[8].add(df.iloc[9], fill_value=0)
    dfo_title = " ".join(df.iloc[10].name.split())
    dfo_values = df.iloc[10].add(df.iloc[11], fill_value=0)
    # Drop problem rows and set new
    df.drop(index=df.index[[2, 3, 4, 8, 9, 10, 11]], inplace=True)
    df.loc[ufo_title] = ufo_values
    df.loc[sfo_title] = sfo_values
    df.loc[dfo_title] = dfo_values
    # Save df to .csv
    df.to_csv(CLEAR_DATA_ROOT / f'{cfg.full_demography.name}.csv')


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    sys.argv.append(f'hydra.run.dir="{RESULTS_ROOT}"')
    main()
