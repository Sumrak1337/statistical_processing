import os
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig

from defaults import RESULTS_ROOT, DATA_ROOT, CLEAR_DATA_ROOT


@hydra.main(version_base=None, config_path='../configs', config_name="config")
def main(cfg: DictConfig):
    cfg_dem = cfg.full_demography

    # Read Excel file
    df = pd.read_excel(DATA_ROOT / f'{cfg_dem.name}.xlsx', sheet_name=0, index_col=0)
    # Drop the first column, because useless
    df.drop(columns=df.columns[0], inplace=True)
    # Rename columns from the first row and remove it
    df.columns = df.iloc[0]
    df.drop(index=df.index[0], inplace=True)
    df.columns = [col.split()[1] for col in df.columns]
    # Save temporary indices
    indices = [f'{idx} test' for idx in df.index if cfg_dem.index_word in idx]
    # Remove rows, where all values are nan
    df.dropna(how='all', inplace=True)
    # Reset useless index names
    df.reset_index(inplace=True, drop=True)
    # Set temporary index names
    df.rename(index={i: idx for i, idx in enumerate(indices)}, inplace=True)
    # Merge rows, where there exists nans. Fix special nan value
    drop_indices = []
    for indices in cfg_dem.problem_rows:
        lst = cfg_dem.problem_rows[indices]
        drop_indices += lst
        index_for_ufo_title = lst[0]
        title = " ".join(df.iloc[index_for_ufo_title].name.split()[:3])
        values = df.iloc[lst[0]]
        for idx in lst[1:]:
            values = values.add(df.iloc[idx], fill_value=0)
        # Rosstat has no info about UFO in 2003. Probably, it's a bug.
        values.fillna(value=cfg_dem.ufo_missed_value, inplace=True)
        df.loc[title] = values
    # Drop problem rows
    df.drop(index=df.index[drop_indices], inplace=True)
    # Set index names
    new_indices = [" ".join(idx.split()[:3]) for idx in df.index]
    df.rename(index={old_idx: new_idx for old_idx, new_idx in zip(df.index, new_indices)}, inplace=True)
    # Save df to .csv
    df.to_csv(CLEAR_DATA_ROOT / f'{cfg_dem.name}.csv')


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    sys.argv.append(f'hydra.run.dir="{RESULTS_ROOT}"')
    main()
