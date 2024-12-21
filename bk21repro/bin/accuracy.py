import os
from bk21repro.constants import *
from bk21repro.data import get_df_from_ibex_dir

if __name__ == '__main__':
    df = get_df_from_ibex_dir(IBEX_DIR)
    acc = df.drop_duplicates([PARTICIPANT_COL, ITEM_COL])\
        .groupby(PARTICIPANT_COL)\
        .correct.mean()\
        .reset_index()\
        .rename({'correct': 'accuracy'}, axis=1)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    acc.to_csv(os.path.join(DATA_DIR, 'accuracy.csv'), index=False)
