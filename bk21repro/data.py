import os
import csv
from tempfile import TemporaryDirectory
import pandas as pd

from bk21repro.constants import *

def get_df_from_ibex_file(path):
    with open(os.path.join('ibex', path), 'r') as f:
        reader = csv.reader(f)
        headers = []
        item = []
        df = []
        question_result = None
        question_time = None
        for line in reader:
            if len(line):
                if line[0].startswith('#'):
                    res = HEADER.match(line[0])
                    if res:
                        ix, col = res.groups()
                        ix = int(ix) - 1
                        headers = headers[:ix]
                        headers.insert(ix, col)
                else:
                    row = dict(zip(headers, line))

                    if row['PennElementType'] == 'PennController':
                        if len(item):
                            item = pd.DataFrame(item)
                            item['correct'] = question_result
                            item['question_response_timestamp'] = question_time
                            df.append(item)
                        question_result = None
                        question_time = None
                        item = []
                    elif row['PennElementType'] == 'Controller-SPR' and row['Label'] != 'practice_trial':
                        item.append(row)
                    elif row['PennElementType'] == 'Selector' and row['Label'] != 'practice_trial':
                        question_result = row['is_correct']
                        if question_result == 'correct':
                            question_result = 1
                        else:
                            question_result = 0
                        question_time = row['EventTime']

    assert len(df), 'No data found in Ibex directory. Have you placed the source data in %s?' % IBEX_DIR
    df = pd.concat(df, axis=0)
    df = df.rename(NAME_MAP, axis=1)
    with TemporaryDirectory() as tmp_dir_path:
        df.to_csv(os.path.join(tmp_dir_path, 'words.csv'), index=False)
        df = pd.read_csv(os.path.join(tmp_dir_path, 'words.csv'))

    return df

def get_df_from_ibex_dir(path):
    dataset = []
    for file in [x for x in os.listdir(path) if x.endswith('.csv')]:
        dataset.append(get_df_from_ibex_file(file))
    dataset = pd.concat(dataset, axis=0)

    return dataset