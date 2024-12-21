import os
import shutil
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
import numpy as np
import pandas as pd

from bk21repro.constants import *
from bk21repro.data import get_df_from_ibex_dir
from bk21repro import resources


# Get experimental lists
with pkg_resources.as_file(pkg_resources.files(resources).joinpath('List1.csv')) as path:
    list1 = pd.read_csv(path)
with pkg_resources.as_file(pkg_resources.files(resources).joinpath('List2.csv')) as path:
    list2 = pd.read_csv(path)
with pkg_resources.as_file(pkg_resources.files(resources).joinpath('List3.csv')) as path:
    list3 = pd.read_csv(path)
assert len(list1) == len(list2) == len(list3), 'Stimulus list lengths do not match'
n_items = len(list1)
list1['selected_list'] = 'List1.csv'
list2['selected_list'] = 'List2.csv'
list3['selected_list'] = 'List3.csv'
lists = pd.concat([list1, list2, list3])
lists = lists[['Item', 'Cloze', 'selected_list']]
lists = lists.rename(dict(Item='ITEM', Cloze='condition'), axis=1)
lists.condition = lists.condition.map(dict(L='LC', M='MC', H='HC'))

# Get item data
if not os.path.exists(OSF):
    config = '''[osf]
    username = %s
    project = b9kns
    '''
    config_path = '.osfcli.config'
    uname = input('Please input your OSF username: ')
    with open(config_path, 'w') as f:
        f.write(config % uname)
    os.system('osf clone')
    shutil.move(os.path.join(OSF, 'osfstorage'), './')
    shutil.rmtree(OSF)
    shutil.move('osfstorage', OSF)
    os.remove(config_path)

BK_orig = pd.read_csv(os.path.join(OSF, 'SPRT_LogLin_216.csv'))
items = BK_orig[['ITEM', 'position', 'critical_word', 'condition', 'cloze', 'log_cloze', 'trigram', 'log_trigram']]
items = items.drop_duplicates()

with pkg_resources.as_file(pkg_resources.files(resources).joinpath('gpt.csv')) as path:
    gpt_items = pd.read_csv(path)
gpt_items = gpt_items.rename(dict(group='ITEM'), axis=1)
gpt_items = gpt_items[['ITEM', 'condition'] + GPT_COLS]

# Get experiment data by munging horrible Ibex output
df = get_df_from_ibex_dir(IBEX_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Merge data
n_prelim = df.ITEM.max() - n_items
df.ITEM -= n_prelim
df = pd.merge(df, lists, on=['ITEM', 'selected_list'])
df.ITEM += 4 # For some reason the BK item numbers start at 5
df = df.sort_values(['SUB', 'time', 'ITEM', 'sentpos'])

# Timestamp things
# Events are timestamped relative to the END of each SPR trial. Fix this.
# 1. Get trial durations
df['item_end'] = df.time
df['item_duration'] = df.groupby(['SUB', 'ITEM'])['RT'].transform('sum')
# 2. Subtract trial durations from timestamps
df.time -= df.item_duration
# 3. Compute word onsets from RT cumsums
df.time += df.groupby(['SUB', 'ITEM']).RT.\
    transform(lambda x: x.cumsum().shift(1, fill_value=0))
# 4. Subtract out the minimum timestamp to make timestamps relative to expt start
df['expt_start'] = df.groupby('SUB')['time'].transform('min')
df.time -= df.expt_start
df.question_response_timestamp -= df.expt_start
df.item_end -= df.expt_start
# 5. Get question RTs
df['question_RT'] = df.question_response_timestamp - df.item_end
# 6. Rescale to seconds
df.time /= 1000
df.question_response_timestamp /= 1000

# Save full word-level dataset
cols = [x for x in COLS if x in df]
df = df[cols]
df.to_csv(os.path.join(DATA_DIR, 'words.csv'), index=False)

# Compile and save item-level dataset
df = pd.merge(df, items, on=['ITEM', 'condition'])
# B&K gave half a count (out of an average 90 completions per item) for items with cloze 0
df['clozeprob'] = df.cloze.where(df.cloze > 0, 0.5 / 90)
df['cloze'] = -np.log(df.clozeprob)
df['critical_offset'] = df['sentpos'] - df['position']
df = df[(df.critical_offset >= 0) & (df.critical_offset < 3)]
df['SUM_3RT'] = df.groupby(['SUB', 'ITEM'])['RT'].transform('sum')
df = df[df.critical_offset == 0]
del df['critical_offset']
df['cutoff'] = df.groupby(['SUB'])['SUM_3RT'].transform('mean') + \
               df.groupby(['SUB'])['SUM_3RT'].transform('std') * 3
df['SUM_3RT_trimmed'] = df[['SUM_3RT', 'cutoff']].min(axis=1)
df['cutoff'] = 300
df['SUM_3RT_trimmed'] = df[['SUM_3RT_trimmed', 'cutoff']].max(axis=1)
del df['cutoff']
df = pd.merge(df, gpt_items, on=['ITEM', 'condition'])
df = df.sort_values(['SUB', 'ITEM'])
df.to_csv(os.path.join(DATA_DIR, 'items.csv'), index=False)



