import os
import re
import shutil
import csv
import numpy as np
import pandas as pd


# Global variables
OSF = 'b9kns'
HEADER = re.compile('.*# (\d+)\. (.+)\.')
NAME_MAP = {
    'Order number of item': 'ITEM',
    'id': 'SUB',
    'Value': 'word',
    'Parameter': 'wordpos',
    'EventTime': 'time',
    'Reading time': 'RT'
}
GPT_COLS = ['gpt2', 'gpt2prob', 'gpt2region',
            'gpt2regionprob', 'glovedistmin', 'glovedistmean' ,
            'unigram', 'unigramregion', 'wlen', 'wlenregion']
COLS = list(NAME_MAP.values()) + [
    'sentence', 'question', 'correct', 'question_response_timestamp', 'question_RT', 'position', 'critical_word',
    'condition', 'cloze', 'log_cloze', 'trigram', 'log_trigram'
] + GPT_COLS

# Get experimental lists
list1 = pd.read_csv(os.path.join('resources', 'List1.csv'))
list2 = pd.read_csv(os.path.join('resources', 'List2.csv'))
list3 = pd.read_csv(os.path.join('resources', 'List3.csv'))
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

gpt_items = pd.read_csv(os.path.join('resources', 'gpt.csv'))
gpt_items = gpt_items.rename(dict(group='ITEM'), axis=1)
gpt_items = gpt_items[['ITEM', 'condition'] + GPT_COLS]

# Get experiment data by munging horrible Ibex output
dataset = []
for path in [x for x in os.listdir('ibex') if x.endswith('.csv')]:
    with open(os.path.join('ibex', path), 'r') as f:
        reader = csv.reader(f)
        headers = []
        item = []
        question_result = None
        question_time = None
        start_time = None
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
                            dataset.append(item)
                        question_result = None
                        question_time = None
                        item = []
                    elif row['PennElementType'] == 'Controller-SPR':
                        item.append(row)
                    elif row['PennElementType'] == 'Selector' and row['Label'] != 'practice_trial':
                        question_result = row['is_correct']
                        if question_result == 'correct':
                            question_result = True
                        else:
                            question_result = False
                        question_time = row['EventTime']

if not os.path.exists('data'):
    os.makedirs('data')

# Merge data
dataset = pd.concat(dataset, axis=0)
dataset.to_csv(os.path.join('data', 'words.csv'), index=False)
dataset = pd.read_csv(os.path.join('data', 'words.csv'))
dataset = dataset.rename(NAME_MAP, axis=1)
n_prelim = dataset.ITEM.max() - n_items
dataset.ITEM -= n_prelim
dataset = pd.merge(dataset, lists, on=['ITEM', 'selected_list'])
dataset.ITEM += 4 # For some reason the BK item numbers start at 5
dataset = dataset.sort_values(['SUB', 'time', 'ITEM', 'wordpos'])

# Timestamp things
# Events are timestamped relative to the END of each SPR trial. Fix this.
# 1. Get trial durations
dataset['item_end'] = dataset.time
dataset['item_duration'] = dataset.groupby(['SUB', 'ITEM'])['RT'].transform('sum')
# 2. Subtract trial durations from timestamps
dataset.time -= dataset.item_duration
# 3. Compute word onsets from RT cumsums
dataset.time += dataset.groupby(['SUB', 'ITEM']).RT.\
    transform(lambda x: x.cumsum().shift(1, fill_value=0))
# 4. Subtract out the minimum timestamp to make timestamps relative to expt start
dataset['expt_start'] = dataset.groupby('SUB')['time'].transform('min')
dataset.time -= dataset.expt_start
dataset.question_response_timestamp -= dataset.expt_start
dataset.item_end -= dataset.expt_start
# 5. Get question RTs
dataset['question_RT'] = dataset.question_response_timestamp - dataset.item_end
# 6. Rescale to seconds
dataset.time /= 1000
dataset.question_response_timestamp /= 1000

# Save full word-level dataset
cols = [x for x in COLS if x in dataset]
dataset = dataset[cols]
dataset.to_csv(os.path.join('data', 'words.csv'), index=False)

# Compile and save item-level dataset
dataset = pd.merge(dataset, items, on=['ITEM', 'condition'])
# B&K gave half a count (out of an average 90 completions per item) for items with cloze 0
dataset['clozeprob'] = dataset.cloze.where(dataset.cloze > 0, 0.5 / 90)
dataset['cloze'] = -np.log(dataset.clozeprob)
dataset['critical_offset'] = dataset['wordpos'] - dataset['position']
dataset = dataset[(dataset.critical_offset >= 0) & (dataset.critical_offset < 3)]
dataset['SUM_3RT'] = dataset.groupby(['SUB', 'ITEM'])['RT'].transform('sum')
dataset = dataset[dataset.critical_offset == 0]
del dataset['critical_offset']
dataset['cutoff'] = dataset.groupby(['SUB'])['SUM_3RT'].transform('mean') + \
                    dataset.groupby(['SUB'])['SUM_3RT'].transform('std') * 3
dataset['SUM_3RT_trimmed'] = dataset[['SUM_3RT', 'cutoff']].min(axis=1)
dataset['cutoff'] = 300
dataset['SUM_3RT_trimmed'] = dataset[['SUM_3RT_trimmed', 'cutoff']].max(axis=1)
del dataset['cutoff']
dataset = pd.merge(dataset, gpt_items, on=['ITEM', 'condition'])
dataset = dataset.sort_values(['SUB', 'ITEM'])
dataset.to_csv(os.path.join('data', 'items.csv'), index=False)



