import os
import re
import shutil
import csv
import pandas as pd


OSF = 'b9kns'
HEADER = re.compile('.*# (\d+)\. (.+)\.')
NAME_MAP = {
    'Order number of item': 'ITEM',
    'id': 'SUB',
    'Value': 'word',
    'Parameter': 'wordpos',
    'Reading time': 'RT'
}
COLS = list(NAME_MAP.values()) + ['sentence', 'question', 'correct', 'position', 'critical_word', 'condition', 'cloze', 'log_cloze', 'trigram', 'log_trigram']

# Get lists
list1 = pd.read_csv(os.path.join('resources', 'List1.csv'))
list2 = pd.read_csv(os.path.join('resources', 'List2.csv'))
list3 = pd.read_csv(os.path.join('resources', 'List3.csv'))
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
    username = cory.shain@gmail.com
    project = b9kns
    '''
    
    with open('.osfcli.config', 'w') as f:
        f.write(config)
    os.system('osf clone')
    shutil.move(os.path.join(OSF, 'osfstorage'), './')
    shutil.rmtree(OSF)
    shutil.move('osfstorage', OSF)

BK_orig = pd.read_csv(os.path.join(OSF, 'SPRT_LogLin_216.csv'))

items = BK_orig[['ITEM', 'position', 'critical_word', 'condition', 'cloze', 'log_cloze', 'trigram', 'log_trigram']]
items = items.drop_duplicates()

# Get experiment data by munging horrible Ibex output
dataset = []
for path in ('results_dev.csv', 'results_prod.csv'):
    with open(os.path.join('ibex', path), 'r') as f:
        reader = csv.reader(f)
        headers = []
        item = []
        question_result = None
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
                            dataset.append(item)
                        question_result = None
                        item = []
                    elif row['PennElementType'] == 'Controller-DashedSentence':
                        item.append(row)
                    elif row['PennElementType'] == 'Selector':
                        question_result = 'is_correct'

if not os.path.exists('data'):
    os.makedirs('data')

# Merge data
dataset = pd.concat(dataset, axis=0)
dataset.to_csv(os.path.join('data', 'all.csv'), index=False)
dataset = pd.read_csv(os.path.join('data', 'all.csv'))
dataset = dataset.rename(NAME_MAP, axis=1)
dataset.ITEM -= 1
dataset = pd.merge(dataset, lists, on=['ITEM', 'selected_list'])
dataset.ITEM += 4 # For some reason the BK item numbers start at 5
dataset = pd.merge(dataset, items, on=['ITEM', 'condition'])
dataset = dataset[COLS]
dataset.to_csv(os.path.join('data', 'all.csv'), index=False)

dataset['critical_offset'] = dataset['wordpos'] - dataset['position']
dataset = dataset[(dataset.critical_offset >= 0) & (dataset.critical_offset < 3)]
dataset['SUM_3RT'] = dataset.groupby(['SUB', 'ITEM'])['RT'].transform('sum')
dataset = dataset[dataset.critical_offset == 0]
del dataset['critical_offset']
dataset['cutoff'] = dataset.groupby(['SUB'])['SUM_3RT'].transform('mean') + dataset.groupby(['SUB'])['SUM_3RT'].transform('std') * 3
dataset['SUM_3RT_trimmed'] = dataset[['SUM_3RT', 'cutoff']].min(axis=1)
dataset['cutoff'] = 300
dataset['SUM_3RT_trimmed'] = dataset[['SUM_3RT_trimmed', 'cutoff']].max(axis=1)
del dataset['cutoff']
print(dataset)
dataset.to_csv(os.path.join('data', 'main.csv'))



