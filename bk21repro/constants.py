import re

OSF = 'b9kns'
HEADER = re.compile('.*# (\d+)\. (.+)\.')
NAME_MAP = {
    'id': 'SUB',
    'Results reception time': 'acquisition_date',
    'Order number of item': 'ITEM',
    'Value': 'word',
    'Parameter': 'sentpos',
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
IBEX_DIR = 'ibex'
DATA_DIR = 'data'
PARTICIPANT_COL = 'SUB'
ITEM_COL = 'ITEM'
INITIAL_CUTOFF = 4500