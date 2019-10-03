import os
from rule.aspectrulemine import AspectMineTool
from rule.opinionrulemine import OpinionMineTool
from rule import rulemine
import config
from utils import datautils


def __gen_word_cnts_file(tok_texts_file, output_file):
    import pandas as pd

    texts = datautils.read_lines(tok_texts_file)
    word_cnts_dict = dict()
    total_word_cnt = 0
    for i, sent_text in enumerate(texts):
        if i % 2 == 1:
            continue
        words = sent_text.split()
        total_word_cnt += len(words)
        for w in words:
            cnt = word_cnts_dict.get(w, 0)
            word_cnts_dict[w] = cnt + 1

    word_cnt_tups = list(word_cnts_dict.items())
    word_cnt_tups.sort(key=lambda x: -x[1])

    word_cnt_rate_tups = list()
    for w, cnt in word_cnt_tups:
        word_cnt_rate_tups.append((w, cnt, cnt / total_word_cnt))
    df = pd.DataFrame(word_cnt_rate_tups, columns=['word', 'cnt', 'p'])
    with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
        df.to_csv(fout, index=False, float_format='%.5f')
    print(total_word_cnt)


dataset = 'se14l'
# dataset = 'se14r'
# dataset = 'se15r'
data_files = config.DATA_DICT[dataset]
# target = 'opinion'
target = 'aspect'

if target == 'aspect':
    term_filter_rate = 0.1
    pattern_filter_rate = 0.6
else:
    term_filter_rate = 0.1
    pattern_filter_rate = 0.6

if dataset.endswith('l'):
    freq_thres = 10
else:
    freq_thres = 10

opinion_terms_file = os.path.join(config.DATA_DIR, 'opinion-terms-full.txt')

dep_tags_file = data_files['train_dep_tags_file']
pos_tags_file = data_files['train_pos_tags_file']
word_cnts_file = data_files['word_cnts_file']
sents_file = data_files['train_sents_file']
train_valid_split_file = data_files['train_valid_split_file']
patterns_file = data_files['{}_rule_patterns_file'.format(target)]
term_filter_file = data_files['{}_term_filter_file'.format(target)]
term_hit_rate_file = data_files['{}_term_hit_rate_file'.format(target)]

if target == 'aspect':
    mine_tool = AspectMineTool(opinion_terms_file)
else:
    mine_tool = OpinionMineTool()

__gen_word_cnts_file(data_files['train_tok_texts_file'], word_cnts_file)
rulemine.gen_rule_patterns(mine_tool, dep_tags_file, pos_tags_file, sents_file, train_valid_split_file,
                           word_cnts_file, freq_thres, term_filter_rate, pattern_filter_rate, patterns_file)
rulemine.gen_filter_terms_vocab_file(mine_tool, dep_tags_file, pos_tags_file, sents_file, term_filter_rate,
                                     term_filter_file)
rulemine.gen_term_hit_rate_file(mine_tool, sents_file, dep_tags_file, pos_tags_file, term_hit_rate_file)
