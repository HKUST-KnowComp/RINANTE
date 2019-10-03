import os
import json
import config
from utils import utils, datautils
from rule import ruleutils
from rule.aspectrulemine import AspectMineTool
from rule.opinionrulemine import OpinionMineTool


def __evaluate(terms_sys_list, terms_true_list, dep_tags_list, pos_tags_list, sent_texts):
    correct_sent_idxs = list()
    hit_cnt, true_cnt, sys_cnt = 0, 0, 0
    for sent_idx, (terms_sys, terms_true, dep_tags, pos_tags) in enumerate(
            zip(terms_sys_list, terms_true_list, dep_tags_list, pos_tags_list)):
        true_cnt += len(terms_true)
        sys_cnt += len(terms_sys)
        # new_hit_cnt = __count_hit(terms_true, aspect_terms)
        new_hit_cnt = utils.count_hit(terms_true, terms_sys)
        if new_hit_cnt == len(terms_true) and new_hit_cnt == len(terms_sys):
            correct_sent_idxs.append(sent_idx)
        hit_cnt += new_hit_cnt
        # if len(terms_true) and new_hit_cnt < len(terms_true):
        #     print(terms_true)
        #     print(terms_sys)
        #     print(sent_texts[sent_idx])
        #     print(pos_tags)
        #     print(dep_tags)
        #     print()

    # __save_never_hit_terms(sents, terms_sys_list, 'd:/data/aspect/semeval14/tmp.txt')

    print('hit={}, true={}, sys={}'.format(hit_cnt, true_cnt, sys_cnt))
    p = hit_cnt / (sys_cnt + 1e-8)
    r = hit_cnt / (true_cnt + 1e-8)
    print(p, r, 2 * p * r / (p + r + 1e-8))
    return correct_sent_idxs


def __write_rule_results(terms_list, sent_texts, output_file):
    if output_file is not None:
        fout = open(output_file, 'w', encoding='utf-8', newline='\n')
        for terms_sys, sent_text in zip(terms_list, sent_texts):
            # sent_obj = {'text': sent_text}
            # if terms_sys:
            #     sent_obj['terms'] = terms_sys
            fout.write('{}\n'.format(json.dumps(list(terms_sys), ensure_ascii=False)))
        fout.close()


def __run_with_mined_rules(mine_tool, rule_patterns_file, term_hit_rate_file, dep_tags_file, pos_tags_file,
                           sent_texts_file, filter_terms_vocab_file, term_hit_rate_thres=0.6,
                           output_result_file=None, sents_file=None):
    l1_rules, l2_rules = ruleutils.load_rule_patterns_file(rule_patterns_file)
    term_vocab = ruleutils.get_term_vocab(term_hit_rate_file, term_hit_rate_thres)

    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sent_texts = datautils.read_lines(sent_texts_file)
    filter_terms_vocab = set(datautils.read_lines(filter_terms_vocab_file))
    # opinion_terms_vocab = set(utils.read_lines(opinion_terms_file))

    terms_sys_list = list()
    for sent_idx, (dep_tag_seq, pos_tag_seq, sent_text) in enumerate(zip(dep_tags_list, pos_tags_list, sent_texts)):
        terms = set()
        l1_terms_new = set()
        for p in l1_rules:
            terms_new = ruleutils.find_terms_by_l1_pattern(
                p, dep_tag_seq, pos_tag_seq, mine_tool, filter_terms_vocab)
            terms.update(terms_new)
            l1_terms_new.update(terms_new)
        for p in l2_rules:
            terms_new = ruleutils.find_terms_by_l2_pattern(
                p, dep_tag_seq, pos_tag_seq, mine_tool, filter_terms_vocab, l1_terms_new)
            terms.update(terms_new)

        terms_new = mine_tool.get_terms_by_matching(dep_tag_seq, pos_tag_seq, sent_text, term_vocab)
        terms.update(terms_new)

        terms_sys_list.append(terms)

        if sent_idx % 10000 == 0:
            print(sent_idx)

    if output_result_file is not None:
        __write_rule_results(terms_sys_list, sent_texts, output_result_file)

    if sents_file is not None:
        sents = datautils.load_json_objs(sents_file)
        # aspect_terms_true = utils.aspect_terms_list_from_sents(sents)
        terms_list_true = mine_tool.terms_list_from_sents(sents)
        sent_texts = [sent['text'] for sent in sents]
        correct_sent_idxs = __evaluate(terms_sys_list, terms_list_true, dep_tags_list, pos_tags_list, sent_texts)


# term_type = 'opinion'
term_type = 'aspect'
target_dataset = 'se14l'
target_dataset_files = config.DATA_DICT[target_dataset]

if target_dataset.endswith('l'):
    res_dataset_files = config.AMAZON_LAPTOP_FILES
else:
    res_dataset_files = config.YELP_RESTAURANT_FILES

sents_file = None
sent_tok_texts_file = res_dataset_files['sent_tok_texts_file']
dep_tags_file = res_dataset_files['dep_tags_file']
pos_tags_file = res_dataset_files['pos_tags_file']
opinion_terms_file = os.path.join(config.DATA_DIR, 'opinion-terms-full.txt')
result_output_file = os.path.join(config.SE14_DIR, 'laptops/amazon-laptops-{}-rm-rule-result.txt'.format(term_type))

hit_rate_thres = 0.6

if term_type == 'aspect':
    mine_tool = AspectMineTool(opinion_terms_file)
    __run_with_mined_rules(
        mine_tool, target_dataset_files['aspect_rule_patterns_file'],
        target_dataset_files['aspect_term_hit_rate_file'], res_dataset_files['dep_tags_file'],
        res_dataset_files['pos_tags_file'], res_dataset_files['sent_texts_file'],
        target_dataset_files['aspect_term_filter_file'], term_hit_rate_thres=hit_rate_thres,
        output_result_file=result_output_file, sents_file=sents_file)

if term_type == 'opinion':
    mine_tool = OpinionMineTool()
    __run_with_mined_rules(
        mine_tool, target_dataset_files['opinion_rule_patterns_file'],
        target_dataset_files['opinion_term_hit_rate_file'], dep_tags_file,
        pos_tags_file, sent_tok_texts_file, target_dataset_files['opinion_term_filter_vocab_file'],
        term_hit_rate_thres=hit_rate_thres, output_result_file=result_output_file, sents_file=sents_file)
