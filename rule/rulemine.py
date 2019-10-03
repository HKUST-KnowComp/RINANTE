from collections import namedtuple
import pandas as pd
from rule import ruleutils
from utils import datautils


RuleMineData = namedtuple('RuleMineData', ['dep_tag_seqs', 'pos_tag_seqs', 'sents'])


def __load_data(dep_tags_file, pos_tags_file, sents_file, train_valid_split_file):
    tvs_line = datautils.read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sents = datautils.load_json_objs(sents_file)

    assert len(tvs_arr) == len(dep_tags_list)

    dep_tags_list_train, dep_tags_list_valid = list(), list()
    pos_tags_list_train, pos_tags_list_valid = list(), list()
    sents_train, sents_valid = list(), list()
    for tvs_label, dep_tags, pos_tags, sent in zip(tvs_arr, dep_tags_list, pos_tags_list, sents):
        if tvs_label == 0:
            dep_tags_list_train.append(dep_tags)
            pos_tags_list_train.append(pos_tags)
            sents_train.append(sent)
        else:
            dep_tags_list_valid.append(dep_tags)
            pos_tags_list_valid.append(pos_tags)
            sents_valid.append(sent)

    data_train = RuleMineData(dep_tags_list_train, pos_tags_list_train, sents_train)
    data_valid = RuleMineData(dep_tags_list_valid, pos_tags_list_valid, sents_valid)
    return data_train, data_valid


def __get_term_filter_dict(dep_tag_seqs, pos_tag_seqs, terms_list_train, filter_rate, tool):
    term_cnts_dict = dict()
    for dep_tag_seq, pos_tag_seq, terms in zip(dep_tag_seqs, pos_tag_seqs, terms_list_train):
        term_cands = tool.get_candidate_terms(dep_tag_seq, pos_tag_seq)
        for t in term_cands:
            cnts = term_cnts_dict.get(t, (0, 0))
            hit_cnt = cnts[0] + 1 if t in terms else cnts[0]
            term_cnts_dict[t] = (hit_cnt, cnts[1] + 1)

    filter_terms = set()
    for t, (hit_cnt, cnt) in term_cnts_dict.items():
        if hit_cnt / cnt < filter_rate:
            filter_terms.add(t)
    return filter_terms


def __get_word_cnts_dict(word_cnts_file):
    with open(word_cnts_file, encoding='utf-8') as f:
        df = pd.read_csv(f)
    word_cnt_dict = dict()
    for w, cnt, _ in df.itertuples(False, None):
        word_cnt_dict[w] = cnt
    return word_cnt_dict


def __find_phrase_word_idx_span(phrase, sent_words):
    phrase_words = phrase.split()
    pleft = 0
    while pleft + len(phrase_words) <= len(sent_words):
        p = pleft
        while p - pleft < len(phrase_words) and sent_words[p] == phrase_words[p - pleft]:
            p += 1
        if p - pleft == len(phrase_words):
            return pleft, p
        pleft += 1
    return None


def __find_related_dep_patterns(dep_tags, pos_tags, term_word_idx_span, mine_tool):
    widx_beg, widx_end = term_word_idx_span
    # print(' '.join(sent_words[widx_beg: widx_end]))
    # for widx in range(widx_beg, widx_end):
    related_dep_tag_idxs = set()
    for i, dep_tag in enumerate(dep_tags):
        rel, gov, dep = dep_tag
        igov, wgov = gov
        idep, wdep = dep
        if not widx_beg <= igov < widx_end and not widx_beg <= idep < widx_end:
            continue
        if widx_beg <= igov < widx_end and widx_beg <= idep < widx_end:
            continue
        # print(dep_tag)
        related_dep_tag_idxs.add(i)

    # print(related_dep_tag_idxs)
    # patterns_l1 = __patterns_from_l1_dep_tags(aspect_word_wc,
    #     [dep_tags[idx] for idx in related_dep_tag_idxs], pos_tags, term_word_idx_span, opinion_terms)
    # print(patterns_new)
    # related_l2 = __find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags)
    # patterns_l2 = __patterns_from_l2_dep_tags(aspect_word_wc, related_l2, pos_tags, term_word_idx_span, opinion_terms)
    # return patterns_l1, patterns_l2
    return mine_tool.get_patterns_from_term(term_word_idx_span, related_dep_tag_idxs, dep_tags, pos_tags)


def __l1_pattern_legal(pattern, word_freq_dict):
    rel, gov, dep = pattern
    if not gov.startswith('_A') and gov != '_OP' and not gov.isupper() and word_freq_dict.get(gov, 0) < 10:
        return False
    if not dep.startswith('_A') and dep != '_OP' and not dep.isupper() and word_freq_dict.get(dep, 0) < 10:
        return False
    return True


def __filter_l1_patterns(patterns, word_freq_dict):
    patterns_new = list()
    for p in patterns:
        if __l1_pattern_legal(p, word_freq_dict):
            patterns_new.append(p)
    return patterns_new


def __filter_l2_patterns(patterns, word_freq_dict):
    patterns_new = list()
    for p in patterns:
        pl, pr = p
        if not __l1_pattern_legal(pl[0], word_freq_dict) or not __l1_pattern_legal(pr[0], word_freq_dict):
            # print(p)
            continue
        patterns_new.append(p)
    return patterns_new


def __find_rule_candidates(dep_tags_list, pos_tags_list, mine_tool, aspect_terms_list, word_cnts_file,
                           freq_thres=10):
    word_freq_dict = __get_word_cnts_dict(word_cnts_file)

    # sents = utils.load_json_objs(sents_file)
    cnt_miss, cnt_patterns = 0, 0
    patterns_l1_cnts, patterns_l2_cnts = dict(), dict()
    for sent_idx, (dep_tags, pos_tags) in enumerate(zip(dep_tags_list, pos_tags_list)):
        assert len(dep_tags) == len(pos_tags)
        sent_words = [dep_tup[2][1] for dep_tup in dep_tags]

        aspect_terms = aspect_terms_list[sent_idx]
        for term in aspect_terms:
            idx_span = __find_phrase_word_idx_span(term, sent_words)
            if idx_span is None:
                cnt_miss += 1
                continue

            patterns_l1_new, patterns_l2_new = __find_related_dep_patterns(
                dep_tags, pos_tags, idx_span, mine_tool)

            patterns_l1_new = __filter_l1_patterns(patterns_l1_new, word_freq_dict)
            patterns_l2_new = __filter_l2_patterns(patterns_l2_new, word_freq_dict)

            for p in patterns_l1_new:
                cnt = patterns_l1_cnts.get(p, 0)
                patterns_l1_cnts[p] = cnt + 1
            for p in patterns_l2_new:
                cnt = patterns_l2_cnts.get(p, 0)
                patterns_l2_cnts[p] = cnt + 1

            # patterns_l1.update(patterns_l1_new)
            # patterns_l2.update(patterns_l2_new)
        # if sent_idx >= 100:
        #     break

    # patterns_l1, patterns_l2 = set(), set()
    patterns_l1 = {p for p, cnt in patterns_l1_cnts.items() if cnt > freq_thres}
    patterns_l2 = {p for p, cnt in patterns_l2_cnts.items() if cnt > freq_thres}

    print(cnt_miss, 'terms missed')
    return patterns_l1, patterns_l2


def __filter_l1_patterns_through_matching(patterns, dep_tags_list, pos_tags_list, terms_list, mine_tool,
                                          filter_terms_vocab, pattern_filter_rate):
    patterns_keep = list()
    for p in patterns:
        hit_cnt, cnt = 0, 0
        for dep_tags, pos_tags, terms_true in zip(dep_tags_list, pos_tags_list, terms_list):
            terms_true = set(terms_true)
            terms = ruleutils.find_terms_by_l1_pattern(
                p, dep_tags, pos_tags, mine_tool, filter_terms_vocab)

            for term in terms:
                if term in filter_terms_vocab:
                    continue
                cnt += 1
                if term in terms_true:
                    hit_cnt += 1

        if hit_cnt / (cnt + 1e-5) > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(p)
    return patterns_keep


def __filter_l2_patterns_through_matching(patterns, dep_tags_list, pos_tags_list, terms_list, mine_tool,
                                          filter_terms_vocab, pattern_filter_rate):
    patterns_keep = list()
    for p in patterns:
        hit_cnt, cnt = 0, 0
        for dep_tags, pos_tags, terms_true in zip(dep_tags_list, pos_tags_list, terms_list):
            terms_true = set(terms_true)
            terms = ruleutils.find_terms_by_l2_pattern(
                p, dep_tags, pos_tags, mine_tool, filter_terms_vocab)
            for term in terms:
                cnt += 1
                if term in terms_true:
                    hit_cnt += 1
                    # print(p)
                    # print(dep_tag_l, dep_tag_r)
                    # print(term)

        if hit_cnt / (cnt + 1e-5) > pattern_filter_rate:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(p)
    return patterns_keep


def __match_l1_pattern(pattern, dep_tag, pos_tags, mine_tool):
    prel, pgov, pdep = pattern
    rel, (igov, wgov), (idep, wdep) = dep_tag
    if rel != prel:
        return False
    return mine_tool.match_pattern_word(pgov, wgov, pos_tags[igov]) and mine_tool.match_pattern_word(
        pdep, wdep, pos_tags[idep])


def __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, mine_tool):
    matched_idxs = list()
    for i, dep_tag in enumerate(dep_tags):
        if __match_l1_pattern(pattern, dep_tag, pos_tags, mine_tool):
            matched_idxs.append(i)
    return matched_idxs


def find_terms_by_l1_pattern(pattern, dep_tags, pos_tags, mine_tool, filter_terms_vocab):
    terms = list()
    matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, mine_tool)
    for idx in matched_dep_tag_idxs:
        term = mine_tool.get_term_from_matched_pattern(pattern, dep_tags, pos_tags, idx)

        if term in filter_terms_vocab:
            continue
        terms.append(term)
        # print(pattern)
        # print(term)
        # print(dep_tags)
        # print([t[2][1] for t in dep_tags])
        # print()
    return terms


def gen_rule_patterns(mine_tool, dep_tags_file, pos_tags_file, sents_file, train_valid_split_file,
                      word_cnts_file, freq_thres, term_filter_rate, pattern_filter_rate, output_file):
    # opinion_terms_vocab = set(utils.read_lines(opinion_terms_file))
    data_train, data_valid = __load_data(dep_tags_file, pos_tags_file, sents_file, train_valid_split_file)

    # aspect_terms_list_train = utils.aspect_terms_list_from_sents(data_train.sents)
    terms_list_train = mine_tool.terms_list_from_sents(data_train.sents)
    filter_terms_vocab = __get_term_filter_dict(
        data_train.dep_tag_seqs, data_train.pos_tag_seqs, terms_list_train, term_filter_rate, mine_tool)

    patterns_l1, patterns_l2 = __find_rule_candidates(
        data_train.dep_tag_seqs, data_train.pos_tag_seqs, mine_tool, terms_list_train,
        word_cnts_file, freq_thres)
    print(len(patterns_l1), 'l1 patterns', len(patterns_l2), 'l2 patterns')

    terms_list_valid = mine_tool.terms_list_from_sents(data_valid.sents)

    patterns_l1 = __filter_l1_patterns_through_matching(
        patterns_l1, data_valid.dep_tag_seqs, data_valid.pos_tag_seqs, terms_list_valid,
        mine_tool, filter_terms_vocab, pattern_filter_rate)

    patterns_l2 = __filter_l2_patterns_through_matching(
        patterns_l2, data_valid.dep_tag_seqs, data_valid.pos_tag_seqs, terms_list_valid,
        mine_tool, filter_terms_vocab, pattern_filter_rate)

    patterns_l1.sort()
    patterns_l2.sort()
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    for p in patterns_l1:
        fout.write('{}\n'.format(' '.join(p)))
    for p in patterns_l2:
        (pl, ipl), (pr, ipr) = p
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()


def gen_filter_terms_vocab_file(mine_tool, dep_tags_file, pos_tags_file, sents_file, term_filter_rate, output_file):
    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sents = datautils.load_json_objs(sents_file)
    # aspect_terms_list = datautils.aspect_terms_list_from_sents(sents)
    terms_list = mine_tool.terms_list_from_sents(sents)
    filter_terms_vocab = __get_term_filter_dict(
        dep_tags_list, pos_tags_list, terms_list, term_filter_rate, mine_tool)
    with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
        for t in filter_terms_vocab:
            fout.write('{}\n'.format(t))


def gen_term_hit_rate_file(mine_tool, train_sents_file, dep_tags_file, pos_tags_file, dst_file):
    dep_tags_list = datautils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = datautils.load_pos_tags(pos_tags_file)
    sents = datautils.load_json_objs(train_sents_file)
    terms_list = mine_tool.terms_list_from_sents(sents)
    term_hit_cnts = dict()
    for terms in terms_list:
        for t in terms:
            cnt = term_hit_cnts.get(t, 0)
            term_hit_cnts[t] = cnt + 1

    all_terms = set(term_hit_cnts.keys())
    print(len(all_terms), 'terms')
    term_cnts = {t: 0 for t in all_terms}
    # for t in term_hit_cnts.keys():
    for dep_tags, pos_tags, sent in zip(dep_tags_list, pos_tags_list, sents):
        sent_text = sent['text'].lower()
        terms = mine_tool.get_terms_by_matching(dep_tags, pos_tags, sent_text, all_terms)
        for t in terms:
            cnt = term_cnts.get(t, 0)
            term_cnts[t] = cnt + 1

    term_hit_rate_tups = list()
    for t, hit_cnt in term_hit_cnts.items():
        total_cnt = term_cnts.get(t, 0)
        if total_cnt > 0:
            term_hit_rate_tups.append((t, hit_cnt / (total_cnt + 1e-5)))

    term_hit_rate_tups.sort(key=lambda x: -x[1])

    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        pd.DataFrame(term_hit_rate_tups, columns=['term', 'rate']).to_csv(
            fout, float_format='%.4f', index=False)
