NOUN_POS_TAGS = {'NN', 'NNP', 'NNS', 'NNPS'}
VB_POS_TAGS = {'VB', 'VBN', 'VBP', 'VBZ', 'VBG', 'VBD'}
JJ_POS_TAGS = {'JJ', 'JJR', 'JJS'}
RB_POS_TAGS = {'RB', 'RBR'}


def load_rule_patterns_file(filename):
    l1_rules, l2_rules = list(), list()
    f = open(filename, encoding='utf-8')
    for line in f:
        vals = line.strip().split()
        assert len(vals) == 3 or len(vals) == 8
        if len(vals) == 3:
            l1_rules.append(vals)
        else:
            l2_rules.append(
                (((vals[0], vals[1], vals[2]), int(vals[3])), ((vals[4], vals[5], vals[6]), int(vals[7]))))
    f.close()
    return l1_rules, l2_rules


def get_term_vocab(aspect_term_hit_rate_file, rate_thres):
    import pandas as pd

    df = pd.read_csv(aspect_term_hit_rate_file)
    df = df[df['rate'] > rate_thres]
    return set(df['term'])


def find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags):
    related = list()
    for i in related_dep_tag_idxs:
        rel, gov, dep = dep_tags[i]
        igov, wgov = gov
        idep, wdep = dep
        for j, dep_tag_j in enumerate(dep_tags):
            if i == j:
                continue
            rel_j, gov_j, dep_j = dep_tag_j
            igov_j, wgov_j = gov_j
            idep_j, wdep_j = dep_j
            if igov == igov_j or igov == idep_j or idep == igov_j or idep == idep_j:
                # print(dep_tags[i], dep_tag_j)
                related.append((dep_tags[i], dep_tag_j))
    return related


def get_noun_phrases(words, pos_tags, nouns_filter):
    assert len(words) == len(pos_tags)

    noun_phrases = list()
    pleft = 0
    while pleft < len(words):
        if pos_tags[pleft] not in NOUN_POS_TAGS:
            pleft += 1
            continue
        pright = pleft + 1
        while pright < len(words) and pos_tags[pright] in {'NN', 'NNS', 'NNP', 'CD'}:
            pright += 1

        # if pleft > 0 and pos_tags[pleft - 1] == 'JJ' and words[pleft - 1] not in opinion_terms:
        #     pleft -= 1

        phrase = ' '.join(words[pleft: pright])
        if nouns_filter is None or phrase not in nouns_filter:
            noun_phrases.append(phrase)
        pleft = pright
    # print(' '.join(words))
    # print(noun_phrases)
    return noun_phrases


def __find_word_spans(text_lower, words):
    p = 0
    word_spans = list()
    for w in words:
        wp = text_lower[p:].find(w)
        if wp < 0:
            word_spans.append((-1, -1))
            continue
        word_spans.append((p + wp, p + wp + len(w)))
        p += wp + len(w)
    return word_spans


def get_noun_phrase_from_seed(dep_tags, pos_tags, base_word_idxs):
    words = [tup[2][1] for tup in dep_tags]
    phrase_word_idxs = set(base_word_idxs)

    ileft = min(phrase_word_idxs)
    iright = max(phrase_word_idxs)
    ileft_new, iright_new = ileft, iright
    while ileft_new > 0:
        if pos_tags[ileft_new - 1] in NOUN_POS_TAGS:
            ileft_new -= 1
        else:
            break
    while iright_new < len(pos_tags) - 1:
        if pos_tags[iright_new + 1] in {'NN', 'NNP', 'NNS', 'CD'}:
            iright_new += 1
        else:
            break

    phrase = ' '.join([words[widx] for widx in range(ileft_new, iright_new + 1)])
    return phrase


def pharse_for_span(span, sent_text_lower, words, pos_tags, dep_tags):
    word_spans = __find_word_spans(sent_text_lower, words)
    widxs = list()
    for i, wspan in enumerate(word_spans):
        if (wspan[0] <= span[0] < wspan[1]) or (wspan[0] < span[1] <= wspan[1]):
            widxs.append(i)

    if not widxs:
        # print(span)
        # print(sent_text_lower[span[0]: span[1]])
        # print(sent_text_lower)
        # print(words)
        # print(word_spans)
        # exit()
        return None

    phrase = get_noun_phrase_from_seed(dep_tags, pos_tags, widxs)
    return phrase


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


def find_terms_by_l2_pattern(pattern, dep_tags, pos_tags, mine_tool, filter_terms_vocab, existing_terms=None):
    (pl, ipl), (pr, ipr) = pattern
    terms = list()
    matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pl, dep_tags, pos_tags, mine_tool)
    for idx in matched_dep_tag_idxs:
        dep_tag_l = dep_tags[idx]
        sw_idx = dep_tag_l[ipl][0]  # index of the shared word

        for j, dep_tag_r in enumerate(dep_tags):
            if dep_tag_r[ipr][0] != sw_idx:
                continue
            if not __match_l1_pattern(pr, dep_tag_r, pos_tags, mine_tool):
                continue

            term = mine_tool.get_term_from_matched_pattern(pl, dep_tags, pos_tags, idx)
            if term is None:
                term = mine_tool.get_term_from_matched_pattern(pr, dep_tags, pos_tags, j)
            if term is None or term in filter_terms_vocab:
                # print(p, 'term not found')
                continue

            # if term not in existing_terms:
            #     print(pattern)
            #     print(term)
            #     print([t[2][1] for t in dep_tags])
            #     print(dep_tags)
            #     print()

            terms.append(term)
    return terms
