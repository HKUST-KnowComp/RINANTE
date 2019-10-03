from rule import ruleutils


class OpinionMineTool:
    def __init__(self):
        pass

    def get_patterns_from_term(self, term_word_idx_span, related_dep_tag_idxs, dep_tags, pos_tags):
        widx_beg, widx_end = term_word_idx_span
        term_pos_tags = set([pos_tags[i] for i in range(widx_beg, widx_end)])
        term_pos_type = OpinionMineTool.__get_term_pos_type(term_pos_tags)
        if term_pos_type is None:
            # print(term)
            return set(), set()

        opinion_word_wc = '_O{}'.format(term_pos_type)

        related_dep_tags = [dep_tags[idx] for idx in related_dep_tag_idxs]
        patterns_l1 = self.__patterns_from_l1_dep_tags(opinion_word_wc, related_dep_tags, pos_tags, term_word_idx_span)
        related_l2 = ruleutils.find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags)
        patterns_l2 = self.__patterns_from_l2_dep_tags(opinion_word_wc, related_l2, pos_tags, term_word_idx_span)
        return patterns_l1, patterns_l2

    @staticmethod
    def get_candidate_terms(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        pos_tag_sets = [ruleutils.JJ_POS_TAGS, ruleutils.RB_POS_TAGS, ruleutils.VB_POS_TAGS]
        terms = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            for pos_tag_set in pos_tag_sets:
                if pos_tag in pos_tag_set:
                    terms.append(w)
        return terms

    @staticmethod
    def get_term_from_matched_pattern(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_O'):
            aspect_position = 1
        elif pattern[2].startswith('_O'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        return w

    @staticmethod
    def match_pattern_word(pw, w, pos_tag):
        if pw == '_OJ' and pos_tag in ruleutils.JJ_POS_TAGS:
            return True
        if pw == '_OR' and pos_tag in ruleutils.RB_POS_TAGS:
            return True
        if pw == '_OV' and pos_tag in ruleutils.VB_POS_TAGS:
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w

    @staticmethod
    def get_terms_by_matching(dep_tags, pos_tags, sent_text, terms_vocab):
        terms = list()
        for t in terms_vocab:
            pbeg = sent_text.find(t)
            if pbeg < 0:
                continue
            pend = pbeg + len(t)
            if pbeg > 0 and sent_text[pbeg - 1].isalpha():
                continue
            if pend < len(sent_text) and sent_text[pend].isalpha():
                continue
            terms.append(t)
        return terms

    @staticmethod
    def terms_list_from_sents(sents):
        opinion_terms_list = list()
        for sent in sents:
            opinion_terms_list.append([t.lower() for t in sent.get('opinions', list())])
        return opinion_terms_list

    @staticmethod
    def __patterns_from_l1_dep_tags(opinion_word_wc, related_dep_tags, pos_tags, term_word_idx_span):
        widx_beg, widx_end = term_word_idx_span
        # print(related_dep_tags)
        patterns = set()
        for dep_tag in related_dep_tags:
            rel, (igov, wgov), (idep, wdep) = dep_tag
            if widx_beg <= igov < widx_end:
                patterns.add((rel, opinion_word_wc, wdep))
                patterns.add((rel, opinion_word_wc, pos_tags[idep]))
            elif widx_beg <= idep < widx_end:
                patterns.add((rel, wgov, opinion_word_wc))
                patterns.add((rel, pos_tags[igov], opinion_word_wc))
            else:
                patterns.add((rel, wgov, wdep))
                patterns.add((rel, pos_tags[igov], wdep))
                patterns.add((rel, wgov, pos_tags[idep]))
        return patterns

    # TODO common
    def __patterns_from_l2_dep_tags(self, opinion_word_wc, related_dep_tag_tups, pos_tags, term_word_idx_span):
        # widx_beg, widx_end = term_word_idx_span
        patterns = set()
        for dep_tag_i, dep_tag_j in related_dep_tag_tups:
            patterns_i = self.__patterns_from_l1_dep_tags(
                opinion_word_wc, [dep_tag_i], pos_tags, term_word_idx_span)
            patterns_j = self.__patterns_from_l1_dep_tags(
                opinion_word_wc, [dep_tag_j], pos_tags, term_word_idx_span)
            # print(dep_tag_i, dep_tag_j)
            # print(patterns_i, patterns_j)

            if dep_tag_i[1][0] == dep_tag_j[1][0] or dep_tag_i[1][0] == dep_tag_j[2][0]:
                patterns_i = {(tup, 1) for tup in patterns_i}
            else:
                patterns_i = {(tup, 2) for tup in patterns_i}

            if dep_tag_j[1][0] == dep_tag_i[1][0] or dep_tag_j[1][0] == dep_tag_i[2][0]:
                patterns_j = {(tup, 1) for tup in patterns_j}
            else:
                patterns_j = {(tup, 2) for tup in patterns_j}
            # print(patterns_i, patterns_j)

            for pi in patterns_i:
                for pj in patterns_j:
                    if pi[0][pi[1]] != pj[0][pj[1]]:
                        # print(pi, pj)
                        continue
                    if pi < pj:
                        patterns.add((pi, pj))
                    else:
                        patterns.add((pj, pi))
        return patterns

    @staticmethod
    def __get_term_pos_type(term_pos_tags):
        for t in term_pos_tags:
            if t in ruleutils.JJ_POS_TAGS:
                return 'J'
        for t in term_pos_tags:
            if t in ruleutils.RB_POS_TAGS:
                return 'R'
        for t in term_pos_tags:
            if t in ruleutils.VB_POS_TAGS:
                return 'V'
        return None
