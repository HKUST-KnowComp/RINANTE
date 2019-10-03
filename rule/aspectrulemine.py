from rule import ruleutils
from utils import datautils


class AspectMineTool:
    def __init__(self, opinion_terms_vocab_file):
        self.opinion_terms_vocab = set(datautils.read_lines(opinion_terms_vocab_file))

    def get_patterns_from_term(self, term_word_idx_span, related_dep_tag_idxs, dep_tags, pos_tags):
        widx_beg, widx_end = term_word_idx_span
        term_pos_tags = set([pos_tags[i] for i in range(widx_beg, widx_end)])
        term_pos_type = AspectMineTool.__get_term_pos_type(term_pos_tags)
        if term_pos_type is None:
            # print(term)
            return set(), set()

        aspect_word_wc = '_A{}'.format(term_pos_type)

        related_dep_tags = [dep_tags[idx] for idx in related_dep_tag_idxs]
        patterns_l1 = self.__patterns_from_l1_dep_tags(aspect_word_wc, related_dep_tags, pos_tags, term_word_idx_span)
        related_l2 = ruleutils.find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags)
        patterns_l2 = self.__patterns_from_l2_dep_tags(aspect_word_wc, related_l2, pos_tags, term_word_idx_span)
        return patterns_l1, patterns_l2

    @staticmethod
    def get_candidate_terms(dep_tag_seq, pos_tag_seq):
        words = [tup[2][1] for tup in dep_tag_seq]
        noun_phrases = ruleutils.get_noun_phrases(words, pos_tag_seq, None)

        verbs = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            if pos_tag in ruleutils.VB_POS_TAGS:
                verbs.append(w)

        return noun_phrases + verbs

    @staticmethod
    def get_term_from_matched_pattern(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
        if pattern[1].startswith('_A'):
            aspect_position = 1
        elif pattern[2].startswith('_A'):
            aspect_position = 2
        else:
            return None

        dep_tag = dep_tags[matched_dep_tag_idx]
        widx, w = dep_tag[aspect_position]
        if pattern[aspect_position] == '_AV':
            return w
        else:
            return ruleutils.get_noun_phrase_from_seed(dep_tags, pos_tags, [widx])

    def match_pattern_word(self, pw, w, pos_tag):
        if pw == '_AV' and pos_tag in ruleutils.VB_POS_TAGS:
            return True
        if pw == '_AN' and pos_tag in ruleutils.NOUN_POS_TAGS:
            return True
        if pw == '_OP' and w in self.opinion_terms_vocab:
            return True
        if pw.isupper() and pos_tag == pw:
            return True
        return pw == w

    @staticmethod
    def get_terms_by_matching(dep_tags, pos_tags, sent_text, terms_vocab):
        sent_text_lower = sent_text.lower()
        matched_tups = list()
        for t in terms_vocab:
            pbeg = sent_text_lower.find(t)
            if pbeg < 0:
                continue
            if pbeg != 0 and sent_text_lower[pbeg - 1].isalpha():
                continue
            pend = pbeg + len(t)
            if pend != len(sent_text_lower) and sent_text_lower[pend].isalpha():
                continue
            matched_tups.append((pbeg, pend))
            # break

        matched_tups = AspectMineTool.__remove_embeded(matched_tups)
        sent_words = [tup[2][1] for tup in dep_tags]
        aspect_terms = set()
        for matched_span in matched_tups:
            phrase = ruleutils.pharse_for_span(matched_span, sent_text_lower, sent_words, pos_tags, dep_tags)
            if phrase is not None:
                aspect_terms.add(phrase)

        return aspect_terms

    @staticmethod
    def terms_list_from_sents(sents):
        aspect_terms_list = list()
        for sent in sents:
            aspect_terms_list.append([t['term'].lower() for t in sent.get('terms', list())])
        return aspect_terms_list

    @staticmethod
    def __remove_embeded(matched_tups):
        matched_tups_new = list()
        for i, t0 in enumerate(matched_tups):
            exist = False
            for j, t1 in enumerate(matched_tups):
                if i != j and t1[0] <= t0[0] and t1[1] >= t0[1]:
                    exist = True
                    break
            if not exist:
                matched_tups_new.append(t0)
        return matched_tups_new

    def __patterns_from_l1_dep_tags(self, aspect_word_wc, related_dep_tags, pos_tags, term_word_idx_span):
        widx_beg, widx_end = term_word_idx_span
        # print(related_dep_tags)
        patterns = set()
        for dep_tag in related_dep_tags:
            rel, gov, dep = dep_tag
            igov, wgov = gov
            idep, wdep = dep
            if widx_beg <= igov < widx_end:
                patterns.add((rel, aspect_word_wc, wdep))
                patterns.add((rel, aspect_word_wc, pos_tags[idep]))
                if wdep in self.opinion_terms_vocab:
                    patterns.add((rel, aspect_word_wc, '_OP'))
            elif widx_beg <= idep < widx_end:
                patterns.add((rel, wgov, aspect_word_wc))
                patterns.add((rel, pos_tags[igov], aspect_word_wc))
                if wgov in self.opinion_terms_vocab:
                    patterns.add((rel, '_OP', aspect_word_wc))
            else:
                patterns.add((rel, wgov, wdep))
                patterns.add((rel, pos_tags[igov], wdep))
                patterns.add((rel, wgov, pos_tags[idep]))
                if wgov in self.opinion_terms_vocab:
                    patterns.add((rel, '_OP', wdep))
                    patterns.add((rel, '_OP', pos_tags[idep]))
                if wdep in self.opinion_terms_vocab:
                    patterns.add((rel, wgov, '_OP'))
                    patterns.add((rel, pos_tags[igov], '_OP'))
        return patterns

    def __patterns_from_l2_dep_tags(self, aspect_word_wc, related_dep_tag_tups, pos_tags, term_word_idx_span):
        # widx_beg, widx_end = term_word_idx_span
        patterns = set()
        for dep_tag_i, dep_tag_j in related_dep_tag_tups:
            patterns_i = self.__patterns_from_l1_dep_tags(
                aspect_word_wc, [dep_tag_i], pos_tags, term_word_idx_span)
            patterns_j = self.__patterns_from_l1_dep_tags(
                aspect_word_wc, [dep_tag_j], pos_tags, term_word_idx_span)
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
            if t in ruleutils.NOUN_POS_TAGS:
                return 'N'
        for t in term_pos_tags:
            if t in ruleutils.VB_POS_TAGS:
                return 'V'
        return None
