import numpy as np
from collections import namedtuple
from utils import datautils

TrainData = namedtuple("TrainData", ["labels_list", "word_idxs_list"])
ValidData = namedtuple("ValidData", ["texts", "labels_list", "word_idxs_list", "word_span_seqs", "tok_texts",
                                     "aspects_true_list", "opinions_true_list"])


def __label_words_with_terms_by_span(word_spans, term_spans, label_val_beg, label_val_in, x):
    for term_span in term_spans:
        is_first = True
        for i, wspan in enumerate(word_spans):
            if wspan[0] >= term_span[0] and wspan[1] <= term_span[1]:
                if is_first:
                    is_first = False
                    x[i] = label_val_beg
                else:
                    x[i] = label_val_in
            if wspan[0] > term_span[1]:
                break


def __find_sub_words_seq(words, sub_words):
    i, li, lj = 0, len(words), len(sub_words)
    while i + lj <= li:
        j = 0
        while j < lj:
            if words[i + j] != sub_words[j]:
                break
            j += 1
        if j == lj:
            return i
        i += 1
    return -1


def __label_words_with_terms(words, terms, label_val_beg, label_val_in, x):
    flg = True
    for term in terms:
        term_words = term.lower().split(' ')
        pbeg = __find_sub_words_seq(words, term_words)
        if pbeg == -1:
            # print(words, terms)
            flg = False
            continue
        x[pbeg] = label_val_beg
        for p in range(pbeg + 1, pbeg + len(term_words)):
            x[p] = label_val_in
    return flg


def label_sentence_by_span(words, word_spans, aspect_term_spans=None, opinion_terms=None):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    if aspect_term_spans is not None:
        __label_words_with_terms_by_span(word_spans, aspect_term_spans, label_val_beg, label_val_in, x)
        label_val_beg, label_val_in = 3, 4

    if opinion_terms is None:
        return x, True

    all_found = __label_words_with_terms(words, opinion_terms, label_val_beg, label_val_in, x)
    return x, all_found


def __get_word_idx_sequence(words_list, vocab):
    seq_list = list()
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    unk_id = len(vocab) - 1
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, unk_id) for w in words])
    return seq_list


def data_from_sents_file(sents, tok_texts, word_span_seqs, vocab, task, filter_non_complete=False):
    words_list = [text.split(' ') for text in tok_texts]
    len_max = max([len(words) for words in words_list])
    print('max sentence len:', len_max)

    labels_list = list()
    words_list_keep = list()
    for sent_idx, (sent, sent_words) in enumerate(zip(sents, words_list)):
        aspect_term_spans, aspect_terms, opinion_terms = None, None, None
        if task != 'opinion':
            aspect_objs = sent.get('terms', list())
            # aspect_terms = [t['term'] for t in aspect_objs]
            aspect_term_spans = [t['span'] for t in aspect_objs]

        if task != 'aspect':
            opinion_terms = sent.get('opinions', list())

        x, all_found = label_sentence_by_span(sent_words, word_span_seqs[sent_idx],
                                              aspect_term_spans, opinion_terms)
        if filter_non_complete and (not all_found):
            continue

        labels_list.append(x)
        words_list_keep.append(words_list[sent_idx])

    word_idxs_list = __get_word_idx_sequence(words_list_keep, vocab)
    return labels_list, word_idxs_list


def __get_valid_data(sents, tok_texts, word_span_seqs, vocab, task):
    labels_list_test, word_idxs_list_test = data_from_sents_file(sents, tok_texts, word_span_seqs, vocab, task)
    # exit()

    aspect_terms_true_list = list() if task != 'opinion' else None
    opinion_terms_true_list = list() if task != 'aspect' else None
    texts = list()
    for sent in sents:
        texts.append(sent['text'])
        if aspect_terms_true_list is not None:
            aspect_term_spans = set()
            cur_sent_aspect_terms = list()
            for t in sent.get('terms', list()):
                term_span = t['span']
                term_span = (term_span[0], term_span[1])
                if term_span not in aspect_term_spans:
                    aspect_term_spans.add(term_span)
                    cur_sent_aspect_terms.append(t['term'].lower())
            aspect_terms_true_list.append(cur_sent_aspect_terms)
        if opinion_terms_true_list is not None:
            opinion_terms_true_list.append([w.lower() for w in sent.get('opinions', list())])

    return ValidData(texts, labels_list_test, word_idxs_list_test, word_span_seqs, tok_texts, aspect_terms_true_list,
                     opinion_terms_true_list)


def load_token_pos_file(filename):
    tok_texts, tok_span_seqs = list(), list()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            tok_texts.append(line.strip())
            tok_spans_str = next(f).strip()
            vals = [int(v) for v in tok_spans_str.split(' ')]
            tok_span_seqs.append([(vals[2 * i], vals[2 * i + 1]) for i in range(len(vals) // 2)])
    return tok_texts, tok_span_seqs


def get_data_semeval(train_sents_file, train_tok_text_file, train_valid_split_file, test_sents_file,
                     test_tok_text_file, vocab, n_train, task):
    tvs_line = datautils.read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    sents = datautils.load_json_objs(train_sents_file)
    # texts = utils.read_lines(train_tok_text_file)
    tok_texts, word_span_seqs = load_token_pos_file(train_tok_text_file)

    sents_train, tok_texts_train, sents_valid, tok_texts_valid = list(), list(), list(), list()
    word_span_seqs_train, word_span_seqs_valid = list(), list()
    for label, s, t, span_seq in zip(tvs_arr, sents, tok_texts, word_span_seqs):
        if label == 0:
            sents_train.append(s)
            tok_texts_train.append(t)
            word_span_seqs_train.append(span_seq)
        else:
            sents_valid.append(s)
            tok_texts_valid.append(t)
            word_span_seqs_valid.append(span_seq)

    labels_list_train, word_idxs_list_train = data_from_sents_file(
        sents_train, tok_texts_train, word_span_seqs_train, vocab, task)
    if n_train > -1:
        labels_list_train = labels_list_train[:n_train]
        word_idxs_list_train = word_idxs_list_train[:n_train]

    train_data = TrainData(labels_list_train, word_idxs_list_train)

    valid_data = __get_valid_data(sents_valid, tok_texts_valid, word_span_seqs_valid, vocab, task)

    sents_test = datautils.load_json_objs(test_sents_file)
    texts_test, word_span_seqs_test = load_token_pos_file(test_tok_text_file)
    print('get test')
    test_data = __get_valid_data(sents_test, texts_test, word_span_seqs_test, vocab, task)
    return train_data, valid_data, test_data


def label_sentence(words, aspect_terms=None, opinion_terms=None):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    if aspect_terms is not None:
        __label_words_with_terms(words, aspect_terms, label_val_beg, label_val_in, x)
        label_val_beg, label_val_in = 3, 4

    if opinion_terms is None:
        return x

    __label_words_with_terms(words, opinion_terms, label_val_beg, label_val_in, x)
    return x


def get_weak_label_data(vocab, true_terms_file, tok_texts_file, task):
    terms_true_list = datautils.load_json_objs(true_terms_file)
    tok_texts = datautils.read_lines(tok_texts_file)
    # print(len(terms_true_list), tok_texts_file, len(tok_texts))
    if len(terms_true_list) != len(tok_texts):
        print(len(terms_true_list), len(tok_texts))
    assert len(terms_true_list) == len(tok_texts)

    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}

    label_seq_list = list()
    word_idx_seq_list = list()
    for terms_true, tok_text in zip(terms_true_list, tok_texts):
        words = tok_text.split(' ')
        label_seq = label_sentence(words, terms_true)
        label_seq_list.append(label_seq)
        word_idx_seq_list.append([word_idx_dict.get(w, 0) for w in words])

    np.random.seed(3719)
    perm = np.random.permutation(len(label_seq_list))
    n_train = len(label_seq_list) - 2000
    idxs_train, idxs_valid = perm[:n_train], perm[n_train:]

    label_seq_list_train = [label_seq_list[idx] for idx in idxs_train]
    word_idx_seq_list_train = [word_idx_seq_list[idx] for idx in idxs_train]
    train_data = TrainData(label_seq_list_train, word_idx_seq_list_train)

    label_seq_list_valid = [label_seq_list[idx] for idx in idxs_valid]
    word_idx_seq_list_valid = [word_idx_seq_list[idx] for idx in idxs_valid]
    tok_texts_valid = [tok_texts[idx] for idx in idxs_valid]
    terms_true_list_valid = [terms_true_list[idx] for idx in idxs_valid]
    aspect_true_list, opinion_true_list = None, None
    if task != 'opinion':
        aspect_true_list = terms_true_list_valid
    if task != 'aspect':
        opinion_true_list = terms_true_list_valid
    valid_data = ValidData(
        None, label_seq_list_valid, word_idx_seq_list_valid, None, tok_texts_valid,
        aspect_true_list, opinion_true_list)

    return train_data, valid_data
