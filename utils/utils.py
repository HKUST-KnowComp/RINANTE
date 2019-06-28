import numpy as np


def get_machine_name():
    import socket
    hostname = socket.gethostname()
    dot_pos = hostname.find('.')
    return hostname[:dot_pos] if dot_pos > -1 else hostname[:]


def get_max_len(sequences):
    max_len = 0
    for s in sequences:
        max_len = max(max_len, len(s))
    return max_len


def pad_sequences(sequences, pad_token, fixed_len=False):
    max_len = get_max_len(sequences)

    padded_seqs, seq_lens = list(), list()
    for seq in sequences:
        padded_seq = seq + [pad_token for _ in range(max_len - len(seq))]
        padded_seqs.append(padded_seq)
        if fixed_len:
            seq_lens.append(max_len)
        else:
            seq_lens.append(len(seq))
    return padded_seqs, seq_lens


def recover_terms(text, word_spans, label_seq, label_beg, label_in):
    p = 0
    terms = list()
    while p < len(label_seq):
        if label_seq[p] == label_beg:
            pend = p + 1
            while pend < len(word_spans) and label_seq[pend] == label_in:
                pend += 1
            term_beg = word_spans[p][0]
            term_end = word_spans[pend - 1][1]
            # print(text[term_beg:term_end])
            terms.append(text[term_beg:term_end])
            p = pend
        else:
            p += 1
    return terms


def prf1(n_true, n_sys, n_hit):
    p = n_hit / (n_sys + 1e-6)
    r = n_hit / (n_true + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    return p, r, f1


def edit_dist(sl: str, sr: str):
    n, m = len(sl), len(sr)
    dists = np.zeros((n + 1, m + 1), np.int32)
    for j in range(m + 1):
        dists[0][j] = j
    for i in range(n + 1):
        dists[i][0] = i

    for i, cl in enumerate(sl):
        for j, cr in enumerate(sr):
            if cl == cr:
                dists[i + 1][j + 1] = dists[i][j]
                continue
            dists[i + 1][j + 1] = min(dists[i][j] + 1, dists[i][j + 1] + 1, dists[i + 1][j] + 1)
    return dists[n][m]


def count_hit(terms_true, terms_pred):
    terms_true, terms_pred = terms_true.copy(), terms_pred.copy()
    terms_true.sort()
    terms_pred.sort()
    idx_pred = 0
    cnt_hit = 0
    matched_true, matched_sys = [False] * len(terms_true), [False] * len(terms_pred)
    for i, t in enumerate(terms_true):
        while idx_pred < len(terms_pred) and terms_pred[idx_pred] < t:
            idx_pred += 1
        if idx_pred == len(terms_pred):
            continue
        if terms_pred[idx_pred] == t:
            matched_true[i] = True
            matched_sys[idx_pred] = True
            cnt_hit += 1
            idx_pred += 1

    # a few misspelled opinion terms are corrected
    for i, t in enumerate(terms_true):
        if matched_true[i] or len(t) < 3:
            continue
        for j in range(len(terms_pred)):
            if not matched_sys[j] and edit_dist(t, terms_pred[j]) < 2:
                # print(t, terms_pred[j])
                cnt_hit += 1

    return cnt_hit


def get_terms_from_label_list(labels, tok_text, label_beg, label_in):
    terms = list()
    words = tok_text.split(' ')
    # print(labels_pred)
    # print(len(words), len(labels_pred))
    assert len(words) == len(labels)

    p = 0
    while p < len(words):
        yi = labels[p]
        if yi == label_beg:
            pright = p
            while pright + 1 < len(words) and labels[pright + 1] == label_in:
                pright += 1
            terms.append(' '.join(words[p: pright + 1]))
            p = pright + 1
        else:
            p += 1
    return terms
