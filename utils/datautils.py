import json
import numpy as np


def get_indexed_word(w, dec=True):
    p = w.rfind('-')
    s = w[:p]
    idx = int(w[p + 1:])
    if dec:
        idx -= 1
    return s, idx


def read_sent_dep_tups_rbsep(fin):
    tups = list()
    for line in fin:
        line = line.strip()
        if not line:
            return tups
        line = line[:-1]
        line = line.replace('(', ' ')
        line = line.replace(', ', ' ')
        rel, gov, dep = line.split(' ')
        w_gov, idx_gov = get_indexed_word(gov, False)
        w_dep, idx_dep = get_indexed_word(dep, False)
        tups.append((rel, (idx_gov, w_gov), (idx_dep, w_dep)))
        # tups.append(line.split(' '))
    return tups


def next_sent_pos(fin):
    pos_tags = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        pos_tags.append(line)
    return pos_tags


def next_sent_dependency(fin):
    dep_list = list()
    for line in fin:
        line = line.strip()
        if not line:
            break
        dep_tup = line.split(' ')
        wgov, idx_gov = get_indexed_word(dep_tup[0])
        wdep, idx_dep = get_indexed_word(dep_tup[1])
        dep_tup = (dep_tup[2], (idx_gov, wgov), (idx_dep, wdep))
        dep_list.append(dep_tup)
    return dep_list


def load_dep_tags_list(filename, space_sep=True):
    f = open(filename, encoding='utf-8')
    sent_dep_tags_list = list()
    while True:
        if space_sep:
            dep_tags = next_sent_dependency(f)
        else:
            dep_tags = read_sent_dep_tups_rbsep(f)
        if not dep_tags:
            break
        sent_dep_tags_list.append(dep_tags)
    f.close()
    return sent_dep_tags_list


def load_pos_tags(filename):
    f = open(filename, encoding='utf-8')
    sent_pos_tags_list = list()
    while True:
        sent_pos_tags = next_sent_pos(f)
        if not sent_pos_tags:
            break
        sent_pos_tags_list.append(sent_pos_tags)
    f.close()
    return sent_pos_tags_list


def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def load_json_objs(filename):
    f = open(filename, encoding='utf-8')
    objs = list()
    for line in f:
        objs.append(json.loads(line))
    f.close()
    return objs


def write_terms_list(terms_list, dst_file):
    fout = open(dst_file, 'w', encoding='utf-8')
    for terms in terms_list:
        fout.write('{}\n'.format(json.dumps(terms, ensure_ascii=False)))
    fout.close()


def __add_unk_word(word_vecs_matrix):
    n_words = word_vecs_matrix.shape[0]
    dim = word_vecs_matrix.shape[1]
    word_vecs = np.zeros((n_words + 1, dim), np.float32)
    for i in range(n_words):
        word_vecs[i] = word_vecs_matrix[i]
    word_vecs[n_words] = np.random.normal(0, 0.1, dim)
    # word_vecs[n_words] = np.random.uniform(-0.1, 0.1, dim)
    return word_vecs


def load_word_vecs(word_vecs_file, add_unk=True):
    import pickle

    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    if add_unk and (not vocab[-1] == '<UNK>'):
        print('add <UNK>')
        word_vecs_matrix = __add_unk_word(word_vecs_matrix)
        vocab.append('<UNK>')

    assert vocab[-1] == '<UNK>'
    return vocab, word_vecs_matrix
