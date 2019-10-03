from os.path import join
from platform import platform

DATA_DIR = 'rinante-data/'

SE14_DIR = join(DATA_DIR, 'semeval14')
SE15_DIR = join(DATA_DIR, 'semeval15')

SE14L_FILES = {
    'train_sents_file': join(SE14_DIR, 'laptops/laptops_train_sents.json'),
    'train_tok_texts_file': join(SE14_DIR, 'laptops/laptops_train_texts_tok_pos.txt'),
    'train_valid_split_file': join(SE14_DIR, 'laptops/laptops_train_valid_split.txt'),
    'test_sents_file': join(SE14_DIR, 'laptops/laptops_test_sents.json'),
    'test_tok_texts_file': join(SE14_DIR, 'laptops/laptops_test_texts_tok_pos.txt'),
    'train_dep_tags_file': join(SE14_DIR, 'laptops/laptops-train-rule-dep.txt'),
    'train_pos_tags_file': join(SE14_DIR, 'laptops/laptops-train-rule-pos.txt'),
    'rule_patterns_file': join(SE14_DIR, 'laptops/se14-rule-patterns.txt'),
    'word_cnts_file': join(SE14_DIR, 'laptops/word_cnts.txt'),
}

SE14R_FILES = {
    'train_sents_file': join(SE14_DIR, 'restaurants/restaurants_train_sents.json'),
    'train_tok_texts_file': join(SE14_DIR, 'restaurants/restaurants_train_texts_tok_pos.txt'),
    'train_valid_split_file': join(SE14_DIR, 'restaurants/restaurants_train_valid_split.txt'),
    'test_sents_file': join(SE14_DIR, 'restaurants/restaurants_test_sents.json'),
    'test_tok_texts_file': join(SE14_DIR, 'restaurants/restaurants_test_texts_tok_pos.txt'),
}

SE15R_FILES = {
    'train_sents_file': join(SE15_DIR, 'restaurants/restaurants_train_sents.json'),
    'train_tok_texts_file': join(SE15_DIR, 'restaurants/restaurants_train_texts_tok_pos.txt'),
    'train_valid_split_file': join(SE15_DIR, 'restaurants/restaurants_train_valid_split.txt'),
    'test_sents_file': join(SE15_DIR, 'restaurants/restaurants_test_sents.json'),
    'test_tok_texts_file': join(SE15_DIR, 'restaurants/restaurants_test_texts_tok_pos.txt'),
}

DATA_DICT = {
    'se14l': SE14L_FILES,
    'se14r': SE14R_FILES,
    'se15r': SE15R_FILES,
}
