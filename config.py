from os.path import join
from platform import platform

DATA_DIR = 'd:/data/rinante-data/'

SE14_DIR = join(DATA_DIR, 'semeval14')
SE15_DIR = join(DATA_DIR, 'semeval15')

SE14L_FILES = {
    'train_sents_file': join(SE14_DIR, 'laptops/laptops_train_sents.json'),
    'train_tok_texts_file': join(SE14_DIR, 'laptops/laptops_train_texts_tok_pos.txt'),
    'train_valid_split_file': join(SE14_DIR, 'laptops/laptops_train_valid_split.txt'),
    'test_sents_file': join(SE14_DIR, 'laptops/laptops_test_sents.json'),
    'test_tok_texts_file': join(SE14_DIR, 'laptops/laptops_test_texts_tok_pos.txt'),
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
