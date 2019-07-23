import os
import config
import tensorflow as tf
import datetime
from models.rinante import RINANTE
from models import modelutils
from utils import utils, datautils
from utils.loggingutils import init_logging
import logging


def __train(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
            test_sents_file, load_model_file, task):
    init_logging('log/{}-train-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    dst_aspects_file, dst_opinions_file = None, None

    # n_train = 1000
    n_train = -1
    n_tags = 5
    batch_size = 64
    lr = 0.001
    share_lstm = False

    logging.info(word_vecs_file)
    logging.info('load model {}'.format(load_model_file))
    logging.info(test_sents_file)

    print('loading data ...')
    vocab, word_vecs_matrix = datautils.load_word_vecs(word_vecs_file)

    logging.info('word vec dim: {}, n_words={}'.format(word_vecs_matrix.shape[1], word_vecs_matrix.shape[0]))
    train_data, valid_data, test_data = modelutils.get_data_semeval(
        train_sents_file, train_tok_texts_file, train_valid_split_file, test_sents_file, test_tok_texts_file,
        vocab, n_train, task)
    print('done')

    test_f1s = list()
    for i in range(5):
        logging.info('turn {}'.format(i))
        model = RINANTE(n_tags, word_vecs_matrix, share_lstm, hidden_size_lstm=hidden_size_lstm,
                        model_file=load_model_file, batch_size=batch_size, lamb=lamb)
        test_af1, _ = model.train(train_data, valid_data, test_data, vocab, n_epochs=n_epochs, lr=lr,
                                  dst_aspects_file=dst_aspects_file, dst_opinions_file=dst_opinions_file)
        test_f1s.append(test_af1)
        logging.info('r={} test_f1={:.4f}'.format(i, test_af1))
        tf.reset_default_graph()
    logging.info('avg_test_f1={:.4f}'.format(sum(test_f1s) / len(test_f1s)))


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')

    hidden_size_lstm = 100
    n_epochs = 170
    train_word_embeddings = False

    # dataset = 'se15r'
    # dataset = 'se14r'
    dataset = 'se14l'
    dataset_files = config.DATA_DICT[dataset]

    lamb = 0.001
    lstm_l2_src = False

    if dataset == 'se15r':
        rule_model_file = os.path.join(config.DATA_DIR, 'model-data/pretrain/yelpr9-rest-se15r.ckpt')
        word_vecs_file = os.path.join(config.DATA_DIR, 'model-data/yelp-w2v-sg-se15r.pkl')
        train_valid_split_file = os.path.join(config.SE15_DIR, 'restaurants/se15r_train_valid_split-150.txt')
    elif dataset == 'se14r':
        rule_model_file = os.path.join(config.DATA_DIR, 'model-data/pretrain/yelpr9-rest-se14r.ckpt')
        word_vecs_file = os.path.join(config.DATA_DIR, 'model-data/yelp-w2v-sg-se14r.pkl')
        train_valid_split_file = os.path.join(config.SE14_DIR, 'restaurants/se14r_train_valid_split-150.txt')
    else:
        rule_model_file = os.path.join(config.DATA_DIR, 'model-data/pretrain/laptops-amazon-se14l.ckpt')
        word_vecs_file = os.path.join(config.DATA_DIR, 'model-data/laptops-amazon-w2v-se14l.pkl')
        train_valid_split_file = os.path.join(config.SE14_DIR, 'laptops/se14l_train_valid_split-150.txt')

    __train(word_vecs_file, dataset_files['train_tok_texts_file'], dataset_files['train_sents_file'],
            train_valid_split_file, dataset_files['test_tok_texts_file'],
            dataset_files['test_sents_file'], rule_model_file, 'aspect')
