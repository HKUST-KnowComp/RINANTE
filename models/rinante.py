import logging
import tensorflow as tf
from utils import utils, datautils
from models.modelutils import TrainData, ValidData


class RINANTE:
    def __init__(self, n_tags, word_embeddings, share_lstm, hidden_size_lstm=100, train_word_embeddings=False,
                 batch_size=20, lr_method='adam', clip=-1, lamb=0.001, lstm_l2_src=False, lstm_l2_tar=False,
                 use_crf=True, model_file=None):
        logging.info('hidden_size_lstm={} batch_size={} lr_method={} lamb={} lstm_l2_src={} lstm_l2_tar={}'.format(
            hidden_size_lstm, batch_size, lr_method, lamb, lstm_l2_src, lstm_l2_tar))

        self.n_tags_src = 3
        self.n_tags = n_tags
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.clip = clip
        self.saver = None
        self.share_lstm = share_lstm
        self.lamb = lamb
        self.lstm_l2_src = lstm_l2_src
        self.lstm_l2_tar = lstm_l2_tar

        # self.n_words, self.dim_word = word_embeddings.shape
        self.use_crf = use_crf

        self.word_idxs = tf.placeholder(tf.int32, shape=[None, None], name='word_idxs')
        # self.rule_hidden_input = tf.placeholder(tf.float32, shape=[None, None, hidden_size_lstm * 2],
        #                                         name='rule_hidden_input')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels_src1 = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.labels_src2 = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.labels_tar = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.is_train = tf.placeholder(tf.bool, name="is_training")

        self.__add_word_embedding_op(word_embeddings, train_word_embeddings)
        # self.__setup_rule_lstm_hidden()
        self.__add_logits_op()
        self.__add_pred_op()
        self.__add_loss_op()
        self.__add_train_op(self.lr_method, self.lr, self.clip)
        self.__init_session(model_file)

    def __init_session(self, model_file):
        self.sess = tf.Session()
        if model_file is None:
            # self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            # self.saver.restore(self.sess, rule_model_file)
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def __add_word_embedding_op(self, word_embeddings_val, train_word_embeddings):
        with tf.variable_scope("words"):
            # _word_embeddings = tf.constant(
            #         word_embeddings_val,
            #         name="_word_embeddings",
            #         dtype=tf.float32)
            self._word_embeddings_var = tf.Variable(
                word_embeddings_val,
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=train_word_embeddings)

            word_embeddings = tf.nn.embedding_lookup(self._word_embeddings_var, self.word_idxs, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, keep_prob=self.dropout)
        # self.word_embeddings = word_embeddings

    def __add_logits_op(self):
        self.lstm_cells = list()
        if self.share_lstm:
            with tf.variable_scope("bi-lstm"):
                cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.word_embeddings,
                        sequence_length=self.sequence_lengths, dtype=tf.float32)
                self.lstm_cells += [cell_fw, cell_bw]
                self.lstm_output = tf.concat([output_fw, output_bw], axis=-1)
                self.lstm_output = tf.nn.dropout(self.lstm_output, keep_prob=self.dropout)
                lstm_output1 = lstm_output2 = self.lstm_output
        else:
            with tf.variable_scope("bi-lstm-1"):
                self.cell_fw1 = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        self.cell_fw1, cell_bw, self.word_embeddings,
                        sequence_length=self.sequence_lengths, dtype=tf.float32)
                self.lstm_cells += [self.cell_fw1, cell_bw]
                self.lstm_output1 = tf.concat([output_fw, output_bw], axis=-1)
                self.lstm_output1 = tf.nn.dropout(self.lstm_output1, keep_prob=self.dropout)
                lstm_output1 = self.lstm_output1

            with tf.variable_scope("bi-lstm-2"):
                cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.word_embeddings,
                        sequence_length=self.sequence_lengths, dtype=tf.float32)
                self.lstm_cells += [cell_fw, cell_bw]
                self.lstm_output2 = tf.concat([output_fw, output_bw], axis=-1)
                self.lstm_output2 = tf.nn.dropout(self.lstm_output2, keep_prob=self.dropout)
                lstm_output2 = self.lstm_output2

        with tf.variable_scope("proj-src1"):
            self.W_src1 = tf.get_variable("W", dtype=tf.float32, shape=[
                2 * self.hidden_size_lstm, self.n_tags_src])
            self.b_src1 = tf.get_variable(
                "b", shape=[self.n_tags_src], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output1)[1]
            output = tf.reshape(lstm_output1, [-1, 2 * self.hidden_size_lstm])
            # pred = tf.matmul(output, self.W_src1) + self.b_src1
            pred = tf.add(tf.matmul(output, self.W_src1), self.b_src1)
            self.logits_src1 = tf.reshape(pred, [-1, nsteps, self.n_tags_src])

        with tf.variable_scope("proj-src2"):
            self.W_src2 = tf.get_variable("W", dtype=tf.float32, shape=[
                2 * self.hidden_size_lstm, 3])
            self.b_src2 = tf.get_variable(
                "b", shape=[self.n_tags_src], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output2)[1]
            output = tf.reshape(lstm_output2, [-1, 2 * self.hidden_size_lstm])
            # pred = tf.matmul(output, self.W_src2) + self.b_src2
            pred = tf.add(tf.matmul(output, self.W_src2), self.b_src2)
            self.logits_src2 = tf.reshape(pred, [-1, nsteps, self.n_tags_src])

        with tf.variable_scope("proj-target"):
            input_size = 2 * self.hidden_size_lstm if self.share_lstm else 4 * self.hidden_size_lstm

            self.W_tar = tf.get_variable("W", dtype=tf.float32, shape=[input_size, self.n_tags])
            self.b_tar = tf.get_variable(
                "b", shape=[self.n_tags], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output1)[1]

            if self.share_lstm:
                output = lstm_output1
            else:
                output = tf.concat([lstm_output1, lstm_output2], axis=-1)
            output = tf.reshape(output, [-1, input_size])
            # pred = tf.matmul(output, self.W_tar) + self.b_tar
            pred = tf.add(tf.matmul(output, self.W_tar), self.b_tar)
            # hid = tf.matmul(output, self.W_tar1) + self.b_tar1
            # hid = tf.layers.batch_normalization(hid, training=self.is_train)
            # hid = tf.nn.dropout(hid, self.dropout)
            # pred = tf.matmul(hid, self.W_tar2) + self.b_tar2
            self.logits_tar = tf.reshape(pred, [-1, nsteps, self.n_tags])

    def __add_pred_op(self):
        if not self.use_crf:
            self.labels_pred_src1 = tf.cast(tf.argmax(self.logits_src1, axis=-1), tf.int32)
            self.labels_pred_src2 = tf.cast(tf.argmax(self.logits_src2, axis=-1), tf.int32)
            self.labels_pred_tar = tf.cast(tf.argmax(self.logits_tar, axis=-1), tf.int32)

    def __add_loss_op(self):
        """Defines the loss"""
        lstm_l2_reg = 0
        if self.lstm_l2_src or self.lstm_l2_tar:
            for lstm_cell in self.lstm_cells:
                lstm_l2_reg += self.lamb * tf.nn.l2_loss(lstm_cell.trainable_variables[0])

        if self.use_crf:
            with tf.variable_scope("crf-src1"):
                log_likelihood, self.trans_params_src1 = tf.contrib.crf.crf_log_likelihood(
                        self.logits_src1, self.labels_src1, self.sequence_lengths)
                self.loss_src1 = tf.reduce_mean(-log_likelihood)
                if self.lstm_l2_src:
                    self.loss_src1 += lstm_l2_reg

            with tf.variable_scope("crf-src2"):
                log_likelihood, self.trans_params_src2 = tf.contrib.crf.crf_log_likelihood(
                        self.logits_src2, self.labels_src2, self.sequence_lengths)
                self.loss_src2 = tf.reduce_mean(-log_likelihood)
                if self.lstm_l2_src:
                    self.loss_src2 += lstm_l2_reg

            with tf.variable_scope("crf-tar"):
                log_likelihood, self.trans_params_tar = tf.contrib.crf.crf_log_likelihood(
                        self.logits_tar, self.labels_tar, self.sequence_lengths)
                self.loss_tar = tf.reduce_mean(-log_likelihood) + self.lamb * tf.nn.l2_loss(
                    self.W_tar)
                # self.loss_tar = tf.reduce_mean(-log_likelihood)
                if self.lstm_l2_tar:
                    self.loss_tar += lstm_l2_reg
        else:
            assert False

    def __add_train_op(self, lr_method, lr, clip=-1):
        """Defines self.train_op that performs an update on a batch"""
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.loss_src1))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op_src1 = optimizer.apply_gradients(zip(grads, vs))

                grads, vs = zip(*optimizer.compute_gradients(self.loss_src2))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op_src2 = optimizer.apply_gradients(zip(grads, vs))

                grads, vs = zip(*optimizer.compute_gradients(self.loss_tar))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op_tar = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op_src1 = optimizer.minimize(self.loss_src1)
                self.train_op_src2 = optimizer.minimize(self.loss_src2)
                self.train_op_tar = optimizer.minimize(self.loss_tar)

    def pretrain(self, data_train_s1: TrainData, data_valid_s1: ValidData, data_train_s2: TrainData,
                 data_valid_s2: ValidData, vocab, n_epochs, lr=0.001, dropout=0.5, save_file=None):
        logging.info('pretrain, n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train_src1 = len(data_train_s1.word_idxs_list)
        n_batches_src1 = (n_train_src1 + self.batch_size - 1) // self.batch_size

        n_train_src2 = len(data_train_s2.word_idxs_list)
        n_batches_src2 = (n_train_src2 + self.batch_size - 1) // self.batch_size

        best_f1 = 0
        best_f11, best_f12 = 0, 0
        batch_idx_src1, batch_idx_src2 = 0, 0
        for epoch in range(n_epochs):
            # losses_src, losses_seg_src = list(), list()
            for i in range(n_batches_src1):
                train_loss_src1 = self.__train_batch(
                    data_train_s1.word_idxs_list, data_train_s1.labels_list, batch_idx_src1, lr, dropout, 'src1')
                batch_idx_src1 = batch_idx_src1 + 1 if batch_idx_src1 + 1 < n_batches_src1 else 0
                train_loss_src2 = self.__train_batch(
                    data_train_s2.word_idxs_list, data_train_s2.labels_list, batch_idx_src2, lr, dropout, 'src2')
                batch_idx_src2 = batch_idx_src2 + 1 if batch_idx_src2 + 1 < n_batches_src2 else 0

                if (i + 1) % 100 == 0:
                    p1, r1, f11, _, _, _ = self.evaluate(
                        data_valid_s1.texts, data_valid_s1.word_idxs_list, data_valid_s1.word_span_seqs,
                        data_valid_s1.tok_texts, data_valid_s1.aspects_true_list, 'src1')

                    p2, r2, f12, _, _, _ = self.evaluate(
                        data_valid_s2.texts, data_valid_s2.word_idxs_list, data_valid_s2.word_span_seqs,
                        data_valid_s2.tok_texts, data_valid_s2.opinions_true_list, 'src2')

                    logging.info('it={}, p={:.4f}, r={:.4f}, f1={:.4f}; src2, p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                        epoch, p1, r1, f11, p2, r2, f12
                    ))

                    if f11 + f12 > best_f1:
                        best_f1 = f11 + f12
                    # if f11 >= best_f11 and f12 >= best_f12:
                    #     best_f11 = f11
                    #     best_f12 = f12
                        if self.saver is not None:
                            self.saver.save(self.sess, save_file)
                            # print('model saved to {}'.format(save_file))
                            logging.info('model saved to {}'.format(save_file))

    def train(self, data_train: TrainData, data_valid: ValidData, data_test: ValidData, vocab,
              n_epochs=10, lr=0.001, dropout=0.5, save_file=None, dst_aspects_file=None, dst_opinions_file=None):
        # embs = self.sess.run(self._word_embeddings_var)
        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train = len(data_train.word_idxs_list)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        best_f1_sum = 0
        best_a_f1, best_o_f1 = 0, 0
        best_test_af1, best_test_of1 = 0, 0
        for epoch in range(n_epochs):
            losses, losses_seg = list(), list()
            for i in range(n_batches):
                train_loss_tar = self.__train_batch(
                    data_train.word_idxs_list, data_train.labels_list, i, lr, dropout, 'tar')
                losses.append(train_loss_tar)

            # if not (epoch + 1) % 5 == 0:
            #     continue

            loss = sum(losses)
            # metrics = self.run_evaluate(dev)
            aspect_p, aspect_r, dev_aspect_f1, opinion_p, opinion_r, dev_opinion_f1 = self.evaluate(
                data_valid.texts, data_valid.word_idxs_list, data_valid.word_span_seqs, data_valid.tok_texts,
                data_valid.aspects_true_list, 'tar', data_valid.opinions_true_list)

            cur_f1_sum = dev_aspect_f1 + dev_opinion_f1
            best_tag = '*' if cur_f1_sum >= best_f1_sum else ''
            logging.info('iter {}, loss={:.4f} p={:.4f} r={:.4f} f1={:.4f}'
                         ' p={:.4f} r={:.4f} f1={:.4f} f1s={:.4f} f1s*={:.4f}{}'.format(
                epoch, loss, aspect_p, aspect_r, dev_aspect_f1, opinion_p, opinion_r,
                dev_opinion_f1, dev_aspect_f1 + dev_opinion_f1, best_f1_sum, best_tag))

            # if aspect_f1 + opinion_f1 > best_f1_sum:
            # if aspect_f1 > best_a_f1 and opinion_f1 > best_o_f1:
            if epoch > 29:
                save_result = (cur_f1_sum >= best_f1_sum)
                # print(aspect_f1 + opinion_f1, best_f1_sum, save_result)
                aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                    data_test.texts, data_test.word_idxs_list, data_test.word_span_seqs, data_test.tok_texts,
                    data_test.aspects_true_list, 'tar', data_test.opinions_true_list,
                    dst_aspects_result_file=dst_aspects_file, dst_opinion_result_file=dst_opinions_file,
                    save_result=save_result)
                logging.info('Test, p={:.4f}, r={:.4f}, f1={:.4f}; p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                    aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1))

                if save_result:
                    best_test_af1 = aspect_f1
                    best_test_of1 = opinion_f1

                if self.saver is not None:
                    self.saver.save(self.sess, save_file)
                    # print('model saved to {}'.format(save_file))
                    logging.info('model saved to {}'.format(save_file))

                if dev_aspect_f1 + dev_opinion_f1 > best_f1_sum:
                    best_f1_sum = dev_aspect_f1 + dev_opinion_f1
                if dev_aspect_f1 > best_a_f1 and dev_opinion_f1 > best_o_f1:
                    best_a_f1 = dev_aspect_f1
                    best_o_f1 = dev_opinion_f1
        return best_test_af1, best_test_of1

    def get_feed_dict(self, word_idx_seqs, task, is_train, label_seqs=None, lr=None, dropout=None):
        word_idx_seqs = [list(word_idxs) for word_idxs in word_idx_seqs]
        word_ids, sequence_lengths = utils.pad_sequences(word_idx_seqs, 0)

        # build feed dictionary
        feed = {
            self.word_idxs: word_ids,
            self.sequence_lengths: sequence_lengths,
            self.is_train: is_train
        }

        if label_seqs is not None:
            label_seqs = [list(labels) for labels in label_seqs]
            labels, _ = utils.pad_sequences(label_seqs, 0)
            if task == 'src1':
                feed[self.labels_src1] = labels
            elif task == 'src2':
                feed[self.labels_src2] = labels
            else:
                feed[self.labels_tar] = labels

        feed[self.lr] = lr
        feed[self.dropout] = dropout

        return feed, sequence_lengths

    def __train_batch(self, word_idxs_list_train, labels_list_train, batch_idx, lr, dropout, task):
        word_idxs_list_batch = word_idxs_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        labels_list_batch = labels_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        feed_dict, _ = self.get_feed_dict(word_idxs_list_batch, task, is_train=True,
                                          label_seqs=labels_list_batch, lr=lr, dropout=dropout)

        if task == 'src1':
            _, train_loss = self.sess.run(
                [self.train_op_src1, self.loss_src1], feed_dict=feed_dict)
        elif task == 'src2':
            _, train_loss = self.sess.run(
                [self.train_op_src2, self.loss_src2], feed_dict=feed_dict)
        else:
            _, train_loss = self.sess.run(
                [self.train_op_tar, self.loss_tar], feed_dict=feed_dict)
        return train_loss

    def evaluate(self, texts, word_idxs_list_valid, word_span_seqs, tok_texts, terms_true_list, task,
                 opinions_ture_list=None, dst_aspects_result_file=None, dst_opinion_result_file=None,
                 save_result=False):
        aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt = 0, 0, 0
        opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt = 0, 0, 0
        correct_sent_idxs = list()
        aspect_terms_sys_list, opinion_terms_sys_list = list(), list()
        for sent_idx, (word_idxs, tok_text, terms_true) in enumerate(zip(
                word_idxs_list_valid, tok_texts, terms_true_list)):
            labels_pred, sequence_lengths = self.predict_batch([word_idxs], task)
            labels_pred = labels_pred[0]

            if word_span_seqs is None:
                aspect_terms_sys = utils.get_terms_from_label_list(labels_pred, tok_text, 1, 2)
            else:
                aspect_terms_sys = utils.recover_terms(texts[sent_idx], word_span_seqs[sent_idx], labels_pred, 1, 2)
                aspect_terms_sys = [t.lower() for t in aspect_terms_sys]
            aspect_terms_sys_list.append(aspect_terms_sys)

            new_hit_cnt = utils.count_hit(terms_true, aspect_terms_sys)
            aspect_true_cnt += len(terms_true)
            aspect_sys_cnt += len(aspect_terms_sys)
            aspect_hit_cnt += new_hit_cnt
            if new_hit_cnt == aspect_true_cnt:
                correct_sent_idxs.append(sent_idx)

            if opinions_ture_list is None:
                continue

            opinion_terms_sys = utils.get_terms_from_label_list(labels_pred, tok_text, 3, 4)
            opinion_terms_sys_list.append(opinion_terms_sys)
            opinion_terms_true = opinions_ture_list[sent_idx]

            new_hit_cnt = utils.count_hit(opinion_terms_true, opinion_terms_sys)
            opinion_hit_cnt += new_hit_cnt
            opinion_true_cnt += len(opinion_terms_true)
            opinion_sys_cnt += len(opinion_terms_sys)

        aspect_p, aspect_r, aspect_f1 = utils.prf1(aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt)
        if opinions_ture_list is None:
            return aspect_p, aspect_r, aspect_f1, 0, 0, 0

        opinion_p, opinion_r, opinion_f1 = utils.prf1(opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt)

        if dst_aspects_result_file is not None and save_result:
            datautils.write_terms_list(aspect_terms_sys_list, dst_aspects_result_file)
            logging.info('write aspects to {}'.format(dst_aspects_result_file))
        if dst_opinion_result_file is not None and save_result:
            datautils.write_terms_list(opinion_terms_sys_list, dst_opinion_result_file)
            logging.info('write opinions to {}'.format(dst_opinion_result_file))

        return aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1

    def predict_batch(self, word_idxs, task):
        fd, sequence_lengths = self.get_feed_dict(word_idxs, task, is_train=False, dropout=1.0)

        if task == 'src1':
            logits = self.logits_src1
            trans_params = self.trans_params_src1
        elif task == 'src2':
            logits = self.logits_src2
            trans_params = self.trans_params_src2
        else:
            logits = self.logits_tar
            trans_params = self.trans_params_tar

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits_val, trans_params_val = self.sess.run(
                [logits, trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits_val, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params_val)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths
