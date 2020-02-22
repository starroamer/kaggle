from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class TweetLoader(object):
    def __init__(self, class_num):
        self.keyword_id_dict = self.load_word_id('keyword_id')
        self.word_id_dict = self.load_word_id('text_word_id_no_stopwords')
        self.keyword_num = len(self.keyword_id_dict)
        self.text_word_num = len(self.word_id_dict)
        self.class_num = class_num

    def load_word_id(self, fname):
        dic = defaultdict(int)
        with open(fname) as fp:
            for line in fp:
                line = line.strip()
                word, idx = line.split('\t')
                dic[word] = int(idx)

        return dic

    def load_data(self, fname, repeat=1, batch=1, test_data_ratio=0.1, predict=False):
        data = [self._parse_data(line, predict) for line in open(fname)] 
        tf.logging.info("read %s complete, generating dataset ..." % fname)
        feature, label = zip(*data)
        feature = list(feature)
        label = list(label)
        dataset = tf.data.Dataset.from_tensor_slices((feature, label))
        dataset = dataset.skip(1)
        tf.logging.info("generate dataset complete")

        if predict:
            dataset = dataset.batch(batch)
            return dataset
        else:
            dataset = dataset.shuffle(buffer_size=10000)
            test_data_num = int(len(data) * test_data_ratio)
            test_dataset = dataset.take(test_data_num).batch(batch)

            train_dataset = dataset.skip(test_data_num)
            train_dataset = train_dataset.batch(batch).repeat(repeat)

            tf.logging.info("split train and test dataset complete")

            return train_dataset, test_dataset

    def _parse_data(self, line, predict):
        line = line.strip()
        flds = line.split('\t')
        idx, keyword, loc, text = flds[:4]
        feature = self._parse_content(keyword, text)
        if predict:
            return (feature, idx)
        else:
            label = 0
            if flds[4].isdigit():
                label = int(flds[4])
            return (feature, tf.one_hot(label, self.class_num))

    def _parse_content(self, keyword, text):
        keyword_vec = np.zeros(self.keyword_num + 1, dtype=float)
        keyword_vec[self.keyword_id_dict[keyword]] = 1

        text_word_vec = np.zeros(self.text_word_num + 1, dtype=float)
        for word in text.split(' '):
            text_word_vec[self.word_id_dict[word]] = 1

        feature = np.append(keyword_vec, text_word_vec).tolist()

        return feature
