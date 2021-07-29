import os
import re
import string
import time
import pickle
import numpy as np
import yaml
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_pipeline


LOG_LEVEL = 1
# 0: all log allowed
# 1: debug disabled
# 2: only warning and error
# 3: only error
LOG_PREF = ['DEBUG', 'INFO', 'WARN', 'ERR']


class MyTimer:
    def __init__(self):
        self.clock = {}
        return

    def tiktok(self, stamp):
        if stamp not in self.clock:
            self.clock[stamp] = time.time()
            diff = -1
        else:
            cur = time.time()
            diff = cur - self.clock[stamp]
            self.clock[stamp] = cur
        return diff


class IndexedStrings(object):
    """String with various indexes."""

    def __init__(self, raw_strings, vocab_size_limit=200, mask_string=None, forward_selection=False):
        """Initializer.
        Args:
            raw_strings: strings with raw texts in it
            mask_string: If not None, replace words with this if bow=False
                if None, default value is UNKWORDZ
        """
        self.raw = raw_strings
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string

        self.as_list = [re.sub('[{}]'.format(string.punctuation), " ", sent).split() for sent in self.raw]
        # self.as_list = [sent.split() for sent in self.raw]

        self.as_np = np.array(self.as_list)

        self.vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.max_sequence = 0

        # Build the vocabulary
        if forward_selection:
            # appearance based
            for sent in self.as_np:
                if len(sent) > self.max_sequence:
                    self.max_sequence = len(sent)
                for word in sent:
                    if word not in self.vocab:
                        if len(self.vocab) >= vocab_size_limit:
                            break
                        self.vocab[word] = len(self.vocab)
                        self.inverse_vocab.append(word)
                if len(self.vocab) >= vocab_size_limit:
                    break
        else:
            # target sentence based
            target_sent = self.as_np[0]
            self.max_sequence = len(target_sent)
            for word in target_sent:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                    self.inverse_vocab.append(word)
            vocab_size_limit = max(0, vocab_size_limit - len(self.vocab))

            buff_vocab = {}
            for sent in self.as_np[1:]:
                if len(sent) > self.max_sequence:
                    self.max_sequence = len(sent)
                for i, word in enumerate(sent):
                    if word in self.vocab:
                        continue
                    if word not in buff_vocab:
                        buff_vocab[word] = 1
                    else:
                        buff_vocab[word] += 1
            sorted_list = sorted(buff_vocab.items(), key=lambda ins: ins[1])
            start_from = max(0, len(sorted_list) - vocab_size_limit)
            sorted_list = sorted_list[start_from:]
            for w_count_pair in sorted_list[::-1]:
                w = w_count_pair[0]
                self.vocab[w] = len(self.vocab)
                self.inverse_vocab.append(w)

        self.vocab[self.mask_string] = len(self.vocab)
        self.inverse_vocab.append(self.mask_string)

    def get_indexed(self, sents=None, bow=True):
        if sents is not None:
            sents = [re.sub('[{}]'.format(string.punctuation), " ", sent).split() for sent in sents]
            sents = np.array(sents)
        else:
            sents = self.as_np
        feature_size = self.num_words() if bow else self.max_sequence
        res = np.zeros([len(sents), feature_size])
        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                if word in self.vocab:
                    word_idx = self.vocab[word]
                else:
                    word_idx = self.vocab[self.mask_string]
                if bow:
                    res[i, word_idx] = 1
                else:
                    if j >= self.max_sequence:
                        break
                    res[i, j] = word_idx
        return res

    def get_indexed_summary(self, sents=None):
        if sents is not None:
            sents = [re.sub('[{}]'.format(string.punctuation), " ", sent).split() for sent in sents]
            sents = np.array(sents)
        else:
            sents = self.as_np
        feature_size = self.num_words()
        res = np.zeros([len(sents), feature_size*3])
        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                if word in self.vocab:
                    word_idx = self.vocab[word]
                else:
                    word_idx = self.vocab[self.mask_string]
                real_idx = word_idx*3
                res[i, real_idx] += j       # LS
                res[i, real_idx + 1] += j*j     # SS^2
                res[i, real_idx + 2] += 1       # N

        return res

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]


def set_log_level(lvl):
    global LOG_LEVEL
    LOG_LEVEL = lvl


def log(content='', lvl=1, end='\n'):
    if LOG_LEVEL > lvl:
        return

    pref = LOG_PREF[lvl] + '[' + time.strftime('%x,%X') + ']:'
    print(pref, end='\t')
    print(content, end=end)


def is_number(src_str):
    try:
        float(src_str)
        return True
    except ValueError:
        return False


def sorting_neigh(z, idx_list, metric, z_target=None):
    z = np.array(z)
    if z_target is None:
        z_target = z[0]
    dist = cdist(z[idx_list], z_target.reshape(1, -1), metric=metric).ravel()
    dist_dic = dict(zip(idx_list, dist))

    sorted_list = sorted(dist_dic.items(), key=lambda ins: ins[1])
    return sorted_list


def calc_distance_objs2obj(li_objs, obj, metric='cosine'):
    if len(li_objs) == 0:
        return np.array([float('inf')])
    li = [o.z for o in li_objs]
    li = np.array(li)
    p = np.array(obj.z)
    return cdist(li, p.reshape(1, -1), metric=metric).ravel()


def load_config(path):
    try:
        f = open(path)
        config = yaml.load(f.read())
    except FileNotFoundError:
        config = None
    return config


def load_RF(model_filename, vec_filename):
    loaded_model = pickle.load(open(model_filename, 'rb'))
    vectorizer = None
    if vec_filename is not None:
        vectorizer = pickle.load(open(vec_filename, 'rb'))
    return loaded_model, vectorizer


def load_DNN(model_filename):
    model = pickle.load(open(model_filename, 'rb'))
    return model


def get_pipeline(model, vectorizer):
    return make_pipeline(vectorizer, model)


def get_prediction(m, x, c=None, get_proba=False):
    """
    Parameters:
    ---------
    m: model
    x: list of instances
    c: class labels
    get_proba: flag for returning confidence score
    """
    if c is not None:
        pass
    score = m.predict_proba(x)
    y_p = np.argmax(score, axis=1)
    if get_proba:
        y_p = [y_p, score]
    return y_p


def get_prediction_instance(m, x, c=None, get_proba=False):
    """
    Parameters:
    ---------
    m: model
    x: list of instances
    c: class labels
    get_proba: flag for returning confidence score
    """
    if c is not None:
        pass
    score = m.predict_proba([x])
    y_p = np.argmax(score, axis=1)
    y_p = y_p[0]
    score = score[0]
    if get_proba:
        y_p = [y_p, score]
    return y_p


def lerp(t, p, q):
    return (1-t) * p + t * q


def interpolate(z1, z2, n):
    z = []
    for i in range(n):
        zi = lerp(1.0*i/(n-1), z1, z2)
        z.append(np.expand_dims(zi, axis=0))
    return np.concatenate(z, axis=0)


def find_pth2workspace(folder_name):
    pref = ''
    pth = os.getcwd()
    pth_segs = pth.split(os.sep)
    pth_segs = pth_segs[::-1]
    found = False
    for dir_name in pth_segs:
        if dir_name != folder_name:
            pref += '../'
        else:
            found = True
            break
    if not found:
        raise Exception('Cannot find workspace {}'.format(folder_name))
    return pref


def distance_neighbors(a):
    a = a if isinstance(a, np.ndarray) else np.array(a)
    d = cdist(a, a[0].reshape(1, -1), metric='cosine').ravel()
    return np.mean(d)


def diversity_neighbors(a):
    a = a if isinstance(a, np.ndarray) else np.array(a)
    b = pickle.loads(pickle.dumps(a))
    b = b - b[0]
    num = len(b)
    normalize = (num**2 + num) / 2
    res = 0.
    for i in range(1, num):
        for j in range(1, num):
            d = cdist(b[i].reshape(1, -1), b[j].reshape(1, -1), metric='cosine').ravel()[0]
            res += 0. if np.isnan(d) else d/normalize
    return res
