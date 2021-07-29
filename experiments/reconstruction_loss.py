import numpy as np
import os
from utils import log
from explanator import Explanator
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist


def get_path2target(_target):
    _cur_pth = os.getcwd()
    _dir_li = _cur_pth.split('/')

    _found = False
    _pref = './'
    for _dir in _dir_li[::-1]:
        if _dir == _target:
            _found = True
            break
        else:
            _pref += '../'

    if not _found:
        _pref = None
    return _pref


def load_data_from_txt(_p, _y):
    _f = open(_p, 'r')
    _X = _f.read().splitlines()
    _Y = [_y] * len(_X)
    _f.close()
    return _X, _Y


def calculate_MRE(source, rec):
    rec_dist_list = []
    corpus = source + rec
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    embeddings = [[], []]
    embeddings[0] = vectorizer.transform(source).toarray()
    embeddings[1] = vectorizer.transform(rec).toarray()
    for i in range(len(source)):
        dist = cdist(embeddings[0][i].reshape(1, -1),
                     embeddings[1][i].reshape(1, -1),
                     metric='cosine').ravel()[0]
        # if dist != 0:
        #     print(dist)
        if np.isnan(dist):
            continue
        rec_dist_list.append(dist)

    log("MRE train: {:.3f}".format(np.mean(rec_dist_list)))
    log("MRE train std: {:.3f}".format(np.std(rec_dist_list)))


def main(args):
    pref = get_path2target(args.workspace)
    pref_data = pref + 'data/' + args.ds + '/'
    file_names = [
        'test0.txt',
        'test1.txt',
        'valid0.txt',
        'valid1.txt',
        'train0.txt',
        'train1.txt'
    ]

    ds = []
    for file in file_names:
        buff, _ = load_data_from_txt(pref_data + file, 0)
        ds += buff

    encoder_input = [text.split() for text in ds]

    generator_path = pref + 'generator/checkpoints/daae/' + args.ds + '/'
    expl = Explanator(pref + 'generator/default.yaml', generator_path)

    rec = expl.reconstruct(encoder_input)
    valid_meters = expl.evaluate_generator(encoder_input)

    rec = [' '.join(text) for text in rec]
    calculate_MRE(ds, rec)

    log_output = ''
    for k, meter in valid_meters.items():
        log_output += ' {} {:.2f},'.format(k, meter.avg)
    log(log_output)
