import sys

import os
import numpy as np
from experiments.neigh_loader import file2array
from utils import load_RF, load_DNN, get_pipeline, get_prediction_instance, find_pth2workspace, log
from explanator import XPROAX, XPROAX_ABELE, XPROAX_XSPELLS
import pandas as pd
sys.path.append('blackBox')


def exceed_range(_ptr, _len):
    if _ptr < 0 or _ptr >= _len:
        return True
    return False


def get_prob(_dict, _w):
    _prob = 0.
    if _w in _dict:
        _prob = _dict[_w]
    return _prob


def update_dict(_dict, _w):
    if _w not in _dict:
        _dict[_w] = 1
    else:
        _dict[_w] += 1
    return _dict


def compute_prob_dict(_w, _ws_neighs):
    """
    Args:
        _w: target word
        _ws_neighs: neighboring sentences

    Returns:
        _pref_dict: probabilities of words appearing right before the target word
        _post_dict: probabilities of words appearing right after the target word
    """
    _pref_dict = {}
    _post_dict = {}
    _appear_count = 0
    for _ws in _ws_neighs:
        if _w not in _ws:
            continue
        _len = len(_ws)
        _appear_count += 1
        _idx = _ws.index(_w)
        _pref_idx = _idx - 1
        _post_idx = _idx + 1

        if exceed_range(_pref_idx, _len):
            _pref_dict = update_dict(_pref_dict, '_pad_')
        else:
            _pref_dict = update_dict(_pref_dict, _ws[_pref_idx])

        if exceed_range(_post_idx, _len):
            _post_dict = update_dict(_post_dict, '_pad_')
        else:
            _post_dict = update_dict(_post_dict, _ws[_post_idx])

    def avg_dict(_dict, _count):
        for _it in _dict:
            _dict[_it] /= _count
        return _dict

    if _appear_count != 0:
        _pref_dict = avg_dict(_pref_dict, _appear_count)
        _post_dict = avg_dict(_post_dict, _appear_count)
    return _pref_dict, _post_dict


def find_insertion(_ws_li, _target_w, _ws_neighs, allow_replace=True):
    """
    Args:
        _ws_li: word sequence of target sentence
        _target_w: target word to be inserted
        _ws_neighs: generated local neighbors
        allow_replace: enable the replace action
    Returns:
        _type:      Operation type, 0=insertion, 1=replacement
        _idx:       position for insertion/replacement
        _best_score: maximum probability
    """
    _min_prob = 0.01

    _buff = _ws_li.copy()
    _pref_prob_dict, _post_prob_dict = compute_prob_dict(_target_w, _ws_neighs)
    _pad = '_pad_'

    if allow_replace:
        _ptr_post = 0
        _ptr_pref = -1
        _len = len(_buff)

        _done = exceed_range(_ptr_post, _len) and exceed_range(_ptr_pref, _len)
        _best_ptrs = None
        _best_score = 0.
        _idx = -1
        _type = -1
        while not _done:
            _flag_pref = exceed_range(_ptr_pref, _len)
            if _flag_pref:
                _pref_prob = get_prob(_pref_prob_dict, _pad)
            else:
                _pref_prob = get_prob(_pref_prob_dict, _buff[_ptr_pref])

            _flag_post = exceed_range(_ptr_post, _len)
            if _flag_post:
                _post_prob = get_prob(_post_prob_dict, _pad)
            else:
                _post_prob = get_prob(_post_prob_dict, _buff[_ptr_post])

            _pref_prob = max(_min_prob, _pref_prob)
            _post_prob = max(_min_prob, _post_prob)
            _cur_prob = _pref_prob * _post_prob

            if _cur_prob > _best_score:
                _best_score = _cur_prob
                _best_ptrs = [_ptr_pref, _ptr_post]

            if _ptr_post - _ptr_pref > 1:
                _ptr_pref += 1
            else:
                if _flag_post:
                    break
                _ptr_post += 1
            _done = _flag_pref and _flag_post

        if _best_ptrs is not None:
            _idx = _best_ptrs[0] + 1
            _type = 0
            if _best_ptrs[1] - _best_ptrs[0] > 1:
                _type = 1
    else:
        _ptr = -1
        _len = len(_buff)

        _done = exceed_range(0, _len)
        _best_ptrs = -1
        _best_score = 0.
        _idx = -1
        _type = -1
        while not _done:
            _ptr += 1
            _flag_pref = exceed_range(_ptr - 1, _len)
            if _flag_pref:
                _pref_prob = get_prob(_pref_prob_dict, _pad)
            else:
                _pref_prob = get_prob(_pref_prob_dict, _buff[_ptr - 1])

            _flag_post = exceed_range(_ptr, _len)
            if _flag_post:
                _post_prob = get_prob(_post_prob_dict, _pad)
            else:
                _post_prob = get_prob(_post_prob_dict, _buff[_ptr])

            _pref_prob = max(_min_prob, _pref_prob)
            _post_prob = max(_min_prob, _post_prob)
            _cur_prob = _pref_prob * _post_prob

            if _cur_prob > _best_score:
                _best_score = _cur_prob
                _best_ptrs = _ptr

            _done = _flag_post

        if _best_ptrs >= 0:
            _idx = _ptr
            _type = 0
    return _type, _idx, _best_score


def allow_deletion(_ws, _idx, _extra_words):
    _flag = True
    for _i in [_idx - 1, _idx + 1]:
        try:
            if _ws[_i] in _extra_words:
                _flag = False
                break
        except IndexError:
            pass
    return _flag


def get_confidence_change_with_edition(_m, _ws_li, _ordered_li, _ws_neighs,
                                       _ign_thresh=0.1, _min_prob=0.01, _edit_thresh=None):
    """
    Args:
        _m:             black box model
        _ws_li:         word sequence of target sentence
        _ordered_li:    list of words organized as [(word, weight), word_type], ordered by weights
        _ws_neighs:     generated local neighbors
        _ign_thresh:    weights threshold, under this value will be ignored
        _min_prob:      minimum probability for the words not showed up in neighbors
        _edit_thresh:   probability threshold, NO edition if confidence is low

    Returns:
        _type:          Operation type, 0=insertion, 1=replacement
        _idx:           position for insertion/replacement
        _best_score:    maximum probability
    """
    _replace = '__'

    _ws = np.array(_ws_li, dtype=object)
    _change_li = []
    if _edit_thresh is None:
        _edit_thresh = _min_prob * _min_prob * 2
    _rec = ' '.join(_ws_li)
    log('Original sentence: {}'.format(_rec))
    _y, _y_score = get_prediction_instance(_m, _rec, get_proba=True)
    _change_li.append(_y_score[_y])
    log('Confidence:\t{:.2f}'.format(_y_score[_y] * 100))

    _steps = 0
    _extra_words = []
    for _target_p in _ordered_li:
        log(_target_p)
        _word_type = _target_p[1]
        _target_w = _target_p[0][0]
        _weight = _target_p[0][1]

        if _word_type == 1:  # local important word
            _type, _idx, _max_prob = find_insertion(_ws, _target_w, _ws_neighs, allow_replace=True)
            log('{}, {}, {}'.format(_type, _idx, _max_prob))
            if _type < 0:  # or _max_prob <= _edit_thresh
                continue

            _extra_words.append(_target_w)
            if _type == 0:
                _ws = np.insert(_ws, _idx, _target_w)
                _steps += 1
            else:
                _steps += 1 if _ws[_idx] == _replace else 2
                _ws[_idx] = _target_w
            # log('Sentence after edition: {}'.format(' '.join(_ws)))
        else:  # target word
            if abs(_weight) < _ign_thresh:
                break
            if _weight < 0:
                continue
            _idxs = np.where(_ws == _target_w)[0]
            for _id in _idxs:
                if not allow_deletion(_ws, _id, _extra_words):
                    continue
                _ws[_id] = _replace
                _steps += 1
        _rec = ' '.join(_ws)
        log('Sentence after edition: {}'.format(_rec))
        _, _new_score = get_prediction_instance(_m, _rec, get_proba=True)
        _new_score = _new_score[_y]

        _change_li.append(_new_score)

    _maximum_drop = _change_li[0] - _change_li[-1]
    if _steps > 0:
        _final_efficient = _maximum_drop / _steps
    else:
        _final_efficient = 0.
    return _maximum_drop, [_final_efficient, _steps], _change_li


def get_confidence_change(_m, _ws_li, _ordered_li, _replace='', _ign_thresh=0.1, _insert=False):
    _ws = np.array(_ws_li, dtype=object)
    _rec = ' '.join(_ws)
    _y, _y_score = get_prediction_instance(_m, _rec, get_proba=True)
    _y_score = _y_score[_y]

    _min_y = _y_score
    _most_efficient = 0.
    _most_efficient_step = 0
    _final_efficient = 0.
    _steps = 0
    _change_li = [_y_score]

    _one_neg = True
    for _pair in _ordered_li:
        if abs(_pair[1]) < _ign_thresh:
            break
        _w = _pair[0]
        if _pair[1] < 0:
            if _insert and _one_neg:
                _ws = np.insert(_ws, len(_ws), _w)
                _steps += 1
                _one_neg = False
            else:
                continue
        else:
            _idxs = np.where(_ws == _w)[0]
            _steps += len(_idxs)
            if _steps == 0:
                continue
            for _id in _idxs:
                _ws[_id] = _replace
        _rec = ' '.join(_ws)
        _, _new_score = get_prediction_instance(_m, _rec, get_proba=True)
        _new_score = _new_score[_y]

        _change_li.append(_new_score)
        _min_y = _new_score
        _efficiency = (_y_score - _new_score) / _steps
        _final_efficient = _efficiency

        if _efficiency > _most_efficient:
            _most_efficient = _efficiency
            _most_efficient_step = _steps
    _max_effect = _y_score - _min_y
    return _max_effect, [_most_efficient, _most_efficient_step], _final_efficient, _change_li


def generate_explanations(ds, expl, m, ign_thres, vocab_size_limit=200, surrogate=None):
    """
        Generate explanations based on the constructed neighborhood for each single input.
        The neighborhood is constructed using the approach implemented in 'explain_instance',
        and is stored in local file for further use in order to save time and to avoid
        re-running the generation process every single time, which is time consuming.
    """
    drop_li = []
    efficiency_li = []
    count = 0
    failed_count = 0

    num_other_words = 10
    forward_selection = False

    explanations = []
    columns = ['input', 'words', 'local_words', 'exemplars']
    for exemplars in ds:
        log('********************************')
        log('Explaining sentence with index {}'.format(count))
        log('********************************')
        if surrogate is not None:
            res = expl.explain_with_given_exemplars(exemplars, num_other_words=num_other_words,
                                                    vocab_size_limit=vocab_size_limit,
                                                    forward_selection=forward_selection,
                                                    sur_model=surrogate)
        else:
            res = expl.explain_with_given_exemplars(exemplars, num_other_words=num_other_words,
                                                    vocab_size_limit=vocab_size_limit,
                                                    forward_selection=forward_selection)
        rec = expl.decode(exemplars)
        ws_li = rec[0]
        neighs = rec[1:]

        k = 1
        extra_words = []
        for p in res[1]:
            if abs(p[1]) < ign_thres or len(extra_words) >= k:
                break
            if p[1] < 0:
                extra_words.append([p[0], p[1]])

        ptr_extra = 0
        order_list = []
        for p in res[0]:
            if ptr_extra < len(extra_words) and abs(p[1]) < abs(extra_words[ptr_extra][1]):
                order_list.append([extra_words[ptr_extra], 1])
                ptr_extra += 1
            else:
                if p[1] < ign_thres:
                    continue
                order_list.append([p, 0])

        max_effect, efficiency, change_li = get_confidence_change_with_edition(m, ws_li, order_list, neighs,
                                                                               _ign_thresh=0.1,
                                                                               _min_prob=0.01,
                                                                               _edit_thresh=None)
        buff = [rec[0], res[0], res[1], rec[1:]]
        explanations.append(buff)
        log('Max confidence change: {:.2f}%'.format(max_effect * 100))
        log('Most efficient change: {:.2f}% at step {}'.format(efficiency[0] * 100, efficiency[1]))

        local_eff = [change_li[i] - change_li[i + 1] for i in range(len(change_li) - 1)]

        if change_li[-1] >= 0.5:
            failed_count += 1
        change_li = [str(round(v * 100, 2)) for v in change_li]
        log('Change list: {}'.format(change_li))
        log()

        count += 1
        drop_li.append(max_effect)
        if efficiency[1] != 0:
            efficiency_li += local_eff
    df = pd.DataFrame(explanations, columns=columns)
    return drop_li, efficiency_li, failed_count, df


def unload_df(df, m, ign_thres):
    drop_li = []
    efficiency_li = []
    count = 0
    failed_count = 0

    columns = ['input', 'words', 'local_words', 'exemplars']
    for _, explanation in df.iterrows():
        log('********************************')
        log('Explaining sentence with index {}'.format(count))
        log('********************************')
        ws_li = explanation[columns[0]]
        neighs = explanation[columns[3]]
        res = [explanation[columns[1]], explanation[columns[2]]]

        k = 1
        extra_words = []
        for p in res[1]:
            if abs(p[1]) < ign_thres or len(extra_words) >= k:
                break
            if p[1] < 0:
                extra_words.append([p[0], p[1]])

        ptr_extra = 0
        order_list = []
        for p in res[0]:
            if ptr_extra < len(extra_words) and abs(p[1]) < abs(extra_words[ptr_extra][1]):
                order_list.append([extra_words[ptr_extra], 1])
                ptr_extra += 1
            else:
                if p[1] < ign_thres:
                    continue
                order_list.append([p, 0])

        max_effect, efficiency, change_li = get_confidence_change_with_edition(m, ws_li, order_list, neighs,
                                                                               _ign_thresh=0.1,
                                                                               _min_prob=0.01,
                                                                               _edit_thresh=None)

        log('Max confidence change: {:.2f}%'.format(max_effect * 100))
        log('Most efficient change: {:.2f}% at step {}'.format(efficiency[0] * 100, efficiency[1]))

        local_eff = [change_li[i] - change_li[i + 1] for i in range(len(change_li) - 1)]

        if change_li[-1] >= 0.5:
            failed_count += 1
        change_li = [str(round(v * 100, 2)) for v in change_li]
        log('Change list: {}'.format(change_li))
        log()

        count += 1
        drop_li.append(max_effect)
        if efficiency[1] != 0:
            efficiency_li += local_eff
    return drop_li, efficiency_li, failed_count


def load_test_set(_p_li, _idxs=None):
    """
    Parameters:
    ----------
    _p_li: a list contains two paths, each file for one class
    _idxs: indexes of target sentences

    Returns:
    ----------
    _test_x: np.array, sentences
    _test_y: np.array, labels
    """

    def load_data_from_txt(_p, _y):
        _f = open(_p, 'r')
        _X = _f.read().splitlines()
        _Y = [_y] * len(_X)
        _f.close()
        return _X, _Y

    _test_x = []
    _test_y = []

    _ds = load_data_from_txt(_p_li[0], 0)
    _test_x += _ds[0]
    _test_y += _ds[1]
    _ds = load_data_from_txt(_p_li[1], 1)
    _test_x += _ds[0]
    _test_y += _ds[1]

    _test_x = np.array(_test_x)
    _test_y = np.array(_test_y)
    if _idxs is not None:
        _test_x = _test_x[_idxs]
        _test_y = _test_y[_idxs]
    return _test_x, _test_y


def load_sentiment_words(pth):
    f = open(pth, 'r')
    buff = f.readlines()
    f.close()

    res = []
    for line in buff:
        res.append(line.split()[0])
    return res


def do_xproax(args, path2workspace, m):
    method_name = args.method
    method_name = method_name.lower()
    generator_config_path = path2workspace + 'generator/default.yaml'
    generator_path = path2workspace + 'generator/checkpoints/daae/' + args.ds + '/'
    metric = 'cosine'

    if method_name == 'abele':
        expl = XPROAX_ABELE(generator_config_path, generator_ath=generator_path, black_box=m)
    else:
        expl = XPROAX(generator_config_path, generator_path=generator_path, black_box=m)
    expl.set_metric(metric)

    pickle_file_pth = path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_' + method_name + '_explain.pickle'
    if os.path.exists(pickle_file_pth):
        df = pd.read_pickle(pickle_file_pth)
        drop_li, efficiency_li, failed_count = unload_df(df, m, args.thresh)
    else:
        dataset_path = path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_' + method_name + '_neigh.txt'
        ds = file2array(dataset_path)
        drop_li, efficiency_li, failed_count, df = generate_explanations(ds, expl, m, args.thresh, args.vocab_size)
        df.to_pickle(pickle_file_pth)

    log('Mean confidence drop: {:.2f}% \u00B1 {:.2f}'.format(np.mean(drop_li) * 100, np.std(drop_li) * 100))
    log('Mean final efficiency: {:.2f}% \u00B1 {:.2f}'.format(np.mean(efficiency_li) * 100, np.std(efficiency_li) * 100))
    log('Failed count: {}'.format(failed_count))


def do_xspells(args, path2workspace, m):
    generator_config_path = path2workspace + 'generator/default.yaml'
    generator_path = path2workspace + 'generator/checkpoints/daae/' + args.ds + '/'
    metric = 'cosine'

    expl = XPROAX_XSPELLS(generator_config_path, generator_path=generator_path, black_box=m)
    expl.set_metric(metric)

    if args.surrogate == 0:
        pickle_file_pth = path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_xspells_latent.pickle'
    else:
        pickle_file_pth = path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_xspells_textual.pickle'
    if os.path.exists(pickle_file_pth):
        df = pd.read_pickle(pickle_file_pth)
        drop_li, efficiency_li, failed_count = unload_df(df, m, args.thresh)
    else:
        dataset_path = path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_xspells_neigh.txt'
        ds = file2array(dataset_path)
        drop_li, efficiency_li, failed_count, df = generate_explanations(ds, expl, m, args.thresh, args.vocab_size,
                                                                         surrogate=args.surrogate)
        df.to_pickle(pickle_file_pth)

    log('Mean confidence drop: {:.2f}% \u00B1 {:.2f}'.format(np.mean(drop_li) * 100, np.std(drop_li) * 100))
    log('Mean final efficiency: {:.2f}% \u00B1 {:.2f}'.format(np.mean(efficiency_li) * 100, np.std(efficiency_li) * 100))
    log('Failed count: {}'.format(failed_count))


def do_xspells_old(args, path2workspace, m):
    columns = ['input', 'exemplar_words', 'counter_words', 'exemplar', 'counter-exemplar']

    dataset_path = path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_xspells.pkl'
    df = pd.read_pickle(dataset_path)

    drop_li = []
    efficiency_li = []
    count = -1
    failed_count = 0

    lower_bound = 2000
    upper_bound = 4000
    for i, entry in df.iterrows():
        count += 1
        flag0 = lower_bound <= count < 2000
        flag1 = upper_bound <= count
        if flag0 or flag1:
            continue
        log('********************************')
        log('Explaining sentence with index {}'.format(count))
        log('********************************')

        ws_li = entry[columns[0]].split()

        exemplar_words = entry[columns[1]]
        counter_words = entry[columns[2]]
        neighs = entry[columns[3]] + entry[columns[4]]
        neighs = [s.split() for s in neighs]

        log(exemplar_words)
        log(counter_words)

        k = 1
        extra_words = []
        for p in counter_words:
            weight = float(p[1]) * -1
            if abs(weight) < args.thresh or len(extra_words) >= k:
                break
            if p[0] not in ws_li:
                extra_words.append([p[0], weight])

        ptr_extra = 0
        order_list = []
        for p in exemplar_words:
            weight = float(p[1])
            if ptr_extra < len(extra_words) and abs(weight) < abs(extra_words[ptr_extra][1]):
                order_list.append([extra_words[ptr_extra], 1])
                ptr_extra += 1
            elif p[0] in ws_li:
                order_list.append([[p[0], weight], 0])

        max_effect, efficiency, change_li = get_confidence_change_with_edition(m, ws_li, order_list, neighs,
                                                                               _ign_thresh=0.1,
                                                                               _min_prob=0.01,
                                                                               _edit_thresh=None)

        log('Max confidence change: {:.2f}%'.format(max_effect * 100))
        log('Most efficient change: {:.2f}% at step {}'.format(efficiency[0] * 100, efficiency[1]))
        if change_li[-1] >= 0.5:
            failed_count += 1
        change_li = [str(round(v * 100, 2)) for v in change_li]
        log('Change list: {}'.format(change_li))
        log()

        drop_li.append(max_effect)
        if efficiency[1] != 0:
            efficiency_li.append(efficiency[0])
    log('Mean confidence drop: {:.2f}% \u00B1 {:.2f}'.format(np.mean(drop_li) * 100,
                                                             np.std(drop_li) * 100))
    log('Mean final efficiency: {:.2f}% \u00B1 {:.2f}'.format(np.mean(efficiency_li) * 100,
                                                              np.std(efficiency_li) * 100))
    log('Failed count: {}'.format(failed_count))


def do_lime(args, path2workspace, m):
    from lime.lime_text import LimeTextExplainer
    from collections import OrderedDict

    replace = '__'
    shift = 0
    # number_sentences = 250
    pref_data = path2workspace + 'data/' + args.ds + '/'
    filenames = [pref_data + 'test0.txt', pref_data + 'test1.txt']
    sentences, _ = load_test_set(filenames)
    sentences = np.concatenate([sentences[shift:args.num + shift], sentences[2000:2000 + args.num]])

    expl = LimeTextExplainer(class_names=[0, 1])

    drop_li = []
    efficiency_li = []
    count = 0
    failed_count = 0
    for i in range(len(sentences)):
        log('********************************')
        log('Explaining sentence with index {}'.format(count))
        log('********************************')
        log('Input: {}'.format(sentences[i]))
        y = get_prediction_instance(m, sentences[i], get_proba=False)
        sign = 1 if y == 1 else -1

        res = expl.explain_instance(sentences[i], m.predict_proba, num_samples=500, num_features=30)
        weights = OrderedDict(res.as_list())

        order_li = [[w, weights[w] * sign] for w in weights]
        for p in order_li:
            log('{}:\t{:.3f}'.format(p[0], p[1]))

        ws_li = sentences[i].split()
        max_effect, efficiency, final_efficiency, change_li = get_confidence_change(m, ws_li, order_li, _replace=replace,
                                                                                    _ign_thresh=args.thresh, _insert=True)
        log('Max confidence change: {:.2f}%'.format(max_effect * 100))
        log('Most efficient change: {:.2f}% at step {}'.format(efficiency[0] * 100, efficiency[1]))
        log('Final change efficiency: {:.2f}%'.format(final_efficiency * 100))

        local_eff = [change_li[i] - change_li[i + 1] for i in range(len(change_li) - 1)]

        if change_li[-1] >= 0.5:
            failed_count += 1
        change_li = [str(round(v * 100, 2)) for v in change_li]
        log('Change list: {}'.format(change_li))
        log()

        count += 1
        drop_li.append(max_effect)
        if efficiency[1] != 0:
            efficiency_li += local_eff
    log('Mean confidence drop: {:.2f}% \u00B1 {:.2f}'.format(np.mean(drop_li) * 100,
                                                             np.std(drop_li) * 100))
    log('Mean final efficiency: {:.2f}% \u00B1 {:.2f}'.format(np.mean(efficiency_li) * 100,
                                                              np.std(efficiency_li) * 100))
    log('Failed count: {}'.format(failed_count))


def do_random(args, path2workspace, m):
    number_sentences = 250
    pref_data = path2workspace + 'data/' + args.ds + '/'
    filenames = [pref_data + 'test0.txt', pref_data + 'test1.txt']
    sentences, _ = load_test_set(filenames)
    sentences = np.concatenate([sentences[0:number_sentences], sentences[2000:2000 + number_sentences]])

    pos_words = load_sentiment_words(path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_pos_words.txt')
    neg_words = load_sentiment_words(path2workspace + 'experiments/storage/' + args.ds + '_exemplars/' + args.model + '_neg_words.txt')

    drop_li = []
    efficiency_li = []
    count = 0
    failed_count = 0
    for i in range(number_sentences):
        log('********************************')
        log('Explaining sentence with index {}'.format(count))
        log('********************************')
        log('Input: {}'.format(sentences[i]))
        y, y_score = get_prediction_instance(m, sentences[i], get_proba=True)
        org_confidence = y_score[y]

        ws_li = sentences[i].split()

        random_pick = np.random.choice(len(pos_words), 1, replace=False)[0]
        picked = pos_words[random_pick] if y == 0 else neg_words[random_pick]

        operation_type = np.random.choice(3, 1, replace=False)[0]
        if operation_type == 0:
            # insertion
            operation_idx = np.random.choice(len(ws_li) + 1, 1, replace=False)[0]
            ws_li.insert(operation_idx, picked)
        elif operation_type == 1:
            # replacement
            operation_idx = np.random.choice(len(ws_li), 1, replace=False)[0]
            ws_li[operation_idx] = picked
        else:
            pass
        rec = ' '.join(ws_li)
        _, y_score = get_prediction_instance(m, rec, get_proba=True)
        edited_confidence = y_score[y]
        max_effect = org_confidence - edited_confidence
        steps = operation_type + 1
        efficiency = [max_effect / steps, steps]
        final_efficiency = efficiency[0]
        change_li = [org_confidence, edited_confidence]
        log('Max confidence change: {:.2f}%'.format(max_effect * 100))
        log('Most efficient change: {:.2f}% at step {}'.format(efficiency[0] * 100, efficiency[1]))
        log('Final change efficiency: {:.2f}%'.format(final_efficiency * 100))

        local_eff = [change_li[i] - change_li[i + 1] for i in range(len(change_li) - 1)]

        if change_li[-1] >= 0.5:
            failed_count += 1
        change_li = [str(round(v * 100, 2)) for v in change_li]
        log('Change list: {}'.format(change_li))
        log()

        count += 1
        drop_li.append(max_effect)
        if efficiency[1] != 0:
            efficiency_li += local_eff
    log('Mean confidence drop: {:.2f}% \u00B1 {:.2f}'.format(np.mean(drop_li) * 100,
                                                             np.std(drop_li) * 100))
    log('Mean final efficiency: {:.2f}% \u00B1 {:.2f}'.format(np.mean(efficiency_li) * 100,
                                                              np.std(efficiency_li) * 100))
    log('Failed count: {}'.format(failed_count))


def main(args):
    method = args.method

    pref = find_pth2workspace(args.workspace)
    pref_model = pref + 'models/' + args.ds + '/'

    if args.model == 'RF':
        pickled_black_box_filename = pref_model + args.model + '_model.sav'
        pickled_vectorizer_filename = pref_model + 'tfidf_vectorizer.pickle'
        rf_model, vectorizer = load_RF(pickled_black_box_filename, pickled_vectorizer_filename)
        m = get_pipeline(rf_model, vectorizer)
    else:
        filename = pref_model + args.model + '_model.sav'
        m = load_DNN(filename)

    if method == 'BASELINE':
        do_random(args, pref, m)
    elif method == 'LIME':
        do_lime(args, pref, m)
    elif method == 'XPROAX' or method == 'ABELE':
        do_xproax(args, pref, m)
    elif method == 'XSPELLS':
        do_xspells(args, pref, m)
        # do_xspells_old(args, pref, m)
