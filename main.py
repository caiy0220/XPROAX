from utils import *
from explanator import XPROAX
import sys
sys.path.append('blackBox')


def load_data_from_txt(_p, _y):
    _f = open(_p, 'r')
    _X = _f.read().splitlines()
    _Y = [_y] * len(_X)
    _f.close()
    return _X, _Y


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


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    metric = 'cosine'

    timer = MyTimer()

    if len(sys.argv) >= 3:
        shift = int(sys.argv[1])
        number_sentences = int(sys.argv[2])
    else:
        shift = 0
        number_sentences = 5

    if len(sys.argv) >= 5:
        dataset = sys.argv[3]
        model_name = sys.argv[4]
    else:
        dataset = 'yelp'
        model_name = 'RF'

    label_range = {0, 1}

    save_neighbors = False

    if save_neighbors:
        neigh_saving_path = 'neigh_' + str(shift) + '_' + str(shift+number_sentences) + '.txt'
        neigh_saving_file = open(neigh_saving_path, 'w')
    else:
        neigh_saving_file = None

    pref_data = './data/' + dataset + '/'
    pref_model = './models/' + dataset + '/'

    filenames = [pref_data + 'test0.txt', pref_data + 'test1.txt']
    sentences, labels = load_test_set(filenames)
    sentences = sentences[shift:number_sentences + shift]
    labels = labels[shift:number_sentences + shift]

    # Load black box model
    if model_name == 'RF':
        pickled_black_box_filename = pref_model + model_name + '_model.sav'
        pickled_vectorizer_filename = pref_model + 'tfidf_vectorizer.pickle'
        rf_model, vectorizer = load_RF(pickled_black_box_filename, pickled_vectorizer_filename)
        m = get_pipeline(rf_model, vectorizer)
    else:
        filename = pref_model + model_name + '_model.sav'
        m = load_DNN(filename)
    log('Black box loaded')

    # Initialization explanation tool
    generator_path = 'generator/checkpoints/daae/' + dataset + '/'
    generator_config_path = './generator/default.yaml'
    expl = XPROAX(generator_config_path, generator_path=generator_path, black_box=m)
    expl.set_metric(metric)

    corpus_path = pref_data + 'generator_train.txt'
    corpus, _ = load_data_from_txt(corpus_path, 0)
    corpus = corpus[:20000]
    expl.load_corpus(corpus)
    log('Explanation module ready')

    sur_model = 0
    num_other_words = 10
    vocab_size_limit = 200
    forward_selection = True

    for i, sentence in enumerate(sentences):
        log('********************************')
        log('Explaining sentence with index {}'.format(i))
        log('********************************')
        t_cost = timer.tiktok('epoch')
        if t_cost > 0:
            log('Last epoch cost: {:.2f} s'.format(t_cost), 0)

        log('Target sentence: {}'.format(sentence))

        encoder_input = sentence.split()
        res = expl.explain_instance(encoder_input, sur_model=sur_model, num_other_words=num_other_words,
                                    vocab_size_limit=vocab_size_limit, forward_selection=forward_selection,
                                    log_f=neigh_saving_file)
        log('Weights of target words:')
        for w_p in res[0]:
            log('\t{}:\t{:.2f}'.format(w_p[0], w_p[1]))

        log('------------------------')
        log('Weights of local important words:')
        for w_p in res[1]:
            log('\t{}:\t{:.2f}'.format(w_p[0], w_p[1]))

        log('')

    t_cost = timer.tiktok('epoch')
    if t_cost > 0:
        log('Last epoch cost: {:.2f} s'.format(t_cost), 0)
