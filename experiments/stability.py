from utils import *
from explanator import XPROAX
from lime.lime_text import LimeTextExplainer
from collections import OrderedDict
import sys
sys.path.append('blackBox')


def load_data_from_txt(_p, _y):
    _f = open(_p, 'r')
    _X = _f.read().splitlines()
    _Y = [_y] * len(_X)
    _f.close()

    return _X, _Y


def main(args):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    timer = MyTimer()
    metric = 'cosine'
    label_range = {0, 1}

    dataset = 'yelp'
    model_name = 'DNN'

    pref = find_pth2workspace(args.workspace)
    pref_data = pref + 'data/' + dataset + '/'
    pref_model = pref + 'models/' + dataset + '/'

    sentences = [
        'great food',
        'great sushi',
        'great pizza',
        'great beer',
        'amazing food',
        'horrible food',
        'bad food'
    ]

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
    generator_path = pref + 'generator/checkpoints/daae/' + dataset + '/'
    generator_config_path = pref + 'generator/default.yaml'
    expl = XPROAX(generator_config_path, generator_path=generator_path, black_box=m)
    expl.set_metric(metric)

    corpus_path = pref_data + 'generator_train.txt'
    corpus, _ = load_data_from_txt(corpus_path, 0)
    np.random.shuffle(corpus)
    corpus = corpus[:20000]
    expl.load_corpus(corpus)

    expl_lime = LimeTextExplainer(class_names=[0, 1])
    log('Explanation module ready')

    sur_model = 0
    num_other_words = 2
    vocab_size_limit = 200
    forward_selection = True

    for i, sentence in enumerate(sentences):
        log('----------------------------------------------')
        log('-       Explanation provided by XPROAX       -')
        log('----------------------------------------------')
        t_cost = timer.tiktok('epoch')
        if t_cost > 0:
            log('Last epoch cost: {:.2f} s'.format(t_cost), 0)

        log('Target sentence: {}'.format(sentence))

        encoder_input = sentence.split()
        res = expl.explain_instance(encoder_input, sur_model=sur_model, num_other_words=num_other_words,
                                    vocab_size_limit=vocab_size_limit, forward_selection=forward_selection)
        log('Weights of target words:')
        for w_p in res[0]:
            log('\t{}:\t{:.2f}'.format(w_p[0], w_p[1]))

        log('Weights of local important words:')
        for w_p in res[1]:
            log('\t{}:\t{:.2f}'.format(w_p[0], w_p[1]))

        log('----------------------------------------------')
        log('-       Explanation provided by LIME         -')
        log('----------------------------------------------')
        y = get_prediction_instance(m, sentences[i], get_proba=False)
        sign = 1 if y == 1 else -1
        res = expl_lime.explain_instance(sentences[i], m.predict_proba, num_samples=500, num_features=30)
        weights = OrderedDict(res.as_list())

        order_li = [[w, weights[w] * sign] for w in weights]
        for p in order_li:
            log('{}:\t{:.3f}'.format(p[0], p[1]))

        log('')
    t_cost = timer.tiktok('epoch')
    if t_cost > 0:
        log('Last epoch cost: {:.2f} s'.format(t_cost), 0)
