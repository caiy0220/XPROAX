"""
Train/Evaluate a black box for specified dataset.
"""

import os
import pickle

from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

# RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.wrappers.scikit_learn import KerasClassifier
from DNN_base import TextsToSequences, Padder, create_model


def load_data_from_txt(_p, _y):
    _f = open(_p, 'r')
    _X = _f.read().splitlines()
    _Y = [_y] * len(_X)
    _f.close()

    return _X, _Y


def load_data(pref, ds, data_type):
    X = []
    Y = []
    pth0 = pref + 'data/' + ds + '/' + data_type + '0.txt'
    pth1 = pref + 'data/' + ds + '/' + data_type + '1.txt'
    ds = load_data_from_txt(pth0, 0)
    X += ds[0]
    Y += ds[1]
    ds = load_data_from_txt(pth1, 1)
    X += ds[0]
    Y += ds[1]
    X, Y = shuffle(X, Y)
    return X, Y


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


class BlackBoxCreator:
    def __init__(self, args):
        self.model = args.model
        self.mode = args.mode
        self.ds = args.ds
        self.pref = find_pth2workspace(args.workspace)
        self.epoch = args.epoch

        if self.mode == '1':
            self.data = load_data(self.pref, self.ds, 'train')
            self.data_valid = load_data(self.pref, self.ds, 'valid')
            self.train()
        else:
            self.data = load_data(self.pref, self.ds, 'test')
            self.data_valid = None
            self.test()

    def train(self):
        if not os.path.exists(self.pref + 'models/' + self.ds):
            os.makedirs(self.pref + 'models/' + self.ds)
        if self.model == 'RF':
            self._train_RF()
        elif self.model == 'DNN':
            self._train_DNN()
        else:
            print('Unknown model type, stop')
            return -1
        return 1

    def _train_RF(self):
        vectorizer = TfidfVectorizer(sublinear_tf='false')
        train_vectors = vectorizer.fit_transform(self.data[0])
        test_vectors = vectorizer.transform(self.data_valid[0])
        pickle.dump(vectorizer, open(self.pref + 'models/' + self.ds + '/tfidf_vectorizer.pickle', "wb"))
        print('Vectorizer is ready')

        # Using random forest for classification.
        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                    max_depth=1000, max_features=1000, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=4, min_samples_split=10,
                                    min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
                                    oob_score=False, random_state=None, verbose=0,
                                    warm_start=False)
        print('************* Start training *************')
        rf.fit(train_vectors, self.data[1])

        # save the model to disk
        filename = self.pref + 'models/' + self.ds + '/' + self.model + '_model.sav'
        pickle.dump(rf, open(filename, 'wb'))
        print('Training done, saving model and doing test')
        print('******************************************\n')

        print('Reloading trained model for validation')
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))

        # Computing interesting metrics/classification report
        pred = loaded_model.predict(test_vectors)
        print(classification_report(self.data_valid[1], pred))

    def _train_DNN(self):
        sequencer = TextsToSequences(num_words=35000)
        padder = Padder(140)
        my_model = KerasClassifier(build_fn=create_model, epochs=self.epoch)

        print('************* Start training *************')
        pipeline = make_pipeline(sequencer, padder, my_model)
        pipeline.fit(self.data[0], self.data[1])
        # save the model to disk
        filename = self.pref + 'models/' + self.ds + '/' + self.model + '_model.sav'
        pickle.dump(pipeline, open(filename, 'wb'))
        print('Training done, saving model and doing test')
        print('******************************************\n')

        print('Reloading trained model for validation')
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))

        # Computing interesting metrics/classification report
        pred = loaded_model.predict(self.data_valid[0])
        print(classification_report(self.data_valid[1], pred))

    def test(self):
        if self.model == 'RF':
            vec_filename = self.pref + 'models/' + self.ds + '/tfidf_vectorizer.pickle'
            vec = pickle.load(open(vec_filename, 'rb'))
            vectors = vec.transform(self.data[0])
            print('Vectorizer is ready')

            model_filename = self.pref + 'models/' + self.ds + '/RF_model.sav'
            rf_model = pickle.load(open(model_filename, 'rb'))
            print('Model is ready, start evaluation')

            # Computing interesting metrics/classification report
            pred = rf_model.predict(vectors)
            print(classification_report(self.data[1], pred))
        elif self.model == 'DNN':
            # load the model from disk
            filename = self.pref + 'models/' + self.ds + '/' + self.model + '_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            print('Model is ready, start evaluation')

            pred = loaded_model.predict(self.data[0])
            print(classification_report(self.data[1], pred))
