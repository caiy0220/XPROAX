import collections
from warnings import warn

from utils import *
from generator.model import *
from generator.vocab import Vocab
from generator.utils_gen import *
from generator.batchify import get_batches
from generator.meter import AverageMeter
from functools import partial
from sklearn.linear_model import Ridge
from deap import creator, tools, base, algorithms

# FOR XSPELLS
import decision_tree
from nltk.corpus import stopwords
from collections import Counter


class AutoencoderObj:
    def __init__(self, _str, _z):
        self.string = _str
        self.z = _z


class CounterInterpolation:
    def __init__(self, obj_from):
        self.obj_from = obj_from
        self.obj_to = None
        self.interpolation = []
        self.exemplars = []
        self.counter_exemplars = []
        self.proba = [[], []]

    def set_interpolation(self, interpolation, obj_to):
        self.interpolation = interpolation
        self.obj_to = obj_to

    def split_interpolation(self, model):
        if len(self.counter_exemplars) != 0:
            return
        _y_target = get_prediction(model, [self.obj_from.string])[0]

        for _obj in self.interpolation:
            _y, _score = get_prediction(model, [_obj.string], get_proba=True)
            if _y[0] != _y_target:
                self.counter_exemplars.append(_obj)
                self.proba[1].append(_score[0])
            else:
                self.exemplars.append(_obj)
                self.proba[0].append(_score[0])


"""
Local Explanator for creating explanation of decision made by a given black box

******************************************************************************    
The input texts of function encode, reconstruct, gen_neighbor should be a list
of sentences, which are split in to a list of words.
******************************************************************************
"""


class Explanator:
    def __init__(self, path, generator_path=None, black_box=None):
        """
        Parameters:
        -------
        path:   path to the configuration file of the explanator
        generator_path:     path to the generative model
        black_box:  path to the black box model
        """
        self.config = load_config(path)
        if self.config is None:
            log('Config file for explanator NOT exist', 3)
            exit()
        self.seed = self.config['seed']
        if self.seed >= 0:
            set_seed(self.seed)

        cuda_on = self.config['cuda_on'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_on else "cpu")

        self.vocab = self._get_vocab(generator_path)
        self.generator, self.args = self._get_model(generator_path)

        self.local_predictor = None
        self.corpus_strings = None
        self.corpus_rec = None
        self.corpus_z = None
        self.corpus_y = None
        self.metric = 'cosine'
        self.bb = black_box
        self.ppl = self.config['population']
        self.verbose = False

    '''
    *******************************************************
    *                Configuring explanator               *
    *******************************************************
    '''
    def set_metric(self, metric):
        """
        Parameters:
        -------
        metric: metric for distance measurement, including L1, L2 and cosine,
                cosine is favored
        """
        self.metric = metric

    def load_corpus(self, corpus_strings):
        """
        Parameters:
        -------
        corpus_strings: loading corpus for the approximation method
        """
        self.corpus_strings = np.array(corpus_strings)
        encoder_input_corpus = [t.split() for t in corpus_strings]
        self.corpus_z = self.encode(encoder_input_corpus)
        # self.corpus_y = self.get_bb_prediction(self.corpus_strings)
        rec = self.decode(self.corpus_z)
        self.corpus_rec = [' '.join(ws) for ws in rec]
        self.corpus_y = self.get_bb_prediction(self.corpus_rec)
        self.corpus_rec = np.array(self.corpus_rec)

    def _get_model(self, generator_path=None):
        if generator_path is None:
            generator_path = self.config['dir']
        ckpt = torch.load(os.path.join(generator_path, self.config['model_name']))
        args = ckpt['args']
        model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[args.model_type](self.vocab, args).to(self.device)
        model.load_state_dict(ckpt['model'])
        model.flatten()
        model.eval()
        return model, args

    def _get_vocab(self, generator_path=None):
        if generator_path is None:
            generator_path = self.config['dir']
        return Vocab(os.path.join(generator_path, self.config['vocab']))

    '''
    *******************************************************
    *                  Surrogate model                    *
    *******************************************************
    '''
    def train_regressor(self, x, y, weights=None):
        """
        Parameters
        -------
        x:  list of training data, containing target instance & generated neighbors
        y:  pred_proba of target class (only support single class for now)
        weights: distance based weights

        Returns
        -------
        res:    whether training succeeded
        """
        self._reset_local_predictor()
        self.local_predictor = Ridge(alpha=1, fit_intercept=True, random_state=self.seed)
        self.local_predictor.fit(x, y, sample_weight=weights)
        return 1

    def show_regressor_coef(self, label_idx=None):
        if label_idx is None:
            idx_li = list(range(len(self.local_predictor.coef_)))
            return sorted(zip(idx_li, self.local_predictor.coef_),
                          key=lambda _x: np.abs(_x[1]), reverse=True)
        else:
            idx_li = list(range(len(self.local_predictor.coef_[label_idx])))
            return sorted(zip(idx_li, self.local_predictor.coef_[label_idx]),
                          key=lambda _x: np.abs(_x[1]), reverse=True)

    '''
    *******************************************************
    *                Generating explanations              *
    *******************************************************
    '''
    def explain_instance(self, words_li, sur_model=0, num_other_words=10,
                         vocab_size_limit=200, forward_selection=False, log_f=None):
        """
        Parameters
        -------
        words_li:           list of ordered words from the target sentence
        sur_model:          type of surrogate model, 0=linear regressor, 1=decision tree, 2=naive bayes
        num_other_words:    number of important words to show, which are contained in neighbors but not in
                            the given sentence
        vocab_size_limit:   size limitation of local vocabulary
        forward_selection:  indexing method for surrogate model built in textual space
        log_f:              file for saving the generated exemplars for further use

        Returns
        -------
        res:    word-level explanation
        """
        pass

    def construct_neighborhood(self, texts, allow_duplicate=False):
        """
        Returns
        -------
        neigh_z:    latent representations of constructed neighborhood
        neigh_str:  texts of constructed neighborhood
        """
        pass

    '''
    *******************************************************
    *                 Experiments related                 *
    *******************************************************
    '''
    def explain_with_given_exemplars(self, exemplars, num_other_words=10,
                                     vocab_size_limit=200, forward_selection=False,
                                     lambda_factor=-1):
        """
        WARNING: Only for running the experiments
        Parameters
        -------
        exemplars:           list of ordered words from the target sentence
        num_other_words:    number of important words to show, which are contained in neighbors but not in
                            the given sentence
        vocab_size_limit:   size limitation of local vocabulary
        forward_selection:  indexing method for surrogate model built in textual space
        lambda_factor:     output exemplars and counter-exemplars when >= 0.

        Returns
        -------
        res:    word-level explanation
        """
        pass

    def evaluate_generator(self, sentences):
        """
        Evaluating the performance of generator
        Parameters
        -------
        sentences: test set for the reconstruction

        Returns
        -------
        meters: reconstruction loss
        """
        valid_batches, _ = get_batches(sentences, self.vocab, self.config['batch_size'], self.device)

        meters = collections.defaultdict(lambda: AverageMeter())
        with torch.no_grad():
            for inputs, targets in valid_batches:
                losses = self.generator.autoenc(inputs, targets)
                for k, v in losses.items():
                    meters[k].update(v.item(), inputs.size(1))
        loss = self.generator.loss({k: meter.avg for k, meter in meters.items()})
        meters['loss'].update(loss)
        return meters

    '''
    *******************************************************
    *                  Basic functions                    *
    *******************************************************
    '''
    def get_bb_prediction(self, x, c=None, get_proba=False):
        """
        Parameters
        -------
        x:  list of strings
        c:  classes
        get_proba:  return the pred_proba if is true

        Returns
        -------
        when get_proba=False: y_p
            y_p : array-like of shape (n_samples)
        when get_proba=True: [y_p, proba]
            y_p : array-like of shape (n_samples)
            proba: array-like of shape (n_samples, n_classes)
        """
        return get_prediction(self.bb, x, c, get_proba)

    def _get_nearest_idxs_from_corpus(self, z_target, count, avoid_label=None):
        z = self.corpus_z
        idxs = list(range(len(self.corpus_z)))
        sorted_list = sorting_neigh(z, idxs, self.metric, z_target=z_target)
        distances_sorted = {i: d for i, d in sorted_list}
        final_idxs, _ = zip(*list(distances_sorted.items()))

        res = []
        for idx in final_idxs:
            if self.corpus_y[idx] != avoid_label:
                res.append(idx)
            if len(res) >= count:
                break
        return res

    def show_neigh_stats(self, b):
        self.verbose = b

    def _get_dis_div(self, a):
        if not self.verbose:
            return
        if len(a) == 0:
            return
        dis = distance_neighbors(a)
        div = diversity_neighbors(a)
        log('Distance: {}, Diversity: {}'.format(dis, div))

    @staticmethod
    def _get_nearest_idxs(z, idxs, metric, count):
        """
        Returns
        -------
        list of idxs sorted by distance to the first element
        """
        if len(idxs) == 0:
            return []

        sorted_list = sorting_neigh(z, idxs, metric)
        distances_sorted = {i: d for i, d in sorted_list}
        final_idxs, final_dists = zip(*list(distances_sorted.items()))
        return list(final_idxs[:count])

    def _get_boundary_from_corpus(self, z_target, example_count, avoid_label=None):
        idxs = self._get_nearest_idxs_from_corpus(z_target, example_count, avoid_label)
        local_corpus_z = self.corpus_z[idxs]
        lower_bound = np.min(local_corpus_z, axis=0)
        upper_bound = np.max(local_corpus_z, axis=0)
        return lower_bound, upper_bound

    @staticmethod
    def _default_kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d**2) / kernel_width**2))

    def _reset_local_predictor(self):
        self.local_predictor = None

    def encode(self, sents):
        """
        Parameters
        -------
        sents: list of sentences

        Returns
        -------
        z_: latent vectors in ndarray
        """
        assert self.config['enc'] == 'mu' or self.config['enc'] == 'z'
        batches, order = get_batches(sents, self.vocab, self.config['batch_size'], self.device)
        z = []
        for inputs, _ in batches:
            mu, log_var = self.generator.encode(inputs)
            if self.config['enc'] == 'mu':
                zi = mu
            else:
                zi = reparameterize(mu, log_var)
            z.append(zi.detach().cpu().numpy())
        z = np.concatenate(z, axis=0)
        z_ = np.zeros_like(z)
        z_[np.array(order)] = z
        return z_

    def decode(self, z):
        """
        Parameters
        -------
        z: list of latent vectors

        Returns
        -------
        sents: list of sentences in words
        """
        z = np.array(z)
        z = z.astype('f')
        sents = []
        i = 0
        while i < len(z):
            zi = torch.tensor(
                z[i:i+self.config['batch_size']],
                device=self.device)
            outputs = self.generator.generate(zi, self.config['max_len'], self.config['dec']).t()
            for s in outputs:
                sents.append([self.vocab.idx2word[idx] for idx in s[1:]])
            i += self.config['batch_size']
        return strip_eos(sents)

    def reconstruct(self, sents):
        z = self.encode(sents)
        sents_rec = self.decode(z)
        return sents_rec

    def random_sampling(self, num):
        z = np.random.normal(size=(num, self.args.dim_z)).astype('f')
        sents = self.decode(z)
        return sents


class XPROAX(Explanator):
    def __init__(self, path, generator_path=None, black_box=None):
        super().__init__(path, generator_path, black_box)

    '''
    *******************************************************
    *                Generating explanations              *
    *******************************************************
    '''
    def explain_instance(self, words_li, sur_model=0, num_other_words=10,
                         vocab_size_limit=200, forward_selection=False, log_f=None):
        enc_input = [words_li]
        neigh_z, neigh_strs = self.construct_neighborhood(enc_input)
        gen_z = neigh_z[0]
        refer_strs = neigh_strs[0]

        gen_z = np.array(gen_z)

        if log_f is not None:
            for vec in gen_z:
                for v in vec:
                    log_f.write('{:.3f} '.format(v))
                log_f.write('\n')  # each vector in one line
            log_f.write('----\n')

        indexing = IndexedStrings(refer_strs,
                                  vocab_size_limit=vocab_size_limit,
                                  forward_selection=forward_selection)  # 100
        train_x = indexing.get_indexed()

        dists = cdist(gen_z, gen_z[0].reshape(1, -1), metric=self.metric).ravel() * 100

        kernel_width = 25
        # kernel_width = float(np.sqrt(np.shape(train_x)[1]) * 0.75)
        kernel = self._default_kernel
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        weights = kernel_fn(dists)

        # TODO: considering using other transparent models as surrogate model
        y, train_y = self.get_bb_prediction(refer_strs, get_proba=True)
        log('Reconstructed input: {}'.format(refer_strs[0]))
        log('Black box prediction: {}'.format(train_y[0]))
        bb_assign = [y[0], train_y[0]]
        self.train_regressor(train_x, train_y, weights)

        res = self.show_regressor_coef(bb_assign[0])
        target_w_li = words_li

        target_weights = []
        other_important_weights = []
        for w_p in res:  # w_p: word pair (index, weight)
            w = indexing.inverse_vocab[w_p[0]]
            if w in target_w_li:
                target_weights.append([w, w_p[1]])
            elif len(other_important_weights) < num_other_words:
                other_important_weights.append([w, w_p[1]])

        self.find_exemplars(gen_z[1:], y[1:], factor=0.5)
        return target_weights, other_important_weights

    def construct_neighborhood(self, texts, allow_duplicate=False):
        timer = MyTimer()
        input_size = len(texts)
        steps = self.config['intpl_steps']
        nearest_count = self.config['nearest_in_corpus']

        z = self.encode(texts)
        texts_rec = self.decode(z)

        if self.corpus_z is None:
            warn('Corpus is not set, using normal sampling instead')
            return None

        neigh_z = [[] for _ in range(input_size)]
        neigh_str = [[] for _ in range(input_size)]

        for _i in range(input_size):
            t_cost = timer.tiktok('epoch')
            if t_cost > 0:
                log('Time cost for last epoch: {:.2f} s'.format(t_cost), 0)
            _obj = AutoencoderObj(' '.join(texts_rec[_i]), z[_i])
            neigh_z[_i] += [_obj.z]
            neigh_str[_i] += [_obj.string]
            _y_target = self.get_bb_prediction([_obj.string])[0]

            _counter_idxs = self._get_nearest_idxs_from_corpus(_obj.z, nearest_count, _y_target)
            ref_z = self.corpus_z[_counter_idxs]
            rec_string = self.corpus_rec[_counter_idxs]

            # Start the first round interpolation
            intpls = []
            for _j in range(len(ref_z)):
                _obj_counter = AutoencoderObj(rec_string[_j], ref_z[_j])
                intpl = self.do_interpolation(_obj, _obj_counter, steps, allow_duplicate=allow_duplicate)
                intpls.append(intpl)
            # Interpolation done

            # Start filling the gap and pushing the boundary towards pivot point
            # obj_li, y_arr = self.do_approaching(_obj, intpls, steps, allow_duplicate=allow_duplicate)
            obj_li, y_arr = self.do_approaching_ea(_obj, intpls, steps,
                                                   max_epochs=50, stable_epochs=5,
                                                   num_crossover=nearest_count
                                                   )
            z_li = [neigh_obj.z for neigh_obj in obj_li]

            same_idx = np.where(y_arr == _y_target)[0]
            diff_idx = np.where(y_arr != _y_target)[0]
            log('#Factuals: {}, #Counterfactuals: {}'.format(len(same_idx), len(diff_idx)))

            closest_idx_li = self._get_nearest_idxs(z_li, same_idx, self.metric, self.ppl)
            closest_idx_li += self._get_nearest_idxs(z_li, diff_idx, self.metric, self.ppl)
            neigh_z[_i] += [obj_li[_idx].z for _idx in closest_idx_li]
            neigh_str[_i] += [obj_li[_idx].string for _idx in closest_idx_li]
        self._get_dis_div(neigh_z[0])
        return neigh_z, neigh_str

    def find_exemplars(self, z, y, factor=0.02, num=5):
        pivot_y = y[0]
        neighs_y = y[1:]
        pivot = z[0]
        neighs = z[1:]
        opposite_idx = np.where(neighs_y != pivot_y)[0]
        same_idx = np.where(neighs_y == pivot_y)[0]

        if len(opposite_idx) < num or len(same_idx) < num:
            return

        exemplars = self._find_exemplars(neighs[same_idx], pivot, factor, num)
        counters = self._find_exemplars(neighs[opposite_idx], pivot, factor, num)

        rec_exemplars = self.decode(exemplars)
        rec_counters = self.decode(counters)
        rec_exemplars = [' '.join(ws) for ws in rec_exemplars]
        rec_counters = [' '.join(ws) for ws in rec_counters]

        log('Exemplars:')
        for ins in rec_exemplars:
            log(ins)
        log('-----------')
        log('Counter exemplars:')
        for ins in rec_counters:
            log(ins)
        log('-----------')

    '''
    *******************************************************
    *               Neighborhood approximation            *
    *******************************************************
    '''
    def do_approaching_ea(self, obj_from, intpls, steps, max_epochs=50, stable_epochs=5,
                          num_crossover=20):
        """
        Use evolutionary algorithm to approach the boundary.
        Here, we use genetic algorithm to maintain the variety in each generation

        Parameters:
        ---------
        obj_from:       pivot point
        intpls:         list of interpolations from pivot to closest counter-exemplars
        steps:          number of steps to be taken in interpolations
        max_epochs:     limitation of iteration
        stable_epochs:  early stop if objective function does not get improved
        num_crossover:  number of crossovers for each generation

        Returns:
        ---------
        neigh_obj_li:   list of (counter-)exemplars
        y_tilde:        labels assigned by black box for the exemplars
        """
        candidate_list = []     # Neighborhood N
        parents_as_obj = []     # Counterfactual landmark C
        buff = []
        for intpl in intpls:
            intpl.split_interpolation(self.bb)
            parents_as_obj.append(intpl.counter_exemplars[0])   # Use closest counterfactual as landmark
            buff += intpl.exemplars[1:] + intpl.counter_exemplars
        self._get_dis_div([obj.z for obj in buff])
        candidate_list += buff
        dist = calc_distance_objs2obj(parents_as_obj, obj_from, self.metric)
        min_dist = min(dist)

        stable_count = 0
        epoch_count = 0
        total_stable_count = 0
        while epoch_count <= max_epochs:
            new_generations_as_obj = []
            buff = []
            idx_li = [i for i, _ in enumerate(parents_as_obj)]
            for i in range(num_crossover):
                # First-stage interpolation between landmarks for exploration
                p0, p1 = np.random.choice(idx_li, size=2, replace=False)
                intpl = self.do_interpolation(parents_as_obj[p0], parents_as_obj[p1], steps)
                intpl.split_interpolation(self.bb)

                qualified_children = intpl.exemplars[1:-1]
                best_children = [intpl.exemplars[0], intpl.exemplars[-1]]
                # Second-stage interpolation push approximation
                for obj_to in qualified_children:
                    intpl2pivot = self.do_interpolation(obj_from, obj_to, steps)
                    intpl2pivot.split_interpolation(self.bb)
                    best_children.append(intpl2pivot.counter_exemplars[0])
                    buff += intpl.exemplars[1:] + intpl.counter_exemplars

                if len(best_children) == 0:
                    continue
                dist = calc_distance_objs2obj(best_children, obj_from, self.metric)
                best_child_idx = np.argmin(dist)    # Use closest counterfactual for next epoch
                new_generations_as_obj.append(best_children[best_child_idx])
            candidate_list += buff
            self._get_dis_div([obj.z for obj in candidate_list])

            if len(new_generations_as_obj) <= 2:
                break

            dist = calc_distance_objs2obj(new_generations_as_obj, obj_from, self.metric)
            new_min = np.min(dist)
            if min_dist > new_min:
                min_dist = new_min
                stable_count = 0
            else:
                stable_count += 1
                total_stable_count += 1
            # Early stop
            if stable_count > stable_epochs:
                break

            # Update the new generation
            parents_as_obj = new_generations_as_obj
            epoch_count += 1
        log('Num of iterations: {}, num of no progress iterations: {}'.format(epoch_count, total_stable_count))

        neigh_obj_li = []
        saved_text = [self._remove_punctuation(obj_from.string)]
        for candidate in candidate_list:
            cleaned = self._remove_punctuation(candidate.string)
            if cleaned in saved_text:
                continue
            neigh_obj_li.append(candidate)
            saved_text.append(cleaned)
        string_li = [obj.string for obj in neigh_obj_li]
        y_tilde = self.get_bb_prediction(string_li)
        return neigh_obj_li, y_tilde

    def do_interpolation(self, obj_from, obj_to, steps, allow_duplicate=False):
        list_path_objs = []
        saved_path_texts = []   # Record saved text to avoid duplication

        # go along the path from target text to counter-exemplar
        z_on_path = interpolate(obj_from.z, obj_to.z, steps)
        rec_on_path = self.decode(z_on_path)
        rec_on_path = [' '.join(_ws) for _ws in rec_on_path]

        for _j in range(len(rec_on_path)):
            cleaned_str = self._remove_punctuation(rec_on_path[_j])     # Ignore punctuation while checking the duplication
            if not allow_duplicate and cleaned_str in saved_path_texts:
                continue
            _obj_on_path = AutoencoderObj(rec_on_path[_j], z_on_path[_j])
            # Recoding the neighboring point
            list_path_objs.append(_obj_on_path)
            saved_path_texts.append(cleaned_str)
        intpl = CounterInterpolation(obj_from)
        intpl.set_interpolation(list_path_objs, obj_to)
        return intpl

    def do_approaching(self, obj_from, intpls, steps, allow_duplicate=False):
        """ WARNING: DEPRECATED """
        pool_all_obj = [obj_from]
        pool_closest_obj = []

        cleaned_str = self._remove_punctuation(obj_from.string)
        save_text_s = [cleaned_str]
        ''' Find all counter-exemplars locate on the boundary '''
        for intpl in intpls:
            intpl.split_interpolation(self.bb)
            pool_closest_obj.append(intpl.counter_exemplars[0])

            buff = intpl.exemplars[1:] + intpl.counter_exemplars
            for obj in buff:
                cleaned_str = self._remove_punctuation(obj.string)
                if cleaned_str not in save_text_s:
                    pool_all_obj.append(obj)
                    save_text_s.append(cleaned_str)

        dist = calc_distance_objs2obj(pool_closest_obj, obj_from, self.metric)
        min_dist = min(dist)

        while len(pool_closest_obj) > 0:
            '''
            Exclusively do interpolation between points on local boundary,
            till the boundary cannot be pushed towards the given pivot point.
            '''
            idx_max = len(pool_closest_obj)
            pool_candidate_obj = []
            # find new counter exemplars on the interpolations between seeds
            for i in range(idx_max):
                for j in range(i+1, idx_max):
                    intpl = self.do_interpolation(pool_closest_obj[i], pool_closest_obj[j], steps, allow_duplicate=allow_duplicate)
                    intpl.split_interpolation(self.bb)
                    # counter-exemplars should have the same label as the seed counter-exemplars
                    pool_candidate_obj += intpl.exemplars[1:-1]     # Ignore two seeds
            dist = calc_distance_objs2obj(pool_candidate_obj, obj_from, self.metric)
            idx_closest = np.where(dist < min_dist * 1.5)[0]
            pool_candidate_obj = np.array(pool_candidate_obj)
            pool_candidate_obj = pool_candidate_obj[idx_closest]
            pool_candidate_obj = pool_candidate_obj[:30]

            pool_closest_obj = []
            for obj_to in pool_candidate_obj:   # pushing the boundary forward
                intpl = self.do_interpolation(obj_from, obj_to, steps, allow_duplicate=allow_duplicate)
                intpl.split_interpolation(self.bb)
                pool_closest_obj.append(intpl.counter_exemplars[0])
                for _obj in intpl.interpolation[1:]:
                    if _obj.string not in save_text_s:
                        pool_all_obj.append(_obj)
                        save_text_s.append(_obj.string)
            dist = calc_distance_objs2obj(pool_closest_obj, obj_from, self.metric)
            idx_closest = np.where(dist < min_dist)[0]
            pool_closest_obj = np.array(pool_closest_obj)
            pool_closest_obj = pool_closest_obj[idx_closest]

            min_dist = min(min(dist), min_dist)

        pool_all_text = [_obj.string for _obj in pool_all_obj]
        y_tilde = self.get_bb_prediction(pool_all_text)
        return pool_all_obj, y_tilde

    '''
    *******************************************************
    *                  Basic functions                    *
    *******************************************************
    '''
    def _find_exemplars(self, z, pivot, factor=0.02, num=5):
        dists = cdist(z, pivot.reshape(1, -1), metric=self.metric).ravel()
        directions = z - pivot

        idx = np.argmin(dists)
        exemplars = np.array([z[idx]])
        exemplars_directions = np.array([directions[idx]])

        z = np.delete(z, idx, 0)
        dists = np.delete(dists, idx, 0)
        directions = np.delete(directions, idx, 0)

        while len(exemplars) < num:
            if len(z) == 0:
                break
            scoring = self._exemplar_scoring(dists, directions,
                                             exemplars_directions,
                                             factor=factor)
            # find the closest point while considering the variety
            idx = np.argmin(scoring)
            exemplars = np.vstack((exemplars, z[idx]))
            exemplars_directions = np.vstack((exemplars_directions, directions[idx]))

            z = np.delete(z, idx, 0)
            dists = np.delete(dists, idx, 0)
            directions = np.delete(directions, idx, 0)
        return exemplars

    def _exemplar_scoring(self, dists, directions, e_directions, factor=0.02):
        scoring = []
        for i, v in enumerate(directions):
            diversity = cdist(e_directions, v.reshape(1, -1), metric=self.metric).ravel()
            sparsity_score = np.sum(diversity)/len(diversity)
            score_v = (1. - factor) * dists[i] - factor * sparsity_score
            scoring.append(score_v)
        return scoring

    @staticmethod
    def _remove_punctuation(src):
        tmp_src = re.sub('[{}]'.format(string.punctuation), " ", src)
        tmp_src = tmp_src.split()
        return ' '.join(tmp_src)

    '''
    *******************************************************
    *               Only for experiments                  *
    *******************************************************
    '''
    def explain_with_given_exemplars(self, exemplars, num_other_words=10,
                                     vocab_size_limit=200, forward_selection=False,
                                     lambda_factor=-1):
        neigh_z = exemplars
        neigh_z = np.array(neigh_z)
        neigh_str = self.decode(neigh_z)

        words_li = neigh_str[0]

        gen_z = neigh_z
        refer_strs = [' '.join(ws) for ws in neigh_str]
        # refer_strs = np.array(refer_strs)

        indexing = IndexedStrings(refer_strs,
                                  vocab_size_limit=vocab_size_limit,
                                  forward_selection=forward_selection)  # 100
        train_x = indexing.get_indexed()

        dists = cdist(gen_z, gen_z[0].reshape(1, -1), metric=self.metric).ravel() * 100

        kernel_width = 25
        # kernel_width = float(np.sqrt(np.shape(train_x)[1]) * 0.75)
        kernel = self._default_kernel
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        weights = kernel_fn(dists)

        y, train_y = self.get_bb_prediction(refer_strs, get_proba=True)
        log('Reconstructed input: {}'.format(refer_strs[0]))
        log('Black box prediction: {}'.format(train_y[0]))
        bb_assign = [y[0], train_y[0]]
        self.train_regressor(train_x, train_y, weights)

        res = self.show_regressor_coef(bb_assign[0])
        target_w_li = words_li

        target_weights = []
        other_important_weights = []
        for w_p in res:     # w_p: word pair (index, weight)
            w = indexing.inverse_vocab[w_p[0]]
            if w in target_w_li:
                target_weights.append([w, w_p[1]])
            elif len(other_important_weights) < num_other_words:
                other_important_weights.append([w, w_p[1]])

        if lambda_factor >= 0.:
            self.find_exemplars(gen_z, y, factor=lambda_factor)

        return target_weights, other_important_weights


class XPROAX_ABELE(Explanator):
    def __init__(self, path, generator_path=None, black_box=None):
        super().__init__(path, generator_path, black_box)
        self.ngen = self.config['generation_limit']
        self.mutpb = self.config['pm']
        self.cxpb = self.config['pc']
        self.tournsize = 3
        self.alpha1 = 0.5
        self.alpha2 = 0.5
        self.halloffame_ratio = 0.1

        self.logbook_idx = 0
        self.toolbox = None

    '''
    *******************************************************
    *                Generating explanations              *
    *******************************************************
    '''
    def explain_instance(self, words_li, sur_model=0, num_other_words=10,
                         vocab_size_limit=200, forward_selection=False, log_f=None):
        enc_input = [words_li]
        neigh_z, neigh_strs = self.construct_neighborhood(enc_input)

        gen_z = neigh_z[0]
        refer_strs = neigh_strs[0]

        gen_z = np.array(gen_z)

        if log_f is not None:
            for vec in gen_z:
                for v in vec:
                    log_f.write('{:.3f} '.format(v))
                log_f.write('\n')  # each vector in one line
            log_f.write('----\n')

        indexing = IndexedStrings(refer_strs,
                                  vocab_size_limit=vocab_size_limit,
                                  forward_selection=forward_selection)  # 100
        train_x = indexing.get_indexed()

        dists = cdist(gen_z, gen_z[0].reshape(1, -1), metric=self.metric).ravel() * 100
        kernel_width = 25
        kernel = self._default_kernel
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        weights = kernel_fn(dists)

        y, train_y = self.get_bb_prediction(refer_strs, get_proba=True)
        log('Reconstructed input: {}'.format(refer_strs[0]))
        log('Black box prediction: {}'.format(train_y[0]))
        bb_assign = [y[0], train_y[0]]
        self.train_regressor(train_x, train_y, weights)

        res = self.show_regressor_coef(bb_assign[0])
        target_w_li = words_li

        target_weights = []
        other_important_weights = []
        for w_p in res:  # w_p: word pair (index, weight)
            w = indexing.inverse_vocab[w_p[0]]
            if w in target_w_li:
                target_weights.append([w, w_p[1]])
            elif len(other_important_weights) < num_other_words:
                other_important_weights.append([w, w_p[1]])

        # self.find_exemplars(gen_z[1:], y[1:], factor=0.5)
        return target_weights, other_important_weights

    def construct_neighborhood(self, texts, allow_duplicate=False):
        timer = MyTimer()
        input_size = len(texts)

        z = self.encode(texts)

        neigh_z = [[] for _ in range(input_size)]
        neigh_str = [[] for _ in range(input_size)]

        for _i in range(input_size):
            t_cost = timer.tiktok('epoch')
            if t_cost > 0:
                log('Time cost for last epoch: {:.2f} s'.format(t_cost), 0)
            if self.toolbox is None:
                self.toolbox = self.setup_toolbox(z[0], self.fitness_equal, self.ppl)
            else:
                self.update_toolbox(z[0], self.fitness_equal, self.ppl)
            ppl_eq, qualified_eq, logbook_eq = self.fit(self.toolbox, self.ppl)
            lZ_eq = self._add_halloffame(ppl_eq, qualified_eq)

            self.update_toolbox(z[0], self.fitness_notequal, self.ppl)
            ppl_noteq, qualified_noteq, logbook_noteq = self.fit(self.toolbox, self.ppl)
            lZ_noteq = self._add_halloffame(ppl_noteq, qualified_noteq)

            # Recoding the stats
            filename = './logs/ABELE/' + str(self.logbook_idx) + '.pickle'
            pickle.dump([logbook_eq, logbook_noteq], open(filename, 'wb'))
            self.logbook_idx += 1

            lZ = [z[0]] + list(lZ_eq) + list(lZ_noteq)
            lStr = self.decode(lZ)
            neigh_z[_i] = lZ
            neigh_str[_i] = [' '.join(s) for s in lStr]
        self._get_dis_div(neigh_z[0])
        return neigh_z, neigh_str

    '''
    *******************************************************
    *                     EA related                      *
    *******************************************************
    '''
    def fitness_equal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = self._sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0

        s = self.decode(np.array([x]))
        s = [' '.join(_s) for _s in s]
        y = self.get_bb_prediction(s)[0]
        s1 = self.decode(np.array([x1]))
        s1 = [' '.join(_s) for _s in s1]
        y1 = self.get_bb_prediction(s1)[0]

        target_similarity_score = 1.0 - self.hamming(y, y1)
        target_similarity = self._sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = self._sigmoid(feature_similarity_score)

        s = self.decode(np.array([x]))
        s = [' '.join(_s) for _s in s]
        y = self.get_bb_prediction(s)[0]
        s1 = self.decode(np.array([x1]))
        s1 = [' '.join(_s) for _s in s1]
        y1 = self.get_bb_prediction(s1)[0]

        target_similarity_score = 1.0 - self.hamming(y, y1)
        target_similarity = 1.0 - self._sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def mutate(self, toolbox, x):
        while True:
            mutated = toolbox.clone(x)
            mutation_mask = np.random.choice([False, True], self.generator.args.dim_z, p=[1 - self.mutpb, self.mutpb])
            mutator = np.random.normal(size=self.generator.args.dim_z)
            mutated[mutation_mask] = mutator[mutation_mask]
            return mutated,

    def update_toolbox(self, x, evaluate, population_size):
        self.toolbox.register("feature_values", self.record_init, x)
        self.toolbox.register("evaluate", evaluate, x)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=population_size)

    def setup_toolbox(self, x, evaluate, population_size):
        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        return toolbox

    def fit(self, toolbox, population_size):
        halloffame_size = int(np.round(population_size * self.halloffame_ratio))

        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(halloffame_size, similar=np.array_equal)

        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", np.mean)
        # stats.register("min", np.min)
        # stats.register("max", np.max)
        stats = tools.Statistics(lambda ind: ind)
        stats.register("dist", distance_neighbors)
        stats.register("divers", diversity_neighbors)

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                                  ngen=self.ngen, stats=stats, halloffame=halloffame,
                                                  verbose=True)
        return population, halloffame, logbook

    @staticmethod
    def _add_halloffame(population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]

        index = np.max(np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist())
        fitness_value_thr = fitness_values[index]

        Z = list()
        for p in population:
            Z.append(p)
        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                Z.append(h)
        return np.array(Z)

    '''
    *******************************************************
    *                  Basic functions                    *
    *******************************************************
    '''
    @staticmethod
    def record_init(x):
        return x

    @staticmethod
    def clone(x):
        return pickle.loads(pickle.dumps(x))

    @staticmethod
    def _sigmoid(x, x0=0.5, k=10.0, L=1.0):
        return L / (1.0 + np.exp(-k * (x - x0)))

    @staticmethod
    def _validate_vector(u, dtype=None):
        # XXX Is order='c' really necessary?
        u = np.asarray(u, dtype=dtype, order='c').squeeze()
        # Ensure values such as u=1 and u=[1] still return 1-D arrays.
        u = np.atleast_1d(u)
        if u.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        return u

    def validate_weights(self, w, dtype=np.double):
        w = self._validate_vector(w, dtype=dtype)
        if np.any(w < 0):
            raise ValueError("Input weights should be all non-negative")
        return w

    def hamming(self, u, v, w=None):
        u = self._validate_vector(u)
        v = self._validate_vector(v)
        if u.shape != v.shape:
            raise ValueError('The 1d arrays must have equal lengths.')
        u_ne_v = u != v
        if w is not None:
            w = self.validate_weights(w)
        return np.average(u_ne_v, weights=w)

    '''
    *******************************************************
    *               Only for experiments                  *
    *******************************************************
    '''
    def explain_with_given_exemplars(self, exemplars, num_other_words=10,
                                     vocab_size_limit=200, forward_selection=False,
                                     lambda_factor=-1):
        neigh_z = exemplars
        neigh_z = np.array(neigh_z)
        neigh_str = self.decode(neigh_z)

        words_li = neigh_str[0]

        gen_z = neigh_z
        refer_strs = [' '.join(ws) for ws in neigh_str]
        # refer_strs = np.array(refer_strs)

        indexing = IndexedStrings(refer_strs,
                                  vocab_size_limit=vocab_size_limit,
                                  forward_selection=forward_selection)  # 100
        train_x = indexing.get_indexed()

        dists = cdist(gen_z, gen_z[0].reshape(1, -1), metric=self.metric).ravel() * 100

        kernel_width = 25
        kernel = self._default_kernel
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        weights = kernel_fn(dists)

        # y, proba = self.get_bb_prediction(refer_strs, get_proba=True)
        y, train_y = self.get_bb_prediction(refer_strs, get_proba=True)
        log('Reconstructed input: {}'.format(refer_strs[0]))
        log('Black box prediction: {}'.format(train_y[0]))
        bb_assign = [y[0], train_y[0]]
        self.train_regressor(train_x, train_y, weights)

        res = self.show_regressor_coef(bb_assign[0])
        target_w_li = words_li

        target_weights = []
        other_important_weights = []
        for w_p in res:     # w_p: word pair (index, weight)
            w = indexing.inverse_vocab[w_p[0]]
            if w in target_w_li:
                target_weights.append([w, w_p[1]])
            elif len(other_important_weights) < num_other_words:
                other_important_weights.append([w, w_p[1]])
        return target_weights, other_important_weights


class XPROAX_XSPELLS(Explanator):
    def __init__(self, path, generator_path=None, black_box=None):
        """
        Generation mode:
        0:  global random
        1:  constrained random
        """
        super().__init__(path, generator_path, black_box)
        self.mode = 0
        self.lower_bound = None
        self.upper_bound = None

    '''
    *******************************************************
    *                Generating explanations              *
    *******************************************************
    '''
    def explain_instance(self, words_li, sur_model=0, num_other_words=10,
                         vocab_size_limit=200, forward_selection=False, log_f=None):
        enc_input = [words_li]
        neigh_z, neigh_strs = self.construct_neighborhood(enc_input)
        gen_z = neigh_z[0]
        refer_strs = neigh_strs[0]

        if log_f is not None:
            for vec in gen_z:
                for v in vec:
                    log_f.write('{:.3f} '.format(v))
                log_f.write('\n')  # each vector in one line
            log_f.write('----\n')

        dists = cdist(gen_z, gen_z[0].reshape(1, -1), metric=self.metric).ravel() * 100
        kernel_width = 25
        # kernel_width = float(np.sqrt(np.shape(train_x)[1]) * 0.75)
        kernel = self._default_kernel
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        weights = kernel_fn(dists)

        if sur_model == 0:
            target_weights, other_important_weights = self.latent_xspells(words_li, gen_z, refer_strs, weights)
        else:
            target_weights, other_important_weights = self.textual_xspells(words_li, refer_strs, weights,
                                                                           vocab_size_limit=vocab_size_limit,
                                                                           forward_selection=False,
                                                                           num_other_words=num_other_words)
        return target_weights, other_important_weights

    def latent_xspells(self, target_wli, lz, refer_strs, weights, num_other_words=10):
        max_try = 5
        while self.train_decision_tree(lz, self.get_bb_prediction(refer_strs), weights) == 0:
            max_try -= 1
            if max_try <= 0:
                log('Training decision tree failed! EXIT', lvl=2)
                return [], []

        exemplars, counter_exemplars = self.get_exemplars(lz, refer_strs, 10, num_other_words)
        if exemplars is None:
            log('CANNOT derive explanation! EXIT', lvl=2)
            return [], []

        log('Exemplars:')
        only_show = 5
        for e in exemplars[0][:only_show]:
            log(e)
        log('Counter-Exemplars:')
        for c in counter_exemplars[0][:only_show]:
            log(c)

        # target_str = refer_strs[0]
        target_weights = []
        other_important_weights = []

        ptr_e = 0
        ptr_c = 0
        ws_e = list(exemplars[1])
        ws_c = list(counter_exemplars[1])
        while ptr_e < len(ws_e) or ptr_c < len(ws_c):
            if ptr_c >= len(ws_c):
                w = ws_e[ptr_e]
                target_dict = exemplars[1]
                ptr_e += 1
                sign = 1
            elif ptr_e >= len(ws_e) or exemplars[1][ws_e[ptr_e]] < counter_exemplars[1][ws_c[ptr_c]]:
                w = ws_c[ptr_c]
                target_dict = counter_exemplars[1]
                ptr_c += 1
                sign = -1
            else:
                w = ws_e[ptr_e]
                target_dict = exemplars[1]
                ptr_e += 1
                sign = 1

            if w in target_wli:
                target_weights.append([w, sign * target_dict[w]])
            else:
                other_important_weights.append([w, sign * target_dict[w]])

        return target_weights, other_important_weights

    def textual_xspells(self, target_wli, refer_strs, weights, vocab_size_limit=200,
                        forward_selection=False, num_other_words=10):
        indexing = IndexedStrings(refer_strs,
                                  vocab_size_limit=vocab_size_limit,
                                  forward_selection=forward_selection)  # 100
        train_x = indexing.get_indexed()

        y, train_y = self.get_bb_prediction(refer_strs, get_proba=True)
        log('Reconstructed input: {}'.format(refer_strs[0]))
        log('Black box prediction: {}'.format(train_y[0]))
        bb_assign = [y[0], train_y[0]]
        self.train_regressor(train_x, train_y, weights)
        res = self.show_regressor_coef(bb_assign[0])

        target_weights = []
        other_important_weights = []
        for w_p in res:  # w_p: word pair (index, weight)
            w = indexing.inverse_vocab[w_p[0]]
            if w in target_wli:
                target_weights.append([w, w_p[1]])
            elif len(other_important_weights) < num_other_words:
                other_important_weights.append([w, w_p[1]])
        return target_weights, other_important_weights

    def construct_neighborhood(self, texts, allow_duplicate=False):
        timer = MyTimer()
        input_size = len(texts)
        if self.corpus_z is None and self.mode == 0:
            warn('Corpus is not loaded')
            return None

        neigh_z = [[] for _ in range(input_size)]
        neigh_str = [[] for _ in range(input_size)]

        for _i in range(input_size):
            t_cost = timer.tiktok('epoch')
            if t_cost > 0:
                log('Time cost for last epoch: {:.2f} s'.format(t_cost), 0)

            if self.mode == 0:  # global random
                if self.lower_bound is None:
                    self.lower_bound, self.upper_bound = self._get_global_bound()
                neigh_z, neigh_str = self.generate_global_instance(texts, self.lower_bound, self.upper_bound)
            else:
                neigh_z, neigh_str = self.generate_neighbors(texts)
        self._get_dis_div(neigh_z[0])
        return neigh_z, neigh_str

    def get_exemplars(self, x, x_text, num_exemplar, num_words):
        exemplars, counter_exemplars = self._get_exemplars(x, x_text, num_exemplar)

        words_exemplar, words_exemplar_count = self.find_common_words(exemplars, num_words)
        if words_exemplar is None:
            warn('Failed to find common words in exemplars')
            return None, None
        words_exemplar_dict = dict(zip(words_exemplar, words_exemplar_count))

        words_counter, words_counter_count = self.find_common_words(counter_exemplars, num_words)
        if words_counter is None:
            warn('Failed to find common words in counter-exemplars')
            return None, None
        words_counter_dict = dict(zip(words_counter, words_counter_count))

        return [exemplars, words_exemplar_dict], [counter_exemplars, words_counter_dict]
    '''
    *******************************************************
    *                  Surrogate model                    *
    *******************************************************
    '''
    def train_decision_tree(self, x, y, weights):
        """
        Parameters
        -------
        x:  sentences presented in a ordered list of words,
            including target sentence and generated neighbors
        y:  labels (assigned by the black box in this case)
        weights: distance based weights

        Returns
        -------
        """
        self._reset_local_predictor()

        if len(y) < self.config['valid_neighs']:
            warn('Not enough neighborhoods')
            # returns 0 if training not succeed
            return 0

        x = np.array(x)
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

        sigma = 0.01
        x_ = (x - np.min(x)) / (np.max(x) - np.min(x) + sigma)
        dist = cdist(x_, x_[0].reshape(1, -1), metric=self.metric).ravel()

        invalid = np.where(np.isnan(dist))[0]
        for i in invalid[::-1]:
            x = np.delete(x, i, 0)
            y = np.delete(y, i, 0)
            dist = np.delete(dist, i, 0)

        self.local_predictor = decision_tree.learn_local_decision_tree(
            x, y, weights, [0, 1], prune_tree=False
        )

        y_tilde = self.local_predictor.predict([x[0]])
        if y_tilde[0] != y[0]:
            warn("Training failed, surrogate model assign wrong labels to target instance, re-training DT")
            return 0
        return 1

    '''
    *******************************************************
    *              Neighborhood construction              *
    *******************************************************
    '''
    def generate_global_instance(self, texts, lower_bound, upper_bound, allow_duplicate=False):
        """
        Randomly sampling in the entire latent space,
        the boundary of latent space is detected by the training corpus
        """
        input_size = len(texts)
        max_attempts = self.config['max_attempts']
        random_sents = self.config['random_sents']

        z = self.encode(texts)
        texts_rec = self.decode(z)
        neigh_z = [[] for _ in range(input_size)]
        neigh_strs = [[] for _ in range(input_size)]

        for i in range(input_size):
            _attempts = 0

            tmp_zl = [z[i]]
            tmp_strs = [' '.join(texts_rec[i])]

            while len(tmp_strs) < random_sents and _attempts < max_attempts:
                cp = self._get_random_instance(lower_bound, upper_bound)
                cp = np.float32(cp)
                cp_rec = self.decode([cp])[0]
                cp_rec = ' '.join(cp_rec)
                if allow_duplicate or cp_rec not in tmp_strs:
                    tmp_zl.append(cp)
                    tmp_strs.append(cp_rec)
                _attempts += 1
                log('Generated Neighbors: {}, Attempts: {}'.format(
                    len(tmp_strs), _attempts), end='\r')

            y_tilde = self.get_bb_prediction(tmp_strs)
            y_target = y_tilde[0]
            same_idx = np.where(y_tilde == y_target)[0]
            diff_idx = np.where(y_tilde != y_target)[0]
            closest_idx_li = self._get_nearest_idxs(tmp_zl, same_idx, self.metric, self.ppl)
            closest_idx_li += self._get_nearest_idxs(tmp_zl, diff_idx, self.metric, self.ppl)
            neigh_z[i] += [tmp_zl[idx] for idx in closest_idx_li]
            neigh_strs[i] += [tmp_strs[idx] for idx in closest_idx_li]

        return neigh_z, neigh_strs

    def generate_neighbors(self, texts, allow_duplicate=False, balance_ratio=1., model=None, dynamic=False):
        """
        Randomly sampling within a sphere locates in latent space,
        the radius of the sphere is configured in configuration file
        """
        input_size = len(texts)
        max_attempts = self.config['max_attempts']
        random_sents = self.config['random_sents']
        ranging = self.config['xspells_range']
        expand_rate = self.config['xspells_expand_rate']
        expand_suppressor = 0.002

        if model is None:
            model = self.bb

        if dynamic:
            def update_radius(r, attempts, alpha):
                return r * np.exp(attempts*alpha)
        else:
            def update_radius(r, attempts, alpha):
                return r

        z = self.encode(texts)
        texts_rec = self.decode(z)

        neigh_z = [[] for _ in range(input_size)]
        neigh_strs = [[] for _ in range(input_size)]

        for i in range(input_size):
            class_count = [1., 1.]  # Avoid DividedByZero error
            neigh_z[i].append(z[i])
            neigh_strs[i].append(' '.join(texts_rec[i]))

            _attempts = 0
            _failed_attempts = 0

            while len(neigh_strs[i]) < random_sents and _attempts < max_attempts:
                dynamic_ranging = update_radius(ranging, _failed_attempts, expand_suppressor * expand_rate)
                cp = np.copy(z[i])
                cp = self._get_neighbor(cp, -dynamic_ranging, dynamic_ranging)

                cp_rec = self.decode([cp])[0]
                cp_rec = ' '.join(cp_rec)
                if allow_duplicate or cp_rec not in neigh_strs[i]:
                    if self.check_balance(balance_ratio, class_count, model, cp_rec):
                        assigned = get_prediction(model, [cp_rec])[0]
                        class_count[assigned] += 1
                        neigh_z[i].append(cp)
                        neigh_strs[i].append(cp_rec)
                else:
                    _failed_attempts += 1
                _attempts += 1
                log('Generated Neighbors: {}, Attempts: {}'.format(
                    len(neigh_z[i]), _attempts), end='\r')
        return neigh_z, neigh_strs

    '''
    *******************************************************
    *                  Basic functions                    *
    *******************************************************
    '''
    def _get_exemplars(self, x, x_text, num_exemplar):
        # exemplars are found mainly based on their distances to pivot point in latent space
        x = np.array(x)
        x_text = np.array(x_text)
        y_tilde = self.local_predictor.predict(x)
        opposite_idx = np.where(y_tilde != y_tilde[0])[0]

        leaf_id = self.local_predictor.apply(x)
        same_leaf_idx = np.where(leaf_id == leaf_id[0])[0]

        counter_exemplars_idx = self._get_nearest_idxs(x, opposite_idx, self.metric, num_exemplar)
        counter_exemplars = x_text[counter_exemplars_idx]

        if len(same_leaf_idx) < num_exemplar:
            print('Not enough exemplars in the leaf, will find by distance instead...', len(same_leaf_idx))
            required_num = num_exemplar - len(same_leaf_idx)
            exemplars_idx = list(same_leaf_idx)
            occupied = set(exemplars_idx)

            same_idx = np.where(y_tilde == y_tilde[0])[0]
            same_idx = list(set(same_idx) - occupied)
            appended_idx = self._get_nearest_idxs(x, same_idx, self.metric, required_num)

            # the closest instance in exemplars will of course be the target itself, and we should skip it
            exemplars_idx += appended_idx
        else:
            exemplars_idx = np.random.choice(same_leaf_idx, size=num_exemplar, replace=False)

        exemplars = x_text[exemplars_idx]
        return exemplars, counter_exemplars

    @staticmethod
    def find_common_words(texts, n):
        if not isinstance(texts[0], list):
            texts = [text.split() for text in texts]
        stop_words = set(stopwords.words('english'))
        stop_words.add('...')
        stop_words.add('<end>')
        stop_words.add('<unk>')
        stop_words.add('_num_')
        stop_words.add('film')
        stop_words.add('movie')
        for punctuation in string.punctuation:
            stop_words.add(punctuation)

        filtered = []

        # tf
        for text in texts:
            filtered += [w for w in text if w not in stop_words]

        try:
            common_words, words_count = zip(*Counter(filtered).most_common())
        except ValueError:
            warn('Cannot find common words, is the corpus empty after removing stop words?')
            return None, None
        relative_count = np.array(words_count) / len(filtered)
        return common_words[:n], relative_count[:n]

    @staticmethod
    def check_balance(balance_ratio, class_count, model, cp_rec):
        if balance_ratio >= 1.:
            return True
        assigned = get_prediction(model, [cp_rec])[0]
        class_ratio = class_count[assigned] / (class_count[0] + class_count[1])
        if class_ratio > balance_ratio:
            return False
        return True

    @staticmethod
    def _get_neighbor(z, lower_bound=-0.5, upper_bound=0.5):
        rng = upper_bound - lower_bound
        shift_vec = np.random.random(np.shape(z))
        shift_vec = shift_vec * rng + lower_bound
        z += shift_vec
        return z

    @staticmethod
    def _get_random_instance(lower_bound, upper_bound):
        dist_bound = upper_bound - lower_bound
        z = np.random.random(np.shape(lower_bound))
        z = z*dist_bound + lower_bound
        return z

    def set_generation_mode(self, mode):
        self.mode = mode

    def _get_global_bound(self):
        lower_bound = np.min(self.corpus_z, axis=0)
        upper_bound = np.max(self.corpus_z, axis=0)
        return lower_bound, upper_bound

    '''
    *******************************************************
    *               Only for experiments                  *
    *******************************************************
    '''
    def explain_with_given_exemplars(self, exemplars, num_other_words=10,
                                     vocab_size_limit=200, forward_selection=False,
                                     lambda_factor=-1, sur_model=0):
        neigh_z = exemplars
        neigh_z = np.array(neigh_z)
        neigh_str = self.decode(neigh_z)

        refer_strs = [' '.join(ws) for ws in neigh_str]

        dists = cdist(neigh_z, neigh_z[0].reshape(1, -1), metric=self.metric).ravel() * 100
        kernel_width = 25
        kernel = self._default_kernel
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        weights = kernel_fn(dists)

        if sur_model == 0:
            target_weights, other_important_weights = self.latent_xspells(neigh_str[0], neigh_z, refer_strs, weights)
        else:
            target_weights, other_important_weights = self.textual_xspells(neigh_str[0], refer_strs, weights,
                                                                           vocab_size_limit=vocab_size_limit,
                                                                           forward_selection=False,
                                                                           num_other_words=num_other_words)
        return target_weights, other_important_weights
