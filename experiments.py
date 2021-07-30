import argparse
from utils import log

if __name__ == '__main__':
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # -mode 0 -ds yelp
    # -mode 1 -ds yelp -model RF -method XPROAX -thresh 0.1
    # -mode 2

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True, help='0: reconstruct loss; 1: effectiveness; 2: stability')
    parser.add_argument('-ds', help='name of training set')

    """ Args for effectiveness """
    parser.add_argument('-model', default='RF', help='model type: RF, DNN')
    parser.add_argument('-method', help='name of method: XPROAX, XSPELLS, LIME, BASELINE, ABELE')
    parser.add_argument('-num', type=int, help='number of testing sentences from each class, max. 2000')
    parser.add_argument('-thresh', default=0.1, type=float, help='threshold defining important components')
    parser.add_argument('-vocab_size', default=200, type=int, help='Size of vocabulary for surrogate model')

    """ Optional """
    parser.add_argument('-surrogate', default=0, type=int, help='Currently only XSPELLS support different surrogate model, 0=LR, 1=DT')
    parser.add_argument('-workspace', default='XPROAX', help='name of the workspace')

    args = parser.parse_args()
    log(args)
    if args.mode == '0':
        from experiments.reconstruction_loss import main
        main(args)
    elif args.mode == '1':
        from experiments.effectiveness import main
        main(args)
    elif args.mode == '2':
        from experiments.stability import main
        main(args)
