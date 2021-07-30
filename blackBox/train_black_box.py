from bb_creator import BlackBoxCreator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-mode', metavar='n', default=0,
                    help='1/0 indicates training/testing the black box')
parser.add_argument('-workspace', default='XPROAX',
                    help='name of the workspace')
parser.add_argument('-ds', metavar='Dataset', required=True,
                    help='name of training set')
parser.add_argument('-model', metavar='model', required=True,
                    help='model type')      # RF, DNN
parser.add_argument('-epoch', type=int, default=50,
                    help='max epoch for training DNN')      # RF, DNN


if __name__ == '__main__':
    args = parser.parse_args()
    bb_obj = BlackBoxCreator(args)
