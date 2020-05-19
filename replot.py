import setup
import pickle
from os.path import join

if __name__ == '__main__':
    with open(join('figures', 'raw.pickle'), 'rb') as fp:
        pickle.load(fp).plots()
