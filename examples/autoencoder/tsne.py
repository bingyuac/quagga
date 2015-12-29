import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold


if __name__ == '__main__':
    # with h5py.File('075drop_codes.hdf5') as h5_file:
    # with h5py.File('095drop_codes.hdf5') as h5_file:
    with h5py.File('red_drop_codes.hdf5') as h5_file:
        codes = h5_file['codes'][...][:6000]
    manifold_learner = manifold.TSNE(n_components=2, random_state=42, init='pca')
    red_codes = manifold_learner.fit_transform(codes)
    plt.scatter(red_codes[:, 0], red_codes[:, 1], alpha=0.6)
    plt.savefig('manifold.pdf', dpi=900)