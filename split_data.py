from scipy.sparse import csr_matrix, load_npz, save_npz
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/tidied_360k")
SPLITS = {
    "train": 0.7,
    "validation": 0.2,
    "test": 0.1
}

def load_tidied_data() -> csr_matrix:
    sparse_matrix = load_npz(DATA_DIR / "user_artist_matrix.npz")
    assert isinstance(sparse_matrix, csr_matrix)
    return sparse_matrix


def load_artist_mappings() -> tuple[np.ndarray, np.ndarray]:
    artist_mbids = np.load(DATA_DIR / "artist_mbid_mappings.npy", allow_pickle=True)
    artists = np.load(DATA_DIR / "artist_mappings.npy", allow_pickle=True)
    return artist_mbids, artists


def isolate_split(matrix: csr_matrix, labels: np.ndarray, label: int) -> csr_matrix:
    # copy the sparse matrix
    split = matrix.copy()
    split.data[labels != label] = 0 # ignore cells from other splits
    split.eliminate_zeros() # remove those cells
    return split


def split_data(sparse_matrix: csr_matrix) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    rng = np.random.default_rng(1235)

    split_labels = np.zeros(sparse_matrix.nnz, dtype=int)

    # iterate through each user in the sparse matrix
    for u in range(sparse_matrix.shape[0]):
        # in compressed matrix data, finds all the non-zero data corresponding to this row
        start, end = sparse_matrix.indptr[u], sparse_matrix.indptr[u + 1]
        # number of artists in the data
        num_artists = end - start
        # random order
        order = rng.permutation(num_artists)
        # store the number of artists in each split
        num_artists_validation = int(SPLITS["validation"] * num_artists)
        num_artists_test = int(SPLITS["test"] * num_artists)
        # assign labels for validation and test (train is zero by default from np.zeros)
        for i in range(num_artists_validation):
            split_labels[start + order[i]] = 1
        for i in range(num_artists_test):
            split_labels[start + order[num_artists_validation + i]] = 2

    train = isolate_split(sparse_matrix, split_labels, 0)
    validation = isolate_split(sparse_matrix, split_labels, 1)
    test = isolate_split(sparse_matrix, split_labels, 2)

    return train, validation, test


def filter_missing_artists(train: csr_matrix, validation: csr_matrix, test: csr_matrix) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    (artist_mbid_mappings, artist_name_mappings) = load_artist_mappings()
    # filter for only artists represented in training data
    artists_to_keep = np.nonzero(train.sum(axis=0))[1]
    # dropping artists messes up the labels for artist column indices, so we need to fix that
    artist_mbid_mappings = artist_mbid_mappings[artists_to_keep]
    artist_name_mappings = artist_name_mappings[artists_to_keep]
    # save new mappings
    np.save(DATA_DIR / "artist_mbid_mappings.npy", artist_mbid_mappings, allow_pickle=True)
    np.save(DATA_DIR / "artist_mappings.npy", artist_name_mappings, allow_pickle=True)
    # return filtered matrices
    return train[:, artists_to_keep], validation[:, artists_to_keep], test[:, artists_to_keep]


def save_split_data(train: csr_matrix, validation: csr_matrix, test: csr_matrix) -> None:
    save_npz(DATA_DIR / "train.npz", train)
    save_npz(DATA_DIR / "validation.npz", validation)
    save_npz(DATA_DIR / "test.npz", test)


def main():
    sparse_matrix = load_tidied_data()
    (train, validation, test) = split_data(sparse_matrix)
    (train, validation, test) = filter_missing_artists(train, validation, test)
    save_split_data(train, validation, test)


if __name__ == "__main__":
    main()