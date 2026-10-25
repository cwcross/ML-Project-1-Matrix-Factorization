# Imports
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import pandas as pd
from pathlib import Path


NUM_USERS = 20000 # the number of users to filter down to


def read_data() -> pd.DataFrame:
    filename = Path("raw_data") / "lastfm_360k_dataset.tsv"
    if filename.exists():
        return pd.read_csv(filename, sep='\t', header=None, names=["user_sha1", "artist_mbid", "artist", "play_count"])
    else:
        raise FileNotFoundError("You must download the data to raw_data/lastfm_360k_dataset.tsv from the link http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz")


def basic_cleanup(fm_df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = fm_df.copy()
    # Set column types
    cleaned_df['user_sha1'] = cleaned_df['user_sha1'].astype('category')
    cleaned_df['artist_mbid'] = cleaned_df['artist_mbid'].astype('category')
    return cleaned_df


def get_users(fm_df: pd.DataFrame) -> pd.Series:
    all_users = fm_df["user_sha1"].drop_duplicates()
    return all_users


def filter_sufficient_data(df: pd.DataFrame) -> pd.DataFrame:
    min_listeners = 5
    min_artists = 5

    filtered = df.copy()

    # guard against endless loop if data is flawed in some way
    # should never be reached, normal data filters in at most a few rounds
    max_iterations = 1000 
    i = 0
    while i < max_iterations:
        # value counts: listeners per artist
        artist_counts = filtered['artist_mbid'].value_counts()
        # value counts: artists per user
        user_counts = filtered['user_sha1'].value_counts()
        # boolean mask: identifies rows with artists with enough listeners
        enough_listeners = filtered['artist_mbid'].map(artist_counts) >= min_listeners
        # boolean mask: identifies rows with users with enough artists
        enough_artists = filtered['user_sha1'].map(user_counts) >= min_artists

        # if every row meets both criteria (boolean mask intersection is all true), stop filtering loop
        if (enough_listeners & enough_artists).all():
            filtered['user_sha1'] = filtered['user_sha1'].cat.remove_unused_categories()
            filtered['artist_mbid'] = filtered['artist_mbid'].cat.remove_unused_categories()
            return filtered
        filtered = filtered[enough_listeners & enough_artists].copy()
        filtered['user_sha1'] = filtered['user_sha1'].cat.remove_unused_categories()
        filtered['artist_mbid'] = filtered['artist_mbid'].cat.remove_unused_categories()
        i += 1
    # raise an error if max iterations exceeded
    raise RuntimeError("The data couldn't be filtered, check that the right dataset is being used")


def get_selection(fm_df: pd.DataFrame, all_users: pd.Series, num_users: int) -> pd.DataFrame:
    rng = np.random.default_rng(1235)
    order = rng.permutation(len(all_users))
    all_users.index = order
    sorted_users = all_users.sort_index()
    selected_users = sorted_users[0:num_users]
    fm_df_subset = fm_df[fm_df["user_sha1"].isin(selected_users)]
    return fm_df_subset


def get_mbid_artist_map(fm_df: pd.DataFrame) -> dict:
    mbid_artist_map = dict(zip(fm_df['artist_mbid'], fm_df['artist']))
    return mbid_artist_map


def consolidate_rows(fm_df_subset: pd.DataFrame) -> pd.DataFrame:
    # create group for each user-artist_mbid pair
    grouped_df = fm_df_subset.groupby(['user_sha1', 'artist_mbid'], observed=True)
    # consolidate each group into a single row in a df, summing playcount
    consolidated_df = grouped_df['play_count'].sum().reset_index()
    return consolidated_df


def create_sparse_matrix(consolidated_df: pd.DataFrame, mbid_artist_map: dict) -> tuple[csr_matrix, dict[str, pd.Index]]:
    # Convert to categoricals to get integer codes for each user/artist
    user_cat = consolidated_df['user_sha1'].astype('category')
    artist_cat = consolidated_df['artist_mbid'].astype('category')

    user_idx = user_cat.cat.codes
    artist_idx = artist_cat.cat.codes

    # Build the sparse matrix: rows = users, columns = artists
    sparse_matrix = csr_matrix(
        (consolidated_df['play_count'], (user_idx, artist_idx)),
        shape=(user_cat.cat.categories.size, artist_cat.cat.categories.size)
    )

    # Keep the labels so you can map back
    users = user_cat.cat.categories
    artist_ids = artist_cat.cat.categories
    artist_labels = pd.Index([mbid_artist_map[artist] for artist in artist_ids])

    mappings = {
        "user_sha1_mappings": users,
        "artist_mbid_mappings": artist_ids,
        "artist_mappings": artist_labels,
    }

    return sparse_matrix, mappings


def save_tidied_data(sparse_matrix: csr_matrix, mappings: dict[str, pd.Index]) -> None:
    data_dir = Path("data/tidied_360k")
    save_npz(data_dir / "user_artist_matrix.npz", sparse_matrix)
    np.save(data_dir / "user_sha1_mappings.npy", mappings["user_sha1_mappings"].to_numpy(), allow_pickle=True)
    np.save(data_dir / "artist_mbid_mappings.npy", mappings["artist_mbid_mappings"].to_numpy(), allow_pickle=True)
    np.save(data_dir / "artist_mappings.npy", mappings["artist_mappings"].to_numpy(), allow_pickle=True)


def main():
    fm_df = read_data()
    cleaned_df = basic_cleanup(fm_df)
    fm_df_subset = filter_sufficient_data(cleaned_df)
    all_users = get_users(fm_df_subset)
    fm_df_subset = get_selection(fm_df_subset, all_users, NUM_USERS)
    mbid_artist_map = get_mbid_artist_map(fm_df_subset)
    fm_df_subset = fm_df_subset.drop(columns=['artist'])
    consolidated_df = consolidate_rows(fm_df_subset)
    # filter again, cutting some rows may have led to artists or users with insufficient data
    consolidated_df = filter_sufficient_data(consolidated_df)
    (sparse_matrix, mappings) = create_sparse_matrix(consolidated_df, mbid_artist_map)
    save_tidied_data(sparse_matrix, mappings)


if __name__ == "__main__":
    main()