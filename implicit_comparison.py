import implicit
from implicit import als, evaluation
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz
from threadpoolctl import threadpool_limits


def load_data() -> csr_matrix:
    data_dir = Path("data")
    data = load_npz(data_dir / "user_artist_matrix.npz")
    assert type(data) is csr_matrix
    return data

def get_comparison(user_data: csr_matrix):
    model = als.AlternatingLeastSquares()
    model.fit(user_data)
    (train, test) = evaluation.train_test_split(user_data.tocoo(), 0.9)
    return evaluation.ndcg_at_k(model, train, test)


def main():
    threadpool_limits(1, "blas")
    data = load_data()
    print(get_comparison(data))


if __name__ == "__main__":
    main()