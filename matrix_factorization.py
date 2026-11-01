# Import necessary libraries here
from dataclasses import dataclass
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from typing import Iterator

@dataclass
class DatasetSplits:
    train: csr_matrix
    validation: csr_matrix
    test: csr_matrix
    user_ids: np.ndarray
    artist_mbids: np.ndarray
    artist_names: np.ndarray


def load_lastfm_dataset() -> DatasetSplits:
    """
    Function to import data from a given file path.
    
    Parameters:
    file_path (str): The path to the data file.
    
    Returns:
    data: The imported data.
    """
    # Implement the logic to read the data from the file
    data_dir = Path("data")

    dataset = DatasetSplits(
        train=load_npz(data_dir / "train.npz"),
        validation=load_npz(data_dir / "validation.npz"),
        test=load_npz(data_dir / "test.npz"),
        user_ids=np.load(data_dir / "user_sha1_mappings.npy", allow_pickle=True),
        artist_mbids=np.load(data_dir / "artist_mbid_mappings.npy", allow_pickle=True),
        artist_names=np.load(data_dir / "artist_mappings.npy", allow_pickle=True)
    )
    return dataset


def overlap_count(A: csr_matrix, B: csr_matrix) -> int:
    """
    Function to count the overlaps in sparse matrices.
    
    Parameters:
    A (csr_matrix): The first sparse matrix to compare.
    B (csr_matrix): The second sparse matrix to compare.
    
    Returns:
    int
    """
    # Check by multiplying the matrices
    # Only left with a nonzero entry if matching locations are nonzero
    # Simplify multiplication by converting to ones in small ints
    A_binary = A.copy()
    B_binary = B.copy()

    A_binary.data = np.ones_like(A_binary.data, dtype=np.int8)
    B_binary.data = np.ones_like(B_binary.data, dtype=np.int8)

    return A_binary.multiply(B_binary).nnz


def validate(data: DatasetSplits) -> None:
    """
    Function to validate the dataset splits.
    
    Parameters:
    data (DatasetSplits): The dataset splits to validate.
    
    Returns:
    None
    """
    # Implement the logic to validate the dataset splits

    # First check: all splits should be the same shape
    if data.train.shape != data.validation.shape or data.train.shape != data.test.shape:
        raise RuntimeError("Data splits have different shapes")
    # Second check: no leakage
    # there should be no overlap in the matrices
    splits = {"train": data.train, "validation": data.validation, "test": data.test}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if overlap_count(splits[a], splits[b]) != 0:
            raise RuntimeError("Data splits are not pairwise disjoint")      
    return   




def build_known_positive_mask(data: DatasetSplits) -> csr_matrix:
    """
    Function to build a known positive mask from the dataset splits.

    Check CSR format; equal matrix shapes; row count equals len(user_ids); column count equals both artist mapping lengths; 
    finite, non-negative stored counts; no stored zeros; no duplicate sparse coordinates; 
    consistent index/mapping alignment. Check the required train-user and train-artist coverage for the chosen warm-start protocol.
    
    Parameters:
    data (DatasetSplits): The dataset splits to build the mask from.
    
    Returns:
    csr_matrix: The known positive mask.
    """
    splits = [data.train, data.validation, data.test]

    # Check CSR format
    for matrix in splits:
        if not isinstance(matrix, csr_matrix):
            raise TypeError("All splits must be CSR matrices")

    # Check dimensions
    shapes = [matrix.shape for matrix in splits]

    if len(set(shapes)) != 1:
        raise ValueError("Dataset splits have different shapes")

    n_users, n_artists = data.train.shape

    if n_users != len(data.user_ids):
        raise ValueError("User mapping does not match matrix rows")

    if n_artists != len(data.artist_mbids):
        raise ValueError("Artist MBID mapping does not match matrix columns")

    if n_artists != len(data.artist_names):
        raise ValueError("Artist name mapping does not match matrix columns")

    processed = []

    for matrix in splits:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()

        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("Interaction counts must be finite")

        if np.any(matrix.data < 0):
            raise ValueError("Interaction counts cannot be negative")

        processed.append(matrix)

    # Check that splits are disjoint
    for i in range(len(processed)):
        for j in range(i + 1, len(processed)):
            if overlap_count(processed[i], processed[j]) != 0:
                raise ValueError("Dataset splits overlap")

    # Check warm-start coverage
    train = processed[0]

    user_coverage = np.diff(train.indptr) > 0
    artist_coverage = np.asarray(train.getnnz(axis=0)).ravel() > 0

    if not np.all(user_coverage):
        raise ValueError("Some users have no training interactions")

    if not np.all(artist_coverage):
        raise ValueError("Some artists have no training interactions")

    # Union of all known positives
    known_positive = processed[0] + processed[1] + processed[2]

    known_positive.data[:] = 1
    known_positive = known_positive.tocsr()
    known_positive.sort_indices()

    return known_positive



def extract_positive_interactions(matrix: csr_matrix, ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input: One CSR split matrix containing play counts.

    Output: (users, artists, play_counts) aligned 1-D arrays. Each output row represents one nonzero matrix coordinate.

    Specification: Preserve the sparse matrix's row/column indices and corresponding count. Do not densify. 
    Ensure zeros have been eliminated and duplicate coordinates summed before extraction.
    """
    # Implement the logic to extract positive interactions
    assert matrix.has_canonical_format, f"matrix has duplicates/unsorted indices"
    assert not (matrix.data == 0).any(), f"matrix has explicit zeros"

    coo = matrix.tocoo()
    users = coo.row
    artists = coo.col
    play_counts = coo.data

    return users, artists, play_counts



def sample_negative_items(positive_users: np.ndarray, known_positive: csr_matrix, n_artists: int, negatives_per_positive: int, rng: np.random.Generator) -> np.ndarray:
    """
    Input:
    - positive_users: 1-D array of user indices corresponding to positive interactions.
    - known_positive: CSR matrix of known positive interactions.
    - n_artists: Total number of artists in the dataset.
    - negatives_per_positive: Number of negative samples to generate per positive interaction.
    - rng: Random number generator.

    Output: Integer array of shape (n_positive, negatives_per_positive) containing sampled artist indices.
    """

    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive cannot be negative")

    if n_artists < 0:
        raise ValueError("n_artists cannot be negative")

    if known_positive.shape[1] != n_artists:
        raise ValueError("n_artists does not match known_positive shape")

    # Ensure we are working with a clean CSR matrix
    known = known_positive.tocsr(copy=True)
    known.sum_duplicates()
    known.eliminate_zeros()
    known.sort_indices()

    positive_users = np.asarray(positive_users, dtype=np.int64)

    if positive_users.ndim != 1:
        raise ValueError("positive_users must be a 1D array")

    n_positive = len(positive_users)

    # Prepare output array
    negative_artists = np.empty(
        (n_positive, negatives_per_positive),
        dtype=np.int64
    )

    if n_positive == 0 or negatives_per_positive == 0:
        return negative_artists

    # Check that all user indices are valid
    if np.any(positive_users < 0) or np.any(positive_users >= known.shape[0]):
        raise IndexError("Invalid user index in positive_users")

    # Sort users so we can process all interactions belonging
    # to the same user together.
    order = np.argsort(positive_users, kind="stable")
    sorted_users = positive_users[order]

    unique_users, starts, counts = np.unique(
        sorted_users,
        return_index=True,
        return_counts=True
    )

    # Process each unique user
    for user_idx, start, count in zip(unique_users, starts, counts):

        # Retrieve artists already listened to by this user
        row_start = known.indptr[user_idx]
        row_end = known.indptr[user_idx + 1]

        known_artists = known.indices[row_start:row_end]

        # Create a mask identifying eligible negative artists
        candidate_mask = np.ones(n_artists, dtype=bool)
        candidate_mask[known_artists] = False

        # Get indices of artists the user has NOT interacted with
        candidates = np.flatnonzero(candidate_mask)

        # Prevent infinite sampling when no negatives are available
        if len(candidates) == 0:
            raise ValueError(
                f"User {user_idx} has interacted with every artist. "
                "Cannot generate negative samples."
            )

        # Sample negatives with replacement
        sampled = rng.choice(
            candidates,
            size=(count, negatives_per_positive),
            replace=True
        )

        # Store samples in their original positions
        negative_artists[order[start:start + count]] = sampled

    return negative_artists


@dataclass
class TrainingBatch:
    users: np.ndarray
    artists: np.ndarray
    targets: np.ndarray # 0 / 1
    weights: np.ndarray

def iter_train_batches(train: csr_matrix, known_positive: csr_matrix, batch_size: int, negatives_per_positive: int, 
                       alpha: float, rng: np.random.Generator) -> Iterator[TrainingBatch]:
    """
    Input: Training CSR matrix; known-positive exclusion mask; number of positive anchors per batch; negatives per positive; 
    positive weighting parameter; RNG.

    Output: Iterator yielding TrainingBatch instances. Each positive anchor contributes one target-1 row and negatives_per_positive target-0 rows. 
    A batch can therefore contain up to batch_size * (1 + negatives_per_positive) examples.
    """
    raise NotImplementedError("iter_train_batches")


def make_evaluation_examples(
    heldout: csr_matrix,
    known_positive: csr_matrix,
    negatives_per_positive: int,
    rng: np.random.Generator
) -> TrainingBatch:

    if heldout.shape != known_positive.shape:
        raise ValueError("Matrix dimensions do not match")

    heldout = heldout.tocsr(copy=True)
    heldout.sum_duplicates()
    heldout.eliminate_zeros()
    heldout.sort_indices()

    # Extract positive interactions
    positive_users, positive_artists, play_counts = (
        extract_positive_interactions(heldout)
    )

    n_positive = len(positive_users)

    # Generate actual negative artist IDs
    negative_artists = sample_negative_items(
        positive_users,
        known_positive,
        heldout.shape[1],
        negatives_per_positive,
        rng
    )

    # Repeat each user for their negative samples
    negative_users = np.repeat(
        positive_users,
        negatives_per_positive
    )

    # Flatten the negative artist matrix
    negative_artists = negative_artists.flatten()

    # Combine positive and negative examples
    users = np.concatenate([
        positive_users,
        negative_users
    ])

    artists = np.concatenate([
        positive_artists,
        negative_artists
    ])

    targets = np.concatenate([
        np.ones(n_positive),
        np.zeros(len(negative_users))
    ]).astype(np.float32)

    weights = np.ones(len(users), dtype=np.float32)

    # Shuffle while keeping arrays aligned
    permutation = rng.permutation(len(users))

    return TrainingBatch(
        users=users[permutation],
        artists=artists[permutation],
        targets=targets[permutation],
        weights=weights[permutation]
    )




    

######
# Initialize Model and Make Predictions
######

@dataclass
class MFModel:
    user_factors: np.ndarray
    artist_factors: np.ndarray


def initialize_model(n_users: int, n_artists: int, n_factors: int, seed: int) -> MFModel:
    """
    Input: User count, artist count, latent factor count, RNG seed.

    Output: MFModel with user factors P.shape == (n_users, n_factors) and artist factors Q.shape == (n_artists, n_factors), float32.

    Initializes small random values using np.random.default_rng(seed). Uses a reproducible seed.
    """

    rng = np.random.default_rng(seed)
    
    user_factors = rng.normal(loc=0.0, scale=0.01, size=(n_users, n_factors)).astype(np.float32)
    artist_factors = rng.normal(loc=0.0, scale=0.01, size=(n_artists, n_factors)).astype(np.float32)
    return MFModel(user_factors=user_factors, artist_factors=artist_factors)



def sigmoid(scores: np.ndarray) -> np.ndarray:
    """
    pretty self explanatory, eh?
    """
    scores = np.clip(scores, -500, 500)
    return (1 / (1 + np.exp(-scores)))


def predict_scores(model: MFModel, users: np.ndarray, artists: np.ndarray) -> np.ndarray:
    """
    Input: Model and aligned arrays of user/artist indices of equal length.

    Output: 1-D array of raw dot-product scores, shape (n_pairs,).  
    """
    users = np.asarray(users)
    artists = np.asarray(artists)

    if users.ndim != 1 or artists.ndim != 1:
        raise ValueError("User and artist arrays must be 1-D")

    if len(users) != len(artists):
        raise ValueError("User and artist arrays must have equal lengths")

    if np.any(users < 0) or np.any(users >= len(model.user_factors)):
        raise ValueError("Invalid user indices")

    if np.any(artists < 0) or np.any(artists >= len(model.artist_factors)):
        raise ValueError("Invalid artist indices")

    user_vectors = model.user_factors[users]
    artist_vectors = model.artist_factors[artists]

    return np.sum(user_vectors * artist_vectors, axis=1)


def predict_probabilities(model: MFModel, users: np.ndarray, artists: np.ndarray) -> np.ndarray:
    """
    Input: Model and aligned arrays of user/artist indices of equal length.

    Output: 1-D array of probabilities, shape (n_pairs,).  
    """
    scores = predict_scores(model, users, artists)
    probabilities = sigmoid(scores)
    return probabilities


def predict_all_artists(model: MFModel, user_idx: int) -> np.ndarray:
    """
    Input: Model and a single user index.

    Output: 1-D array of probabilities for all artists, shape (n_artists,).  
    """
    if user_idx < 0 or user_idx >= len(model.user_factors):
        raise ValueError("Invalid user index")
    user_vector = model.user_factors[user_idx]
    scores = model.artist_factors @ user_vector
    probabilities = sigmoid(scores)
    return probabilities


##################
# Loss and SGD
##################

def binary_cross_entropy_loss(predicted_probability: float, target: float) -> float:
    """
    Input: Predicted probability, binary target, example weight.

    Output: Scalar weighted BCE loss.
    """
    predicted_probability = np.clip(predicted_probability, 1e-7, 1.0 - 1e-7)
    return float(-(target * np.log(predicted_probability) + (1 - target) * np.log(1 - predicted_probability)))


def calculate_pair_loss(user_vector: np.ndarray, artist_vector: np.ndarray, target: float, weight: float,
    regularization: float,) -> float:
    """
    Input: One user vector, one artist vector, binary target, positive example weight, non-negative L2 coefficient.

    Output: Scalar objective value for that one pair.
    """
    score = np.dot(user_vector, artist_vector)
    predicted_probability:float = sigmoid(np.array(score))
    loss = binary_cross_entropy_loss(predicted_probability, target)

    #TODO SOMEONE DOUBLE CHECK ME HERE I DIDNT TAKE LINEAR ALGEBRA!!!!!
    l2_penalty = (regularization/2) * (np.dot(user_vector,user_vector.T) + np.dot(artist_vector, artist_vector.T)) 
    return float(weight * loss + l2_penalty)


def calculate_score_gradient(predicted_probability: float, target: float, weight: float) -> float:
    """
    Input: Predicted probability, binary target, example weight.

    Output: Scalar g = weight * (predicted_probability - target)    
    """
    score_gradient = weight * (predicted_probability - target)
    return score_gradient


def sgd_update(model: MFModel, user_idx: int, artist_idx: int, target: float, weight: float, learning_rate: float, regularization: float) -> float:
    """
    Input: Model, one user index, one artist index, binary target, positive sample weight, learning rate, L2 coefficient.

    Output: Scalar pre-update pair objective (or document an alternative consistently).

    Read/copy old p_u and q_i; calculate score and sigmoid; calculate g; calculate both gradients from the old vectors; update those two rows in place.
    """
    old_user_vector = model.user_factors[user_idx].copy()
    old_artist_vector = model.artist_factors[artist_idx].copy()

    score = np.dot(old_user_vector, old_artist_vector)
    predicted_probability = sigmoid(np.array(score))
    score_gradient_weighted_error = calculate_score_gradient(predicted_probability, target, weight)

    user_gradient = score_gradient_weighted_error * old_artist_vector + regularization * old_user_vector
    user_updated = old_user_vector - learning_rate * user_gradient

    artist_gradient = score_gradient_weighted_error * old_user_vector + regularization * old_artist_vector
    artist_updated = old_artist_vector - learning_rate * artist_gradient

    model.artist_factors[artist_idx] = artist_updated
    model.user_factors[user_idx] = user_updated

    return calculate_pair_loss(old_user_vector, old_artist_vector, target, weight, regularization)


def calculate_training_loss(model: MFModel, examples: TrainingBatch, regularization: float) -> float:
    """
    Input: Model and a fixed batch of labelled examples.

    Output: Mean per-example weighted BCE plus per-example L2 penalty.
    """
    total_loss = 0.0
    for user_idx, artist_idx, target, weight in zip(examples.users, examples.artists, examples.targets, examples.weights):
        if user_idx < model.user_factors.shape[0] and artist_idx < model.artist_factors.shape[0]:
            
            user_vector = model.user_factors[user_idx]
            artist_vector = model.artist_factors[artist_idx]
            loss = calculate_pair_loss(user_vector, artist_vector, target, weight, regularization)
            total_loss += loss

    mean_loss = total_loss / len(examples.users)
    return mean_loss



@dataclass
class TrainingConfig:
    n_factors: int
    epochs: int
    learning_rate: float
    regularization: float
    negatives_per_positive: int
    positive_weight_alpha: float
    batch_size: int
    seed: int
    top_k: int
    early_stopping_patience: int



def generate_shuffled_batches(train: csr_matrix, known_positive: csr_matrix, batch_size: int,
                              rng: np.random.Generator, config: TrainingConfig) -> list[TrainingBatch]:
    npp = config.negatives_per_positive
    users, artists, _ = extract_positive_interactions(train)

    perm = rng.permutation(len(users))
    users, artists = users[perm], artists[perm]

    batches = []
    for i in range(0, len(users), batch_size):
        pos_users = users[i:i + batch_size]
        pos_artists = artists[i:i + batch_size]

        # shape (n, npp); flatten is row-major, so it lines up with np.repeat
        neg_artists = sample_negative_items(pos_users, known_positive, train.shape[1], npp, rng).reshape(-1)
        neg_users = np.repeat(pos_users, npp)

        b_users = np.concatenate([pos_users, neg_users])
        b_artists = np.concatenate([pos_artists, neg_artists])
        b_targets = np.concatenate([np.ones(len(pos_users)), np.zeros(len(neg_users))]).astype(np.float32)
        b_weights = np.ones(len(b_users), dtype=np.float32)  # keep 1.0 until this works

        order = rng.permutation(len(b_users))
        batches.append(TrainingBatch(b_users[order], b_artists[order], b_targets[order], b_weights[order]))

    return batches

def train_one_epoch(model, train, known_positive, config, rng) -> float:
    batches = generate_shuffled_batches(train, known_positive, config.batch_size, rng, config)

    total_loss, total_examples = 0.0, 0
    for batch in batches:
        for u, a, t, w in zip(batch.users, batch.artists, batch.targets, batch.weights):
            total_loss += sgd_update(model, u, a, t, w, config.learning_rate, config.regularization)
            total_examples += 1

    mean_loss = total_loss / max(1, total_examples)
    print(f"Loss for this epoch: {mean_loss}")
    return mean_loss
   


################
# Validation and Testing
################

def evaluate_binary_loss(model: MFModel, examples: TrainingBatch, regularization: float) -> float:
    """
    Input: Model and a fixed batch of labelled examples.

    Output: Mean per-example weighted BCE.
    """
    total_loss = 0.0
    total_weight = 0.0
    predicted_probs = predict_probabilities(model, examples.users, examples.artists)

    for i in range(len(predicted_probs)):
        probability = predicted_probs[i]
        target = examples.targets[i]
        weight = examples.weights[i]

        user_vector = model.user_factors[examples.users[i]]
        artist_vector = model.artist_factors[examples.artists[i]]
        l2_penalty = (regularization / 2) * (np.dot(user_vector, user_vector) + np.dot(artist_vector, artist_vector))
        total_loss += weight * binary_cross_entropy_loss(probability, target) + l2_penalty
        total_weight += weight

    if total_weight > 0:
        return total_loss / total_weight
    else:
        return 0.0




def evaluate_accuracy(model: MFModel, examples: TrainingBatch, threshold: float = 0.5) -> float:
    """
    Input: Model, labelled sampled evaluation examples, probability threshold.

    Output: Fraction of examples clasified correctly.
    """
    predicted_probs = predict_probabilities(model, examples.users, examples.artists)
    predicted_labels = (predicted_probs >= threshold).astype(int)
    accuracy = np.mean(predicted_labels == examples.targets)
    return accuracy



def evaluate_ranking(model: MFModel, train: csr_matrix, heldout: csr_matrix, k:int) -> dict[str, float]:
    """
    Input: Model, training CSR matrix, held-out CSR matrix, integer k.

    Output: Dictionary of ranking metrics (precision@k, recall@k, etc).
    """
    if train.shape != heldout.shape:
        raise ValueError("train and heldout matrices must have same shape")
    if k <= 0:
        raise ValueError("k needs to be positive")

    n_users, n_artists = train.shape
    precision_total = 0.0
    recall_total = 0.0
    ndcg_total = 0.0
    users_with_heldout = 0

    for user_idx in range(n_users):
        heldout_items = heldout.getrow(user_idx).indices
        if heldout_items.size == 0:
            continue

        users_with_heldout += 1
        seen_items = train.getrow(user_idx).indices
        candidate_items = np.setdiff1d(np.arange(n_artists), seen_items, assume_unique=True)

        if candidate_items.size == 0:
            continue

        candidate_scores = predict_scores(model, np.full(candidate_items.shape[0], user_idx, dtype=np.int64), candidate_items)
        ordered_candidates = candidate_items[np.argsort(candidate_scores)[::-1]]
        top_k_items = ordered_candidates[:k]

        relevant_items = set(heldout_items.tolist())
        hits = np.intersect1d(top_k_items, heldout_items, assume_unique=True)

        precision_total += len(hits) / max(1, len(top_k_items))
        recall_total += len(hits) / len(heldout_items)

        dcg = 0.0
        for rank, item_idx in enumerate(top_k_items, start=1):
            if item_idx in relevant_items:
                dcg += 1.0 / np.log2(rank + 1)

        ideal_dcg = 0.0
        for rank in range(1, min(len(heldout_items), k) + 1):
            ideal_dcg += 1.0 / np.log2(rank + 1)

        if ideal_dcg > 0:
            ndcg_total += dcg / ideal_dcg
        else:
            ndcg_total += 0.0

    if users_with_heldout == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}

    return {
        "precision_at_k": precision_total / users_with_heldout,
        "recall_at_k": recall_total / users_with_heldout,
        "ndcg_at_k": ndcg_total / users_with_heldout,
    }


def evaluate_model(model: MFModel, data: DatasetSplits, validation_examples: TrainingBatch, k:int, regularization: float) -> dict[str, float]:
    """
    Input: Model, complete aligned data splits, fixed validation examples, ranking cutoff.

    Output: Dictionary with validation log loss, optionally others like validation accuracy, Recall@K, and NDCG@K. U choose, Sam.
    """
    loss = evaluate_binary_loss(model, validation_examples, regularization)
    accuracy = evaluate_accuracy(model, validation_examples)
    ranking = evaluate_ranking(model, data.train, data.validation, k)

    return {
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "precision_at_k": float(ranking.get("precision_at_k", 0.0)),
        "recall_at_k": float(ranking.get("recall_at_k", 0.0)),
        "ndcg_at_k": float(ranking.get("ndcg_at_k", 0.0)),
    }


######
# Reccomendation, saving, etc
#######

@dataclass
class Recommendation:
    artist_idx: int
    artist_mbid: str
    artist_name: str
    score: float
    

def recommend_for_user(model: MFModel, user_idx: int, train: csr_matrix, artist_mbids: np.ndarray, 
                       artist_names: np.ndarray, k:int) -> list[Recommendation]:
    """
    Input: Trained model; valid user index; training CSR matrix; aligned MBID/name mappings; recommendation count.

    Output: Up to k Recommendation objects sorted by descending raw score.
    """
    if user_idx < 0 or user_idx >= model.user_factors.shape[0]:
        raise ValueError("Invalid user index")

    if k <= 0:
        return []

    n_artists = model.artist_factors.shape[0]

    if len(artist_mbids) != n_artists or len(artist_names) != n_artists:
        raise ValueError("Artists don't match model dimensions")

    # Predict scores for all artists
    scores = predict_all_artists(model, user_idx)

    # Exclude previously listened-to artists
    seen_artists = train.getrow(user_idx).indices

    scores = scores.copy()
    scores[seen_artists] = -np.inf

    # Get top K artists
    top_indices = np.argsort(scores)[::-1]
    top_indices = top_indices[np.isfinite(scores[top_indices])][:k]

    recommendations = []

    for artist_idx in top_indices:
        recommendations.append(
            Recommendation(
                artist_idx=int(artist_idx),
                artist_mbid=str(artist_mbids[artist_idx]),
                artist_name=str(artist_names[artist_idx]),
                score=float(scores[artist_idx])
            )
        )

    return recommendations

def save_model(model: MFModel, config: TrainingConfig, output_path: str | Path) -> None:
    """
    Input: Model, training configuration, output path.

    Output: None. Saves model and config to disk.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path / "model.npz",
        user_factors=model.user_factors,
        artist_factors=model.artist_factors
    )

    np.savez(
        output_path / "config.npz",
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
        }
    )



def load_model(input_path: Path) -> tuple[MFModel, TrainingConfig]: 
    """
    Input: Path to model directory.

    Output: Tuple of (MFModel, TrainingConfig) loaded from disk.
    """
    input_path = Path(input_path)

    model_data = np.load(input_path / "model.npz")

    model = MFModel(
        user_factors=model_data["user_factors"],
        artist_factors=model_data["artist_factors"]
    )

    config_data = np.load(input_path / "config.npz")

    config = TrainingConfig(
        **{
            field: config_data[field].item()
            for field in TrainingConfig.__dataclass_fields__
        }
    )

    return model, config



@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    validation_recall_at_k: list[float]
    validation_ndcg_at_k: list[float] # Sam this is Normalized Discounted Cumulative Gain you're gonna wanna look into it -Charlie


def train_model(data: DatasetSplits, config: TrainingConfig) -> tuple[MFModel, TrainingHistory]:
    """
    Input: Complete aligned data splits, validated training configuration.

    Output: Tuple of (MFModel, TrainingHistory) after training to convergence or early stopping.
    """
    n_users, n_artists = data.train.shape
    model = initialize_model(n_users, n_artists, config.n_factors, config.seed)
    history = TrainingHistory([], [], [], [])
    known_positive = (
    data.train + data.validation + data.test
    ).tocsr()

    known_positive.sum_duplicates()
    known_positive.eliminate_zeros()
    known_positive.sort_indices()

    known_positive.data = np.ones(
        known_positive.nnz,
        dtype=np.int8
    )
    for epoch in range(config.epochs):
        rng = np.random.default_rng(config.seed + epoch)
        train_loss = train_one_epoch(model, data.train, known_positive, config, rng)        
        history.train_loss.append(train_loss)
        validation_examples = make_evaluation_examples(data.validation, known_positive, config.negatives_per_positive, rng)
        print("User factor dimensions:", model.user_factors.shape)
        print("Artist factor dimensions:", model.artist_factors.shape)

        print("Validation users:", validation_examples.users.min(),
            validation_examples.users.max())

        print("Validation artists:", validation_examples.artists.min(),
            validation_examples.artists.max())
        validation_loss = evaluate_binary_loss(model, validation_examples, config.regularization)
        history.validation_loss.append(validation_loss)

        # ranking_metrics = evaluate_ranking(model, data.train, data.validation, config.top_k)
        # history.validation_recall_at_k.append(ranking_metrics.get('recall_at_k', 0.0))
        # history.validation_ndcg_at_k.append(ranking_metrics.get('ndcg_at_k', 0.0))

        # Early stopping logic can be added here based on validation metrics

    return model, history

def train_different_parameters():
    rates = [0.05, 0.2, 0.5]
    regularizations = [0.01, 0.1]
    batch_sizes = [512, 1024]
    data = load_lastfm_dataset()

    for rate in rates:
        learn = rate
        for regularization in regularizations:
            reg = regularization
            for batch_size in batch_sizes:
                batch = batch_size
                start_time = time()
                config = TrainingConfig(
                    n_factors=16,
                    epochs=50,
                    learning_rate=learn,
                    regularization=reg,
                    negatives_per_positive=3,
                    positive_weight_alpha=0.5,
                    batch_size=batch,  # positive anchors per batch
                    seed=1235,
                    top_k=10,
                    early_stopping_patience=5,
                )
                current_run:str = ""
            
            
                current_run += f"Config: {config}\n"
            
                print("Starting training with config:", config)
                model, history = train_model(data, config)
                # Further evaluation and saving of the model can be done here
                end_time = time()
                current_run += f"Training time: {end_time - start_time:.2f} seconds for {config.epochs} epochs. {(end_time - start_time)/config.epochs:.2f} seconds per epoch\n"
                current_run += f"Training time: {(end_time - start_time)/60:.2f} minutes for {config.epochs} epochs. {((end_time - start_time)/config.epochs)/60:.2f} minutes per epoch\n"
                current_run += f"Final training loss: {history.train_loss[-1]:.4f}\n"
                print(current_run)
            
                with open("training_log_params.txt", "a") as f:
                    f.write(current_run + "\n\n\n")

                        

    

from time import time
def main():
    # log time difference between start and end of training
    start_time = time()
    data = load_lastfm_dataset()
    config = TrainingConfig(
        n_factors=2,
        epochs=3,
        learning_rate=0.25,
        regularization=0.01,
        negatives_per_positive=3,
        positive_weight_alpha=.5,
        batch_size=1024,  # positive anchors per batch
        seed=1235,
        top_k=10,
        early_stopping_patience=5,
    )
    current_run:str = ""


    current_run += f"Config: {config}\n"

    print("Starting training with config:", config)
    model, history = train_model(data, config)
    # Further evaluation and saving of the model can be done here
    end_time = time()
    current_run += f"Training time: {end_time - start_time:.2f} seconds for {config.epochs} epochs. {(end_time - start_time)/config.epochs:.2f} seconds per epoch\n"
    current_run += f"Training time: {(end_time - start_time)/60:.2f} minutes for {config.epochs} epochs. {((end_time - start_time)/config.epochs)/60:.2f} minutes per epoch\n"
    current_run += f"Final training loss: {history.train_loss[-1]:.4f}\n"
    current_run += f"Training loss history: {history.train_loss}"
    current_run += f"Validation loss history: {history.validation_loss}"
    print(current_run)

    with open("training_log.txt", "a") as f:
        f.write(current_run + "\n\n\n")
main()

"""
Starting Config:
TrainingConfig(
    n_factors=16,
    epochs=50,
    learning_rate=0.03,
    regularization=0.01,
    negatives_per_positive=3,
    positive_weight_alpha=0.5,
    batch_size=1024,  # positive anchors per batch
    seed=1235,
    top_k=10,
    early_stopping_patience=5,
)


"""