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
    A (csr_marix): The first sparse matrix to compare.
    B (csr_matrix): The second sparse matrix to compare.
    
    Returns:
    int
    """
    # Check by multiplying the matrices
    # Only left with a nonzero entry if matching locations are nonzero
    # Simplify multiplication by converting to ones in small ints
    A.data = np.ones_like(A.data, dtype=np.int8)
    B.data = np.ones_like(B.data, dtype=np.int8)
    return A.multiply(B).nnz


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
    # Implement the logic to build the known positive mask
    raise NotImplementedError("build_known_positive_mask")



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

    raise NotImplementedError("sample_negative_items")

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

def make_evaluation_examples(heldout:csr_matrix, known_positive: csr_matrix, negatives_per_positive: int, 
                             rng: np.random.Generator) -> TrainingBatch:
    """
    Input: Validation or test CSR matrix; known-positive exclusion mask; fixed negative ratio; RNG.

    Output: TrainingBatch containing held-out positives (target=1) and sampled negatives (target=0), with unit weights.
    """
    raise NotImplementedError("make_evaluation_examples")


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

    user_factors = np.random.default_rng(seed).normal(loc=0.0, scale=0.01, size=(n_users, n_factors)).astype(np.float32)
    artist_factors = np.random.default_rng(seed).normal(loc=0.0, scale=0.01, size=(n_artists, n_factors)).astype(np.float32)
    return MFModel(user_factors=user_factors, artist_factors=artist_factors)



def sigmoid(scores: np.ndarray) -> np.ndarray:
    """
    pretty self explanatory, eh?
    """

    return (1 / (1 + np.exp(-scores)))


def predict_scores(model: MFModel, users: np.ndarray, artists: np.ndarray) -> np.ndarray:
    """
    Input: Model and aligned arrays of user/artist indices of equal length.

    Output: 1-D array of raw dot-product scores, shape (n_pairs,).  
    """
    user_vectors = model.user_factors[users]
    artist_vectors = model.artist_factors[artists]

    scores = np.sum(user_vectors * artist_vectors, axis=1)
    return scores


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
    return -(target * np.log(predicted_probability) + (1 - target) * np.log(1 - predicted_probability))


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



def generate_shuffled_batches(train: csr_matrix, batch_size: int, rng: np.random.Generator, config: TrainingConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Input: Training CSR matrix; number of positive anchors per batch; RNG; Training configuration.

    Output: List of tuples (users, artists) representing shuffled batches of positive and negative interactions based on negative to positive ratio.
    """
    negatives_per_positive = config.negatives_per_positive

    users, artists, _ = extract_positive_interactions(train)
    indices = np.arange(len(users))
    rng.shuffle(indices)

    shuffled_users = users[indices]
    shuffled_artists = artists[indices]

    # Generate batches of positive anchors and their corresponding negative samples

    batches = []
    for i in range(0, len(shuffled_users), batch_size):
        #TODO Can someone look at this I'm getting red lines and idk why. Also Owen, am I calling your function correctly?
        # negative_samples = sample_negative_items(shuffled_users[i:i + batch_size], train, train.shape[1], negatives_per_positive, rng)

        #TODO add in the negative samples to the batch.
        batch = []
        batch_users = shuffled_users[i:i + batch_size]
        batch_artists = shuffled_artists[i:i + batch_size]
        batch.append((batch_users, batch_artists))
        batches.append(batch)

    return batches

def train_one_epoch(model: MFModel, train: csr_matrix, config: TrainingConfig, rng: np.random.Generator) -> float:
    """
    Input: Mutable model; training CSR matrix; negative-exclusion mask; validated training configuration; RNG.

    Output: Scalar average training objective for this epoch.
    
    Generate shuffled batches, process each labelled example with sequential SGD updates, accumulate a consistently defined loss, 
    and return the epoch metric.
    """
    batches = generate_shuffled_batches(train, config.batch_size, rng, config)

    total_loss = 0.0
    total_examples = 0

    for batch in batches:
        for batch_users, batch_artists in batch:

            # Currently only positive samples are implemented
            targets = np.ones(len(batch_users), dtype=np.float32)
            weights = np.ones(len(batch_users), dtype=np.float32)

            for user_idx, artist_idx, target, weight in zip(
                batch_users, batch_artists, targets, weights
            ):
                if user_idx < model.user_factors.shape[0] and artist_idx < model.artist_factors.shape[0]:

                    # sgd_update returns the individual example's loss
                    loss = sgd_update(
                        model,
                        user_idx,
                        artist_idx,
                        target,
                        weight,
                        config.learning_rate,
                        config.regularization
                    )

                    total_loss += loss
                    total_examples += 1

    print(f"Loss for this epoch: {total_loss / total_examples if total_examples > 0 else 0.0}")
    return total_loss / total_examples if total_examples > 0 else 0.0
   


################
# Validation and Testing
################

def evaluate_binary_loss(model: MFModel, examples: TrainingBatch) -> float:
    """
    Input: Model and a fixed batch of labelled examples.

    Output: Mean per-example weighted BCE.
    """
    raise NotImplementedError("evaluate_binary_loss")


def evaluate_accuracy(model: MFModel, examples: TrainingBatch, threshold: float = 0.5) -> float:
    """
    Input: Model, labelled sampled evaluation examples, probability threshold.

    Output: Fraction of examples clasified correctly.
    """
    raise NotImplementedError("evaluate_accuracy")


def evaluate_ranking(model: MFModel, train: csr_matrix, heldout: csr_matrix, k:int) -> dict[str, float]:
    """
    Input: Model, training CSR matrix, held-out CSR matrix, integer k.

    Output: Dictionary of ranking metrics (precision@k, recall@k, etc).
    """
    raise NotImplementedError("evaluate_ranking")


def evaluate_model(model: MFModel, data: DatasetSplits, validation_examples: TrainingBatch, k:int) -> dict[str, float]:
    """
    Input: Model, complete aligned data splits, fixed validation examples, ranking cutoff.

    Output: Dictionary with validation log loss, optionally others like validation accuracy, Recall@K, and NDCG@K. U choose, Sam.
    """
    raise NotImplementedError("evaluate_model")


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
    pass

def save_model(model: MFModel, config: TrainingConfig, output_path: str | Path) -> None:
    """
    Input: Model, training configuration, output path.

    Output: None. Saves model and config to disk.
    """
    raise NotImplementedError("save_model")


def load_model(input_path: Path) -> tuple[MFModel, TrainingConfig]: 
    """
    Input: Path to model directory.

    Output: Tuple of (MFModel, TrainingConfig) loaded from disk.
    """
    raise NotImplementedError("load_model")


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
    model = initialize_model(len(data.user_ids), len(data.artist_mbids), config.n_factors, config.seed)
    history = TrainingHistory([], [], [], [])
    for epoch in range(config.epochs):
        rng = np.random.default_rng(config.seed + epoch)
        train_loss = train_one_epoch(model, data.train, config, rng)
        history.train_loss.append(train_loss)

        # validation_examples = make_evaluation_examples(data.validation, build_known_positive_mask(data), config.negatives_per_positive, rng)
        # validation_loss = evaluate_binary_loss(model, validation_examples)
        # history.validation_loss.append(validation_loss)

        # ranking_metrics = evaluate_ranking(model, data.train, data.validation, config.top_k)
        # history.validation_recall_at_k.append(ranking_metrics.get('recall_at_k', 0.0))
        # history.validation_ndcg_at_k.append(ranking_metrics.get('ndcg_at_k', 0.0))

        # Early stopping logic can be added here based on validation metrics

    return model, history

from time import time
def main():
    """
    Main function to load data, train the model, and evaluate it.
    """
    # log time difference between start and end of training
    start_time = time()
    data = load_lastfm_dataset()
    config = TrainingConfig(
        n_factors=16,
        epochs=10,
        learning_rate=0.01,
        regularization=0.1,
        negatives_per_positive=3,
        positive_weight_alpha=0.5,
        batch_size=1024,  # positive anchors per batch
        seed=1235,
        top_k=10,
        early_stopping_patience=5,
    )
    model, history = train_model(data, config)
    print("Training completed.")
    print("Training Loss History:", history.train_loss)
    # Further evaluation and saving of the model can be done here
    end_time = time()
    print(f"Training time: {end_time - start_time:.2f} seconds for {config.epochs} epochs. {(end_time - start_time)/config.epochs:.2f} seconds per epoch")
    print(f"Training time: {(end_time - start_time)/60:.2f} minutes for {config.epochs} epochs. {((end_time - start_time)/config.epochs)/60:.2f} minutes per epoch")
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