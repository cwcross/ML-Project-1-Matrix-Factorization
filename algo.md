# Algorithm

Our collaborative filtering model uses matrix factorization optimized with SGD.

## Data Tidying

The data comes in a tabular format:

| User | Artist | Play Count |
|------|------|--|
| Alice | Radiohead | 25 |
| Alice | Bjork | 12 |
| Alice | Daft Punk | 3 |
| Alice | U2 | 0 |
| Alice | Green Day | 5 |
| Bob | Radiohead | 8 |
| Bob | Daft Punk | 4 |
| Bob | Taylor Swift | 30 |
| Chris | Bjork | 15 |
| Chris | U2 | 9 |
| Chris | Green Day | 10 |
| ... | ... | ... |

### Basic Cleaning

We first clean this data:
* Combine duplicates
* Join artists based on universal ID

### Filtering

We want to ensure sufficient data for each user and artist. k-core filtering with $k=5$ ensures each user has interactions with at least 5 artists, and each artist has interactions with at least 5 users.

### Sampling

360k users is a huge dataset, and we'll only work with 20k for simplicity. We randomly sample 20k users.

### Re-Run Filtering

Now that we've removed some users, it's possible not all users and artists will meet the $k=5$ threshold, so we run the filter again.

### Sparse Matrix

Then we convert the data into a sparse matrix.

A sparse matrix of user-artist relationships $R$ takes the form:

$$
R = \begin{bmatrix}
25 & 12 & 3 & 0 & 5 & 0 & ...\\
8 & 0 & 4 & 0 & 0 & 30 \\
0 & 15 & 0 & 0 & 10 & 0 \\
\vdots & & & & & & \ddots
\end{bmatrix}
$$

Each row represents a user and each column represents an artist. Each entry represents a user-artist relationship $r_{ui}$ for a user $u$ and artist $i$.

## Data Splitting

We want to make predictions about a user's taste in artists. To do this, for each user, we train on some of their user-artist interactions, and test on some withheld user-artist interactions.

However, every user-artist interaction for a given user is in the same row of the sparse matrix.

To withhold some user-artist interactions for each user, we need to withhold a random subset of each row, instead of some random rows.

To do this, we create a random ordering of the entries in a row.

Next, we calculate the number of entries to go into each split. The test split gets 10% of entries; validation get 20%; train gets the rest.

The number of entries per split for a user with $n$ artists:

$$
n_\text{test} = \max(1, \lfloor 0.1n \rfloor) \\
n_\text{val} = \max(1, \lfloor 0.2n \rfloor) \\
n_\text{train} = n - n_\text{test} - n_\text{val}
$$

Floors are used to round to an integer. Maximums guarantee at least one artist in each split for each user.

### More Filtering

Splits are per-user, which means they don't consider which particular artists are in which split. A possible consequence of this is that a particular artist ends up in the validation or test data, but not the training data.

We filter our these artists from the data. A vector $q_i$ for such an artist will be trained to score low for everyone. Filtering out these artists that are not included in training data means that our evaluation is not unfairly penalized by an artifact of splitting. On a trial run with 20,000 users, only 3 artists met this criteria.

## Training

### Weights

The data is in the form of raw play counts in a sparse matrix ($R_\text{train}$, $R_\text{val}$, $R_\text{test}$). Our dataset is heavy-tailed, meaning it there is a huge amount of listener interaction data from a small minority of users.

We want these play counts to have an influence, because they can function as a proxy for the degree to which a user likes an artist. However, we want to minimize the effect a few "super-listeners" have on the data use log-base weights for each user-artist relationship $r_{ui}$:

$$
w_ui = 1 + \alpha \log(1 + r_{ui})
$$

We create weighted sparse matrices using the tidied sparse matrices.

$\alpha$ is a hyperparameter we can use to modify the influence of a higher play count.

### Negatives

#### Negative to Positive Ratio

We want to train our algorithm using the positive examples in the data ($r_{ui} > 0$). This is the most valuable signal in our data and we want to emphasize it most. Negative examples ($r_{ui} = 0$) don't necessarily indicate *lack* of interest, just that a user hasn't yet listened to an artist. However, we do want to use *some* negative data in our training process, because it is still informative.

We strike a balance between using *every* negative example and *none* by choosing a ratio of $m$ negative examples for each positive example we train on.

$m$ is a hyperparameter we can use to modify the influence of negatives in the data.

#### What Counts as a Negative

A negative in the data is meant to represent no recorded interaction with an artist. But this is not strictly true in the training data. Consider the following simplified example:

- User $u$ listens to artists $i_1$, $i_2$, and $i_3$
- These three examples are split into $R_\text{train}$, $R_\text{val}$, and $R_\text{test}$, respectively

Reacall that user-artist relationship data is split randomly per row, not a general withholding of certain artists or users. Thus, the un-split data $R$ has the same shape as each split $R_\text{train}$, $R_\text{val}$, and $R_\text{test}$. They contain the same users and artists, they only differ in which non-zero data points they keep.

As a consequence, $R_\text{train}$ has a $0$ for user $u$ and artists $i_2$ and $i_3$. This is part of the motivation for only sampling *some* negative examples: they are not true indications of a user's preference, just a lack of data.

#### Sampling Method

Our data is stored as a sparse matrix, which only records the positive data and its location in the matrix, not the negative examples. This makes it nontrivial to randomly sample $m$ negative examples.

We begin with an array of users in the sparse matrix, with one occurence for each positive interaction with an artist. Therefore, a user $u_1$ with 3 artist interactions would appear 3 times, and a user $u_2$ with 2 artist interactions would appear twice:

$$
\text{users}_\text{pos} = [u_1, u_1, u_1, u_2, u_2]
$$

We want $m$ negative examples for each positive example. We can start with an array of the users for these negative examples. The array will have $m$ entries for each corresponding entry in $\text{users}_\text{pos}$. For example, if $m=2$:

$$
\text{users}_\text{neg} = [u_1, u_1, u_1, u_1, u_1, u_1, u_2, u_2, u_2, u_2]
$$

For each of these users, we need to find a negative artist interaction (where $r_{ui}=0$). We pick a random artist $i$ (drawing with replacement) and check if $r_{ui}=0$, resampling if not. Once we have an artist, we count it as an example with $y=0$ and $w=1$.

The result of this is several arrays:

| Array | Description |
|--|--|
| $\text{users}_\text{pos}$ | each user's index $u$ appears $n$ times, where $n$ is the number of artists they have a positive interaction with |
| $\text{artists}_\text{pos}$ | parallel to $\text{users}_\text{pos}$, which has one entry per positive example; this array contains the artist index $i$ corresponding with each example |
| $W_\text{pos}$ | parallel to $\text{users}_\text{pos}$, which has one entry per positive example; this array contains the weight $w$ corresponding with each example |
| $\text{users}_\text{neg}$ | each users's index $u$ appears $n \times m$ times, where $n$ is the number of artists they have a positive interaction with, and $m$ is the hyperparameter controlling the number of negative examples per positive example |
| $\text{artists}_\text{neg}$ | parallel to $\text{users}_\text{neg}$, which has one entry per negative example; this array contains the artist index $i$ corresponding with each example |

### Combining Data and Shuffling

We take all the arrays resulting from the previous step and combine them:

- Stack $\text{users}_\text{pos}$ with $\text{users}_\text{neg}$ to create $\text{users}$
- Stack $\text{artists}_\text{pos}$ with $\text{artists}_\text{neg}$ to create $\text{artists}$
- Stack $W_\text{pos}$ with an array of all $1$ the length of $\text{users}_\text{neg}$, $W_\text{neg}$ (weight for all negative examples is $1$), to create $W$
- Create a new array $Y_\text{pos}$ the length of $\text{users}_\text{pos}$, with every entry $1$ ($y=1$ for positive examples); create a new array $Y_\text{neg}$ the length of $\text{user}_\text{neg}$, with every entry $0$ ($y=0$ for negative examples); stack these to create $Y$

The result is four parallel arrays $\text{users}, \text{artists}, W, Y$.

Then we shuffle the entries of these arrays, keeping them aligned with one another. We create a random ordering of entries and apply it to every array. This means we interleave the positive and negative examples, giving smoother gradient updates in a single epoch.

These four arrays represent all the data we feed into the model to train it in a given epoch.

### Updates

We loop over the examples in the data. For each example $(u, i, y, w)$ from arrays $(\text{users}, \text{artists}, W, Y)$:

1. Record score.

$$
s = p_u^\top q_i
$$

---













---

Loss for SGD:

$$
\ell = w\left[-y \log \hat p - (1 - y)\log(1 - \hat p)\right] + \frac{\lambda}{2}\left(\lVert p_u \rVert^2 + \lVert q_i \rVert^2\right)
$$