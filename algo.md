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




---

Loss for SGD:

$$
\ell = w\left[-y \log \hat p - (1 - y)\log(1 - \hat p)\right] + \frac{\lambda}{2}\left(\lVert p_u \rVert^2 + \lVert q_i \rVert^2\right)
$$