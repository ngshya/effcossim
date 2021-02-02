# Efficient Pairwise Cosine Similarity Computation

The (i, j)-entry of the output matrix is the cosine distance between the i-th row of A and the j-th row of B. This function is only a wrapper, it uses the implementation of cosine_similarity from scikit-learn and the  implementation of awesome_cossim_topn from sparse_dot_topn. 
For more details, please check:
 - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
 - https://github.com/ing-bank/sparse_dot_topn

To install this package:
```bash
pip install effcossim
```

Sample code:
```python
from numpy import array
from effcossim.pcs import pairwise_cosine_similarity, pp_pcs

A = array([
    [1, 2, 3], 
    [0, 1, 2],
    [5, 1, 1]
])

B = array([
    [1, 1, 2], 
    [0, 1, 2],
    [5, 0, 1], 
    [0, 0, 4]
])

# scikit-learn implementation
M1 = pairwise_cosine_similarity(
    A=A, B=B, 
    efficient=False, 
    dense_output=True
)

# sparse_dot_topn implementation
M2 = pairwise_cosine_similarity(
    A=A, B=B, 
    efficient=True, 
    n_top=4, 
    lower_bound=0.5, 
    n_jobs=2, 
    dense_output=True
)
```

When `efficient=True`, in each row of the output matrix only the top `n_top` entries above `lower_bound` are retained (lower memory impacts). Furthermore, if `n_jobs` is larger than 1, parallel computations are applied (higher speed).

If multiple comparisons are required, the parallel implementation can be used. 

```python
l1 = [random(m=10000, n=1000, density=0.3,) for _ in range(6)]
l2 = [random(m=10000, n=1000, density=0.3,) for _ in range(6)]

L = pp_pcs(
    l1=l1, 
    l2=l2, 
    n_workers=2, 
    efficient=True, 
    n_top=10, 
    lower_bound=0.3, 
    n_jobs=2, 
    dense_output=False
)
```

The output is a list where the k-th element is the output of 

```python 
pairwise_cosine_similarity(l1[k], l2[k])
```

For further examples, check the notebook.