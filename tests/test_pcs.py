import sys
sys.path.insert(0, './../')
sys.path.insert(0, './')
from numpy import array
from scipy.sparse import isspmatrix, random, csr_matrix
from pytest import approx
from effcossim.pcs import pairwise_cosine_similarity, pp_pcs


A = array(
    [
        [1, 2, 3], 
        [0, 1, 2],
        [5, 1, 1]
    ]
)

B = array(
    [
        [1, 1, 2], 
        [0, 1, 2],
        [5, 0, 1], 
        [0, 0, 4]
    ]
)

C = random(
    m=100,       
    n=30, 
    density=0.1, 
    format='csr', 
    random_state=1102
)

D = random(
    m=111, 
    n=30, 
    density=0.2, 
    format='coo', 
    random_state=1102
)

E = array(
    [
        [[1, 2, -3], [3, 4, 5]], 
        [[0, 1, 2], [0, 1, 4]],
        [[5, 1, 1], [-1, -5, -8]]
    ]
)

F = array(
    [
        [[1, 1, 2], [3, 4, 5]], 
        [[-1, 1, 2], [3, 4, 5]],
        [[-5, 0, 1], [0, 1, 2]],
        [[0, 0, 4], [0, 1, 2]]
    ]
)

G = csr_matrix(
    [
        [1, 2, -3, 3, 4, 5], 
        [0, 1, 2, 0, 1, 4],
        [5, 1, 1, -1, -5, -8]
    ]
)

H = csr_matrix(
    [
        [1, 1, 2, 3, 4, 5], 
        [-1, 1, 2, 3, 4, 5],
        [-5, 0, 1, 0, 1, 2],
        [0, 0, 4, 0, 1, 2]
    ]
)

l1 = [random(m=1000, n=1000, density=0.3,) for _ in range(6)]
l2 = [random(m=900, n=1000, density=0.3,) for _ in range(6)]


def test_pcs_efficient_false_dense_output_true():
    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=False, 
        dense_output=True
    )
    assert not isspmatrix(output), "The output is sparse."
    assert output.sum() == approx(8.202465881449182), "Output not correct."


def test_pcs_efficient_false_dense_output_false():
    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=False, 
        dense_output=False
    )
    assert isspmatrix(output), "The output is not sparse."
    assert output.sum() == approx(8.202465881449182), "Output not correct."


def test_pcs_efficient_true_dense_output_false():
    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=True, 
        n_top=4, 
        lower_bound=0.0, 
        n_jobs=2, 
        dense_output=False
    )
    assert isspmatrix(output), "The output is not sparse."
    assert output.sum() == approx(8.202465881449182), "Output not correct."


def test_pcs_efficient_true_dense_output_true():
    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=True, 
        n_top=4, 
        lower_bound=0.0, 
        n_jobs=2, 
        dense_output=True
    )
    assert not isspmatrix(output), "The output is sparse."
    assert output.sum() == approx(8.202465881449182), "Output not correct."


def test_pcs_efficient_true_n_jobs():

    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=True, 
        n_top=4, 
        lower_bound=0.0, 
        n_jobs=-1, 
        dense_output=True
    )
    assert output.sum() == approx(8.202465881449182), "Output not correct."

    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=True, 
        n_top=4, 
        lower_bound=0.0, 
        n_jobs=1, 
        dense_output=True
    )
    assert output.sum() == approx(8.202465881449182), "Output not correct."


def test_pcs_efficient_true_lower_bound():

    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=True, 
        n_top=4, 
        lower_bound=0.9, 
        n_jobs=-1, 
        dense_output=True
    )
    assert (output > 0).sum() == 5, "Error in lower_bound."

    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=True, 
        n_top=4, 
        lower_bound=0.99, 
        n_jobs=-1, 
        dense_output=True
    )
    assert (output > 0).sum() == 1, "Error in lower_bound."


def test_pcs_efficient_true_n_top():

    output = pairwise_cosine_similarity(
        A=A, B=B, 
        efficient=True, 
        n_top=2, 
        lower_bound=0.1, 
        n_jobs=-1, 
        dense_output=True
    )
    assert (output > 0).sum() == 6, "Error in n_top."


def test_pcs_efficient_true_sparse():

    output = pairwise_cosine_similarity(
        A=C, B=D, 
        efficient=True, 
        n_top=2, 
        lower_bound=0.1, 
        n_jobs=-1, 
        dense_output=True
    )
    assert  output.sum() == approx(126.79083764959829), "Sparse input error."

def test_pcs_efficient_true_multi_dimensional():

    output_1 = pairwise_cosine_similarity(
        A=E, B=F, 
        efficient=True, 
        n_top=3, 
        lower_bound=0.1, 
        n_jobs=-1, 
        dense_output=True
    )
    output_2 = pairwise_cosine_similarity(
        A=G, B=H, 
        efficient=True, 
        n_top=3, 
        lower_bound=0.1, 
        n_jobs=-1, 
        dense_output=True
    )
    assert  output_1.sum() == approx(output_2.sum()), "Multi-dimensional error."
    
def test_pp_pcs():
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
    assert isinstance(L, list), "Parallel run error."