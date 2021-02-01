from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from multiprocessing import cpu_count, Pool
from scipy.sparse import csr_matrix, isspmatrix
from sparse_dot_topn import awesome_cossim_topn
from numpy import float64, array, prod, ndarray
from functools import partial


def pairwise_cosine_similarity(
    A, 
    B, 
    efficient=False, 
    n_top=10, 
    lower_bound=0.2, 
    n_jobs=1, 
    dense_output=False
):
    '''
    This function computes pairwise cosine distances efficiently. 
    The (i, j)-entry of the output matrix is the cosine distance between the 
    i-th row of A and the j-th row of B. This function is only a wrapper, it
    uses the implementation of cosine_similarity from scikit-learn and the 
    implementation of awesome_cossim_topn from sparse_dot_topn. 
    For more details, please check:
    - https://scikit-learn.org/stable/index.html
    - https://github.com/ing-bank/sparse_dot_topn

    Parameters
    ----------
    A : scipy sparse matrix or a numpy multidimensional array
        First matrix
    B : scipy sparse matrix or a numpy multidimensional array
        Second matrix
    efficient : bool, optional
        If True, the efficient implementation (sparse_dot_topn) will be used, 
        otherwise, the scikit-learn implementation will be use. The default 
        value is False. If efficient=False, then n_top, lower_bound and n_jobs 
        will be ignored.
    n_top : int, optional
        For each row of the output matrix, keep only the top n_top values, 
        other entries will be cast to zero. The default value is 10.
    lower_bound : float, optional
        Entries under this value will be cast to zero. The default value is 0.2.
    n_jobs : int, optional
        Number of workers for parallel computation. If negative, the number of 
        workers will be set to the number of cores. The default value is 1.
    dense_output : bool, optional
        Whether the output matrix should be a dense or a sparse matrix. 
        The default value is False.

    Returns
    -------
    scipy sparse matrix or a numpy multidimensional array
        Output matrix containing pairwise distances.
    '''

    assert isspmatrix(A) or isinstance(A, (ndarray)), \
        "A should be scipy sparse matrix or a numpy multidimensional array."
    assert isspmatrix(B) or isinstance(B, (ndarray)), \
        "B should be scipy sparse matrix or a numpy multidimensional array."
    assert isinstance(efficient, bool), "efficient should be a boolean."
    assert isinstance(n_top, int) and (n_top > 0), \
        "n_top should be a positive integer."
    assert isinstance(lower_bound, float) and \
        (lower_bound >= -1) and (lower_bound <= 1), \
        "lower_bound should be a real number in the range [-1, +1]."
    assert isinstance(n_jobs, int), "n_top should be an integer."
    assert isinstance(dense_output, bool), "dense_output should be a boolean."

    dim_A_1, dim_A_2 = A.shape[0], prod(A.shape[1:])
    dim_B_1, dim_B_2 = B.shape[0], prod(B.shape[1:])
    assert (dim_A_2 == dim_B_2), "Matrices second dimensions don't match."

    # Only 2D arrays
    A = A.reshape((dim_A_1, dim_A_2))
    B = B.reshape((dim_B_1, dim_B_2))

    if not efficient:

        # Standard scikit-learn implementation        
        M = cosine_similarity(X=A, Y=B, dense_output=dense_output)

        if not dense_output:
            M = csr_matrix(M)

        return M
    
    else:

        # Normalizing rows
        A = normalize(
            X=csr_matrix(A).astype(float64), 
            norm="l2", 
            axis=1, 
            copy=False
        )

        # Transpose the second matrix and normalizing columns
        B = normalize(
            X=csr_matrix(B).transpose().astype(float64), 
            norm="l2", 
            axis=0, 
            copy=False
        )

        # Number of workers
        if n_jobs == 1:
            use_threads = False
        elif n_jobs <= 0:
            n_jobs = cpu_count()
            use_threads = True
        else:
            use_threads = True
        
        # Efficient awesome_cossim_topn
        # Original implementation: 
        # https://github.com/ing-bank/sparse_dot_topn
        M = awesome_cossim_topn(
            A=A, 
            B=B, 
            ntop=n_top, 
            lower_bound=lower_bound, 
            use_threads=use_threads, 
            n_jobs=n_jobs
        )

        # Dense output
        if dense_output:
            M = array(M.todense())

        return M

                
def pp_pcs(
    l1, 
    l2, 
    n_workers=4,
    efficient=False, 
    n_top=10, 
    lower_bound=0.2, 
    n_jobs=1, 
    dense_output=False
):
    '''
    Parallel computation of the pairwise cosine similarity of two lists of 
    matrices, in an element-wise way. 

    Parameters
    ----------
    l1 : list
        First list of scipy sparse matrices of numpy multi-dimensional array.
    l2 : list
        Second list of scipy sparse matrices of numpy multi-dimensional array.
    n_workers : int, optional
        Number of workers in the Pool, by default 4. The final number of jobs 
        is n_workers x n_jobs.
    efficient : bool, optional
        If True, the efficient implementation (sparse_dot_topn) will be used, 
        otherwise, the scikit-learn implementation will be use. The default 
        value is False. If efficient=False, then n_top, lower_bound and n_jobs 
        will be ignored.
    n_top : int, optional
        For each row of the output matrix, keep only the top n_top values, 
        other entries will be cast to zero. The default value is 10.
    lower_bound : float, optional
        Entries under this value will be cast to zero. The default value is 0.2.
    n_jobs : int, optional
        Number of jobs for parallel computation. If negative, the number of 
        workers will be set to the number of cores. The default value is 1.
    dense_output : bool, optional
        Whether the output matrix should be a dense or a sparse matrix. 
        The default value is False.

    Returns
    -------
    list
        List of output distance matrices.
    '''

    assert isinstance(l1, list), "l1 should be a list."
    assert isinstance(l2, list), "l2 should be a list."
    assert isinstance(n_workers, int) and (n_workers > 0), \
        "n_workers should be a positive integer."

    pool = Pool(n_workers)
    f = partial(
        pairwise_cosine_similarity, 
        efficient=efficient, 
        n_top=n_top, 
        lower_bound=lower_bound, 
        n_jobs=n_jobs, 
        dense_output=dense_output
    )
    output = pool.starmap(f, zip(l1, l2))
    pool.close()
    pool.join()
    return output