from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn
from numpy import float64, array


def pairwise_cosine_similarity(
    A, 
    B, 
    efficient=False, 
    n_top=10, 
    lower_bound=0.01, 
    n_jobs=1, 
    dense_output=False
):

    if not efficient:
        
        M = cosine_similarity(X=A, Y=B, dense_output=dense_output)

        return M
    
    else:

        A = normalize(
            X=csr_matrix(A).astype(float64), 
            norm="l2", 
            axis=1, 
            copy=False
        )
        B = normalize(
            X=csr_matrix(B).transpose().astype(float64), 
            norm="l2", 
            axis=0, 
            copy=False
        )

        if n_jobs == 1:
            use_threads=False
        elif n_jobs <= 0:
            jobs = cpu_count()
            use_threads = True
        else:
            use_threads = True
        
        M = awesome_cossim_topn(
            A=A, 
            B=B, 
            ntop=n_top, 
            lower_bound=lower_bound, 
            use_threads=use_threads, 
            n_jobs=n_jobs
        )

        if dense_output:
            M = array(M.todense())

        return M

                
