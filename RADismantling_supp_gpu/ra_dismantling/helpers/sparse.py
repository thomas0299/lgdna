from cupyx.scipy.sparse import csr_matrix


def _sparse_estimator_wrapper(x: csr_matrix, estimator: str, **kwargs):
    return getattr(x, estimator)(**kwargs).ravel()


def _sparse_sum(x: csr_matrix, **kwargs):
    return _sparse_estimator_wrapper(x, "sum", **kwargs)
