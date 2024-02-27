import hnswlib

MAX_SAMPLE_SIZE = 10000


def choose_exploratory_factor(n_samples, k, allow_override=True):
# ef always should be >K. higher ef, more accurate query

    if n_samples > MAX_SAMPLE_SIZE and allow_override:
        ef = min(n_samples - 10, 500)
    else:
        ef = max(100, k + 1)
    return ef


def choose_exploratory_factor_construction(
    n_samples, default_ef_construction, k, allow_override=True
):
    if n_samples < MAX_SAMPLE_SIZE and allow_override:
        ef_construction = max(100, k + 1)
    else:
        ef_construction = default_ef_construction
    return ef_construction


def choose_m(M, n_samples, n_components, allow_override=True):
    if n_components > 30 & n_samples <= 50000 and allow_override:
        M = 48  # good for scRNA seq where dimensionality is high
    else:
        M = M
    return M


def create_hnsw_index(
    data, distance_metric, k, M, default_ef_construction, n_threads=None, allow_override=True
):
    n_components = data.shape[1]
    n_samples = data.shape[0]
    hnsw_index = hnswlib.Index(space=distance_metric, dim=n_components)

    ef = choose_exploratory_factor(n_samples, k, allow_override)
    M = choose_m(M, n_samples, n_components, allow_override)
    ef_construction = choose_exploratory_factor_construction(
        n_samples, default_ef_construction, k, allow_override
    )

    if n_threads is not None:
        hnsw_index.set_num_threads(n_threads)

    hnsw_index.init_index(max_elements=n_samples, ef_construction=ef_construction, M=M)
    hnsw_index.add_items(data)
    hnsw_index.set_ef(ef)

    return hnsw_index
