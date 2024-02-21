import hnswlib

MAX_SAMPLE_SIZE = 10000


def choose_exploratory_factor(n_samples, k, clusters_are_oversized=False):
# ef always should be >K. higher ef, more accurate query

    if n_samples > MAX_SAMPLE_SIZE and not clusters_are_oversized:
        ef = min(n_samples - 10, 500)
    else:
        ef = max(100, k + 1)
    return ef


def choose_exploratory_factor_construction(
    n_samples, default_ef_construction, k, clusters_are_oversized=False
):
    if clusters_are_oversized:
        ef_construction = 200
    elif n_samples < MAX_SAMPLE_SIZE:
        ef_construction = max(100, k + 1)
    else:
        ef_construction = default_ef_construction
    return ef_construction


def choose_m(n_samples, n_components, clusters_are_oversized=False):
    if clusters_are_oversized:
        M = 30
    elif n_components > 30 & n_samples <= 50000:
        M = 48  # good for scRNA seq where dimensionality is high
    else:
        M = 24
    return M


def create_hnsw_index(
    data, distance, k, num_threads, default_ef_construction,
    clusters_are_oversized=False
):
    n_components = data.shape[1]
    n_samples = data.shape[0]
    hnsw_index = hnswlib.Index(space=distance, dim=n_components)

    ef = choose_exploratory_factor(n_samples, k, clusters_are_oversized)
    M = choose_m(n_samples, n_components, clusters_are_oversized)
    ef_construction = choose_exploratory_factor_construction(
        n_samples, default_ef_construction, k, clusters_are_oversized
    )

    if not clusters_are_oversized:
        hnsw_index.set_num_threads(num_threads)

    hnsw_index.init_index(max_elements=n_samples, ef_construction=ef_construction, M=M)
    hnsw_index.add_items(data)
    hnsw_index.set_ef(ef)

    return hnsw_index
