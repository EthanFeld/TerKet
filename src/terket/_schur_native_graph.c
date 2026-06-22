#include "_schur_native_internal.h"

static PyObject *empty_order_and_width_result(void)
{
    PyObject *empty = PyList_New(0);
    PyObject *zero = PyLong_FromLong(0);
    PyObject *result;
    if (empty == NULL || zero == NULL) {
        Py_XDECREF(empty);
        Py_XDECREF(zero);
        return NULL;
    }
    result = PyTuple_Pack(2, empty, zero);
    Py_DECREF(empty);
    Py_DECREF(zero);
    return result;
}


static int build_cubic_adjacency(
    Py_ssize_t nvars,
    PyObject *q2,
    PyObject *q3,
    Py_ssize_t *out_nwords,
    uint64_t **out_adjacency
)
{
    Py_ssize_t nwords = (nvars + 63) >> 6;
    size_t total_words = (size_t) nvars * (size_t) nwords;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;
    uint64_t *adjacency = PyMem_Calloc(total_words > 0 ? total_words : 1, sizeof(uint64_t));

    if (adjacency == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        if (parse_pair_key(key, &left, &right) < 0) {
            PyMem_Free(adjacency);
            return -1;
        }
        bitset_set(adjacency + ((size_t) left * (size_t) nwords), right);
        bitset_set(adjacency + ((size_t) right * (size_t) nwords), left);
    }

    pos = 0;
    while (PyDict_Next(q3, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        if (parse_triple_key(key, &a, &b, &c) < 0) {
            PyMem_Free(adjacency);
            return -1;
        }
        bitset_set(adjacency + ((size_t) a * (size_t) nwords), b);
        bitset_set(adjacency + ((size_t) a * (size_t) nwords), c);
        bitset_set(adjacency + ((size_t) b * (size_t) nwords), a);
        bitset_set(adjacency + ((size_t) b * (size_t) nwords), c);
        bitset_set(adjacency + ((size_t) c * (size_t) nwords), a);
        bitset_set(adjacency + ((size_t) c * (size_t) nwords), b);
    }

    *out_nwords = nwords;
    *out_adjacency = adjacency;
    return 0;
}


static void initialize_remaining_set(uint64_t *remaining, Py_ssize_t nvars)
{
    Py_ssize_t idx;
    for (idx = 0; idx < nvars; ++idx) {
        bitset_set(remaining, idx);
    }
}


static void copy_remaining_neighbors(
    const uint64_t *adjacency,
    const uint64_t *remaining,
    Py_ssize_t nwords,
    Py_ssize_t var,
    uint64_t *neighbors
)
{
    const uint64_t *row = adjacency + ((size_t) var * (size_t) nwords);
    Py_ssize_t word_idx;
    for (word_idx = 0; word_idx < nwords; ++word_idx) {
        neighbors[word_idx] = row[word_idx] & remaining[word_idx];
    }
}


static void eliminate_var_from_graph(
    uint64_t *adjacency,
    uint64_t *remaining,
    const uint64_t *neighbors,
    uint64_t *work,
    Py_ssize_t nwords,
    Py_ssize_t var
)
{
    Py_ssize_t left;

    memcpy(work, neighbors, (size_t) nwords * sizeof(uint64_t));
    bitset_clear(remaining, var);

    while ((left = bitset_pop_first(work, nwords)) >= 0) {
        uint64_t *left_row = adjacency + ((size_t) left * (size_t) nwords);
        Py_ssize_t inner_word;
        for (inner_word = 0; inner_word < nwords; ++inner_word) {
            left_row[inner_word] |= neighbors[inner_word];
        }
        bitset_clear(left_row, left);
        bitset_clear(left_row, var);
    }

    memset(adjacency + ((size_t) var * (size_t) nwords), 0, (size_t) nwords * sizeof(uint64_t));
}


PyObject *min_fill_cubic_order_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *q2;
    PyObject *q3;
    Py_ssize_t nwords;
    uint64_t *adjacency = NULL;
    uint64_t *remaining = NULL;
    uint64_t *neighbors = NULL;
    uint64_t *best_neighbors = NULL;
    uint64_t *work = NULL;
    PyObject *order = NULL;
    PyObject *width_obj = NULL;
    Py_ssize_t order_idx = 0;
    Py_ssize_t remaining_count;
    Py_ssize_t max_scope = 1;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOO", &nvars, &q2, &q3)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }
    if (nvars == 0) {
        return empty_order_and_width_result();
    }

    if (build_cubic_adjacency(nvars, q2, q3, &nwords, &adjacency) < 0) {
        goto error;
    }
    remaining = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    neighbors = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    best_neighbors = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    work = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    if (
        adjacency == NULL || remaining == NULL || neighbors == NULL ||
        best_neighbors == NULL || work == NULL
    ) {
        PyErr_NoMemory();
        goto error;
    }

    initialize_remaining_set(remaining, nvars);
    remaining_count = nvars;

    order = PyList_New(nvars);
    if (order == NULL) {
        goto error;
    }

    while (remaining_count > 0) {
        Py_ssize_t best_var = -1;
        Py_ssize_t best_degree = 0;
        uint64_t best_fill = 0;
        int has_best = 0;
        Py_ssize_t word_idx;

        for (word_idx = 0; word_idx < nwords; ++word_idx) {
            uint64_t word = remaining[word_idx];
            while (word != 0) {
                Py_ssize_t var = (word_idx << 6) + lowest_bit_index(word);
                Py_ssize_t degree;
                uint64_t fill = 0;
                Py_ssize_t left;
                Py_ssize_t inner_word;

                word &= word - 1;
                if (var >= nvars) {
                    continue;
                }

                copy_remaining_neighbors(adjacency, remaining, nwords, var, neighbors);
                memcpy(work, neighbors, (size_t) nwords * sizeof(uint64_t));
                degree = bitset_count(neighbors, nwords);

                while ((left = bitset_pop_first(work, nwords)) >= 0) {
                    const uint64_t *left_row = adjacency + ((size_t) left * (size_t) nwords);
                    for (inner_word = 0; inner_word < nwords; ++inner_word) {
                        fill += (uint64_t) popcount_u64(work[inner_word] & ~left_row[inner_word]);
                    }
                }

                if (
                    !has_best ||
                    fill < best_fill ||
                    (fill == best_fill && (
                        degree < best_degree ||
                        (degree == best_degree && var < best_var)
                    ))
                ) {
                    has_best = 1;
                    best_var = var;
                    best_degree = degree;
                    best_fill = fill;
                    memcpy(best_neighbors, neighbors, (size_t) nwords * sizeof(uint64_t));
                }
            }
        }

        if (best_var < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to find an elimination variable.");
            goto error;
        }

        {
            PyObject *var_obj = PyLong_FromSsize_t(best_var);
            if (var_obj == NULL) {
                goto error;
            }
            PyList_SET_ITEM(order, order_idx, var_obj);
            ++order_idx;
        }

        {
            Py_ssize_t scope = bitset_count(best_neighbors, nwords) + 1;
            if (scope > max_scope) {
                max_scope = scope;
            }

            eliminate_var_from_graph(adjacency, remaining, best_neighbors, work, nwords, best_var);
            --remaining_count;
        }
    }

    width_obj = PyLong_FromSsize_t(max_scope);
    if (width_obj == NULL) {
        goto error;
    }

    PyMem_Free(adjacency);
    PyMem_Free(remaining);
    PyMem_Free(neighbors);
    PyMem_Free(best_neighbors);
    PyMem_Free(work);

    {
        PyObject *result = PyTuple_Pack(2, order, width_obj);
        Py_DECREF(order);
        Py_DECREF(width_obj);
        return result;
    }

error:
    PyMem_Free(adjacency);
    PyMem_Free(remaining);
    PyMem_Free(neighbors);
    PyMem_Free(best_neighbors);
    PyMem_Free(work);
    Py_XDECREF(order);
    Py_XDECREF(width_obj);
    return NULL;
}


static int degree_heap_less(Py_ssize_t left, Py_ssize_t right, const Py_ssize_t *degrees)
{
    return degrees[left] < degrees[right] || (degrees[left] == degrees[right] && left < right);
}


static void degree_heap_swap(Py_ssize_t *heap, Py_ssize_t *positions, Py_ssize_t left, Py_ssize_t right)
{
    Py_ssize_t tmp = heap[left];
    heap[left] = heap[right];
    heap[right] = tmp;
    positions[heap[left]] = left;
    positions[heap[right]] = right;
}


static void degree_heap_sift_up(
    Py_ssize_t *heap,
    Py_ssize_t *positions,
    const Py_ssize_t *degrees,
    Py_ssize_t pos
)
{
    while (pos > 0) {
        Py_ssize_t parent = (pos - 1) / 2;
        if (!degree_heap_less(heap[pos], heap[parent], degrees)) {
            break;
        }
        degree_heap_swap(heap, positions, pos, parent);
        pos = parent;
    }
}


static void degree_heap_sift_down(
    Py_ssize_t *heap,
    Py_ssize_t *positions,
    const Py_ssize_t *degrees,
    Py_ssize_t heap_size,
    Py_ssize_t pos
)
{
    while (1) {
        Py_ssize_t left = (2 * pos) + 1;
        Py_ssize_t right = left + 1;
        Py_ssize_t best = pos;
        if (left < heap_size && degree_heap_less(heap[left], heap[best], degrees)) {
            best = left;
        }
        if (right < heap_size && degree_heap_less(heap[right], heap[best], degrees)) {
            best = right;
        }
        if (best == pos) {
            break;
        }
        degree_heap_swap(heap, positions, pos, best);
        pos = best;
    }
}


PyObject *min_degree_cubic_order_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *q2;
    PyObject *q3;
    Py_ssize_t nwords;
    uint64_t *adjacency = NULL;
    uint64_t *remaining = NULL;
    uint64_t *best_neighbors = NULL;
    uint64_t *work = NULL;
    Py_ssize_t *degrees = NULL;
    Py_ssize_t *heap = NULL;
    Py_ssize_t *positions = NULL;
    PyObject *order = NULL;
    PyObject *width_obj = NULL;
    Py_ssize_t order_idx = 0;
    Py_ssize_t heap_size;
    Py_ssize_t max_scope = 1;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOO", &nvars, &q2, &q3)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }
    if (nvars == 0) {
        return empty_order_and_width_result();
    }

    if (build_cubic_adjacency(nvars, q2, q3, &nwords, &adjacency) < 0) {
        goto error;
    }
    remaining = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    best_neighbors = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    work = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    degrees = PyMem_Malloc((size_t) nvars * sizeof(Py_ssize_t));
    heap = PyMem_Malloc((size_t) nvars * sizeof(Py_ssize_t));
    positions = PyMem_Malloc((size_t) nvars * sizeof(Py_ssize_t));
    if (
        adjacency == NULL || remaining == NULL || best_neighbors == NULL || work == NULL ||
        degrees == NULL || heap == NULL || positions == NULL
    ) {
        PyErr_NoMemory();
        goto error;
    }

    initialize_remaining_set(remaining, nvars);
    heap_size = nvars;

    {
        Py_ssize_t var;
        for (var = 0; var < nvars; ++var) {
            degrees[var] = bitset_count(adjacency + ((size_t) var * (size_t) nwords), nwords);
            heap[var] = var;
            positions[var] = var;
        }
        for (var = nvars / 2; var > 0; --var) {
            degree_heap_sift_down(heap, positions, degrees, heap_size, var - 1);
        }
    }

    order = PyList_New(nvars);
    if (order == NULL) {
        goto error;
    }

    while (heap_size > 0) {
        Py_ssize_t best_var = heap[0];
        Py_ssize_t best_degree = degrees[best_var];
        Py_ssize_t scan_word;
        copy_remaining_neighbors(adjacency, remaining, nwords, best_var, best_neighbors);
        degree_heap_swap(heap, positions, 0, heap_size - 1);
        --heap_size;
        positions[best_var] = -1;
        if (heap_size > 0) {
            degree_heap_sift_down(heap, positions, degrees, heap_size, 0);
        }

        {
            PyObject *var_obj = PyLong_FromSsize_t(best_var);
            if (var_obj == NULL) {
                goto error;
            }
            PyList_SET_ITEM(order, order_idx, var_obj);
            ++order_idx;
        }

        {
            Py_ssize_t scope = bitset_count(best_neighbors, nwords) + 1;
            if (scope > max_scope) {
                max_scope = scope;
            }

            eliminate_var_from_graph(adjacency, remaining, best_neighbors, work, nwords, best_var);
        }

        for (scan_word = 0; scan_word < nwords; ++scan_word) {
            uint64_t word = best_neighbors[scan_word];
            while (word != 0) {
                Py_ssize_t neighbor = (scan_word << 6) + lowest_bit_index(word);
                Py_ssize_t inner_word;
                Py_ssize_t new_degree = 0;
                Py_ssize_t heap_pos;
                word &= word - 1;
                if (neighbor >= nvars || positions[neighbor] < 0) {
                    continue;
                }
                for (inner_word = 0; inner_word < nwords; ++inner_word) {
                    new_degree += popcount_u64(
                        adjacency[((size_t) neighbor * (size_t) nwords) + (size_t) inner_word]
                        & remaining[inner_word]
                    );
                }
                degrees[neighbor] = new_degree;
                heap_pos = positions[neighbor];
                degree_heap_sift_up(heap, positions, degrees, heap_pos);
                degree_heap_sift_down(
                    heap,
                    positions,
                    degrees,
                    heap_size,
                    positions[neighbor]
                );
            }
        }
    }

    width_obj = PyLong_FromSsize_t(max_scope);
    if (width_obj == NULL) {
        goto error;
    }

    PyMem_Free(adjacency);
    PyMem_Free(remaining);
    PyMem_Free(best_neighbors);
    PyMem_Free(work);
    PyMem_Free(degrees);
    PyMem_Free(heap);
    PyMem_Free(positions);

    {
        PyObject *result = PyTuple_Pack(2, order, width_obj);
        Py_DECREF(order);
        Py_DECREF(width_obj);
        return result;
    }

error:
    PyMem_Free(adjacency);
    PyMem_Free(remaining);
    PyMem_Free(best_neighbors);
    PyMem_Free(work);
    PyMem_Free(degrees);
    PyMem_Free(heap);
    PyMem_Free(positions);
    Py_XDECREF(order);
    Py_XDECREF(width_obj);
    return NULL;
}


PyObject *cubic_order_width_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *q2;
    PyObject *q3;
    PyObject *order_obj;
    PyObject *order_seq = NULL;
    Py_ssize_t nwords;
    uint64_t *adjacency = NULL;
    uint64_t *remaining = NULL;
    uint64_t *neighbors = NULL;
    uint64_t *work = NULL;
    PyObject *width_obj = NULL;
    Py_ssize_t remaining_count;
    Py_ssize_t max_scope = 0;
    Py_ssize_t order_len;
    Py_ssize_t order_idx;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOOO", &nvars, &q2, &q3, &order_obj)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }

    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (order_seq == NULL) {
        return NULL;
    }
    order_len = PySequence_Fast_GET_SIZE(order_seq);
    if (order_len != nvars) {
        PyErr_Format(PyExc_ValueError, "Expected elimination order of length %zd, received %zd.", nvars, order_len);
        goto error;
    }

    if (nvars == 0) {
        width_obj = PyLong_FromLong(0);
        Py_DECREF(order_seq);
        return width_obj;
    }

    if (build_cubic_adjacency(nvars, q2, q3, &nwords, &adjacency) < 0) {
        goto error;
    }
    remaining = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    neighbors = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    work = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    if (adjacency == NULL || remaining == NULL || neighbors == NULL || work == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    initialize_remaining_set(remaining, nvars);

    remaining_count = nvars;
    for (order_idx = 0; order_idx < order_len; ++order_idx) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(order_seq, order_idx));
        Py_ssize_t scope;
        uint64_t var_bit;
        if (var == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (var < 0 || var >= nvars) {
            PyErr_SetString(PyExc_ValueError, "Elimination order contains an out-of-range variable.");
            goto error;
        }
        var_bit = 1ULL << ((size_t) var & 63U);
        if ((remaining[word_index(var)] & var_bit) == 0) {
            PyErr_SetString(PyExc_ValueError, "Elimination order must contain each variable exactly once.");
            goto error;
        }

        copy_remaining_neighbors(adjacency, remaining, nwords, var, neighbors);

        scope = bitset_count(neighbors, nwords) + 1;
        if (scope > max_scope) {
            max_scope = scope;
        }

        eliminate_var_from_graph(adjacency, remaining, neighbors, work, nwords, var);
        --remaining_count;
    }

    if (remaining_count != 0) {
        PyErr_SetString(PyExc_ValueError, "Elimination order must contain each variable exactly once.");
        goto error;
    }

    width_obj = PyLong_FromSsize_t(max_scope);

error:
    PyMem_Free(adjacency);
    PyMem_Free(remaining);
    PyMem_Free(neighbors);
    PyMem_Free(work);
    Py_XDECREF(order_seq);
    return width_obj;
}


typedef struct {
    Py_ssize_t candidate;
    Py_ssize_t width;
    unsigned long long work;
} NativeCutsetExtensionScore;


static int compare_cutset_extension_scores(const void *left_ptr, const void *right_ptr)
{
    const NativeCutsetExtensionScore *left = (const NativeCutsetExtensionScore *) left_ptr;
    const NativeCutsetExtensionScore *right = (const NativeCutsetExtensionScore *) right_ptr;
    if (left->width != right->width) {
        return (left->width < right->width) ? -1 : 1;
    }
    if (left->work != right->work) {
        return (left->work < right->work) ? -1 : 1;
    }
    if (left->candidate != right->candidate) {
        return (left->candidate < right->candidate) ? -1 : 1;
    }
    return 0;
}


PyObject *rank_q3_free_cutset_extensions_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *q2;
    PyObject *selected_obj;
    PyObject *candidates_obj;
    PyObject *order_hint_obj = Py_None;
    PyObject *selected_seq = NULL;
    PyObject *candidates_seq = NULL;
    PyObject *order_hint_seq = NULL;
    char *base_removed = NULL;
    char *removed = NULL;
    Py_ssize_t *edge_left = NULL;
    Py_ssize_t *edge_right = NULL;
    Py_ssize_t edge_count = 0;
    Py_ssize_t edge_capacity = 0;
    Py_ssize_t *hint_order = NULL;
    Py_ssize_t *position_map = NULL;
    long long *diff = NULL;
    NativeCutsetExtensionScore *scores = NULL;
    PyObject *results = NULL;
    Py_ssize_t candidate_count;
    Py_ssize_t idx;
    PyObject *key;
    PyObject *value;
    Py_ssize_t pos;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOOO|O", &nvars, &q2, &selected_obj, &candidates_obj, &order_hint_obj)) {
        return NULL;
    }
    if (!PyDict_Check(q2)) {
        PyErr_SetString(PyExc_TypeError, "q2 must be a dict.");
        return NULL;
    }
    selected_seq = PySequence_Fast(selected_obj, "selected_vars must be a sequence.");
    candidates_seq = PySequence_Fast(candidates_obj, "candidate_vars must be a sequence.");
    if (selected_seq == NULL || candidates_seq == NULL) {
        goto cleanup;
    }
    if (order_hint_obj != Py_None) {
        order_hint_seq = PySequence_Fast(order_hint_obj, "order_hint must be a sequence.");
        if (order_hint_seq == NULL) {
            goto cleanup;
        }
        if (PySequence_Fast_GET_SIZE(order_hint_seq) != nvars) {
            PyErr_SetString(PyExc_ValueError, "order_hint must have length nvars.");
            goto cleanup;
        }
    }

    candidate_count = PySequence_Fast_GET_SIZE(candidates_seq);
    results = PyList_New(candidate_count);
    if (results == NULL) {
        goto cleanup;
    }
    if (candidate_count == 0) {
        goto cleanup;
    }

    base_removed = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(char));
    removed = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(char));
    position_map = PyMem_Malloc((size_t) (nvars > 0 ? nvars : 1) * sizeof(Py_ssize_t));
    diff = PyMem_Malloc((size_t) (nvars > 0 ? nvars : 1) * sizeof(long long));
    scores = PyMem_Malloc((size_t) candidate_count * sizeof(NativeCutsetExtensionScore));
    edge_capacity = PyDict_Size(q2);
    if (edge_capacity < 1) {
        edge_capacity = 1;
    }
    edge_left = PyMem_Malloc((size_t) edge_capacity * sizeof(Py_ssize_t));
    edge_right = PyMem_Malloc((size_t) edge_capacity * sizeof(Py_ssize_t));
    if (
        base_removed == NULL || removed == NULL || position_map == NULL ||
        diff == NULL || scores == NULL || edge_left == NULL || edge_right == NULL
    ) {
        PyErr_NoMemory();
        goto cleanup;
    }

    if (order_hint_seq != NULL) {
        hint_order = PyMem_Malloc((size_t) nvars * sizeof(Py_ssize_t));
        if (hint_order == NULL) {
            PyErr_NoMemory();
            goto cleanup;
        }
        for (idx = 0; idx < nvars; ++idx) {
            Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(order_hint_seq, idx));
            if (var == -1 && PyErr_Occurred()) {
                goto cleanup;
            }
            if (var < 0 || var >= nvars) {
                PyErr_SetString(PyExc_ValueError, "order_hint contains an out-of-range variable.");
                goto cleanup;
            }
            hint_order[idx] = var;
        }
    }

    for (idx = 0; idx < PySequence_Fast_GET_SIZE(selected_seq); ++idx) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(selected_seq, idx));
        if (var == -1 && PyErr_Occurred()) {
            goto cleanup;
        }
        if (var < 0 || var >= nvars) {
            PyErr_SetString(PyExc_ValueError, "selected_vars contains an out-of-range variable.");
            goto cleanup;
        }
        base_removed[var] = 1;
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long long coeff;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto cleanup;
        }
        coeff = PyLong_AsLongLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto cleanup;
        }
        if (coeff == 0) {
            continue;
        }
        edge_left[edge_count] = left;
        edge_right[edge_count] = right;
        ++edge_count;
    }

    for (idx = 0; idx < candidate_count; ++idx) {
        Py_ssize_t candidate = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(candidates_seq, idx));
        Py_ssize_t best_width = PY_SSIZE_T_MAX;
        unsigned long long best_work = (unsigned long long) -1;
        int mode;
        if (candidate == -1 && PyErr_Occurred()) {
            goto cleanup;
        }
        if (candidate < 0 || candidate >= nvars) {
            PyErr_SetString(PyExc_ValueError, "candidate_vars contains an out-of-range variable.");
            goto cleanup;
        }

        memcpy(removed, base_removed, (size_t) nvars * sizeof(char));
        removed[candidate] = 1;

        for (mode = 0; mode < (order_hint_seq != NULL ? 3 : 2); ++mode) {
            Py_ssize_t remaining_count = 0;
            Py_ssize_t edge_idx;
            Py_ssize_t max_cut = 0;
            long long running = 0;
            unsigned long long work;

            if (mode == 0 && order_hint_seq != NULL) {
                for (edge_idx = 0; edge_idx < nvars; ++edge_idx) {
                    position_map[edge_idx] = -1;
                }
                for (edge_idx = 0; edge_idx < nvars; ++edge_idx) {
                    Py_ssize_t var = hint_order[edge_idx];
                    if (!removed[var]) {
                        position_map[var] = remaining_count++;
                    }
                }
            } else if ((mode == 0 && order_hint_seq == NULL) || mode == 1) {
                for (edge_idx = 0; edge_idx < nvars; ++edge_idx) {
                    position_map[edge_idx] = -1;
                }
                for (edge_idx = 0; edge_idx < nvars; ++edge_idx) {
                    if (!removed[edge_idx]) {
                        position_map[edge_idx] = remaining_count++;
                    }
                }
            } else {
                for (edge_idx = 0; edge_idx < nvars; ++edge_idx) {
                    position_map[edge_idx] = -1;
                }
                for (edge_idx = nvars - 1; edge_idx >= 0; --edge_idx) {
                    if (!removed[edge_idx]) {
                        position_map[edge_idx] = remaining_count++;
                    }
                    if (edge_idx == 0) {
                        break;
                    }
                }
            }

            if (remaining_count <= 0) {
                best_width = 0;
                best_work = 1ULL;
                continue;
            }
            memset(diff, 0, (size_t) remaining_count * sizeof(long long));
            work = 0ULL;

            for (edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
                Py_ssize_t left = edge_left[edge_idx];
                Py_ssize_t right = edge_right[edge_idx];
                Py_ssize_t left_pos;
                Py_ssize_t right_pos;
                if (removed[left] || removed[right]) {
                    continue;
                }
                left_pos = position_map[left];
                right_pos = position_map[right];
                if (left_pos < 0 || right_pos < 0 || left_pos == right_pos) {
                    continue;
                }
                if (left_pos > right_pos) {
                    Py_ssize_t tmp = left_pos;
                    left_pos = right_pos;
                    right_pos = tmp;
                }
                diff[left_pos] += 1;
                diff[right_pos] -= 1;
                if (work != (unsigned long long) -1) {
                    work += 1ULL;
                }
            }

            for (edge_idx = 0; edge_idx + 1 < remaining_count; ++edge_idx) {
                running += diff[edge_idx];
                if ((Py_ssize_t) running > max_cut) {
                    max_cut = (Py_ssize_t) running;
                }
            }

            {
                Py_ssize_t width = max_cut + 1;
                unsigned long long scale = (width >= (Py_ssize_t) (8 * sizeof(unsigned long long)))
                    ? (unsigned long long) -1
                    : (1ULL << (unsigned long long) width);
                if (scale == (unsigned long long) -1 || work == (unsigned long long) -1) {
                    work = (unsigned long long) -1;
                } else {
                    unsigned long long base = work + (unsigned long long) remaining_count;
                    if (base > ((unsigned long long) -1) / scale) {
                        work = (unsigned long long) -1;
                    } else {
                        work = base * scale;
                    }
                }
                if (width < best_width || (width == best_width && work < best_work)) {
                    best_width = width;
                    best_work = work;
                }
            }
        }

        scores[idx].candidate = candidate;
        scores[idx].width = best_width;
        scores[idx].work = best_work;
    }

    qsort(scores, (size_t) candidate_count, sizeof(NativeCutsetExtensionScore), compare_cutset_extension_scores);
    for (idx = 0; idx < candidate_count; ++idx) {
        PyObject *item = PyTuple_New(3);
        PyObject *cand_obj;
        PyObject *width_obj;
        PyObject *work_obj;
        if (item == NULL) {
            goto cleanup;
        }
        cand_obj = PyLong_FromSsize_t(scores[idx].candidate);
        width_obj = PyLong_FromSsize_t(scores[idx].width);
        work_obj = PyLong_FromUnsignedLongLong(scores[idx].work);
        if (cand_obj == NULL || width_obj == NULL || work_obj == NULL) {
            Py_XDECREF(cand_obj);
            Py_XDECREF(width_obj);
            Py_XDECREF(work_obj);
            Py_DECREF(item);
            goto cleanup;
        }
        PyTuple_SET_ITEM(item, 0, cand_obj);
        PyTuple_SET_ITEM(item, 1, width_obj);
        PyTuple_SET_ITEM(item, 2, work_obj);
        PyList_SET_ITEM(results, idx, item);
    }

cleanup:
    PyMem_Free(base_removed);
    PyMem_Free(removed);
    PyMem_Free(edge_left);
    PyMem_Free(edge_right);
    PyMem_Free(hint_order);
    PyMem_Free(position_map);
    PyMem_Free(diff);
    PyMem_Free(scores);
    Py_XDECREF(selected_seq);
    Py_XDECREF(candidates_seq);
    Py_XDECREF(order_hint_seq);
    return results;
}
