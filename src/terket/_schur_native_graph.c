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
    best_neighbors = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    work = PyMem_Calloc((size_t) nwords, sizeof(uint64_t));
    if (adjacency == NULL || remaining == NULL || best_neighbors == NULL || work == NULL) {
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
        int has_best = 0;
        Py_ssize_t word_idx;

        for (word_idx = 0; word_idx < nwords; ++word_idx) {
            uint64_t word = remaining[word_idx];
            while (word != 0) {
                Py_ssize_t var = (word_idx << 6) + lowest_bit_index(word);
                Py_ssize_t degree;
                Py_ssize_t inner_word;

                word &= word - 1;
                if (var >= nvars) {
                    continue;
                }

                degree = 0;
                for (inner_word = 0; inner_word < nwords; ++inner_word) {
                    uint64_t neighbors_word = adjacency[((size_t) var * (size_t) nwords) + (size_t) inner_word] & remaining[inner_word];
                    degree += popcount_u64(neighbors_word);
                }

                if (!has_best || degree < best_degree || (degree == best_degree && var < best_var)) {
                    has_best = 1;
                    best_var = var;
                    best_degree = degree;
                    for (inner_word = 0; inner_word < nwords; ++inner_word) {
                        best_neighbors[inner_word] = adjacency[((size_t) var * (size_t) nwords) + (size_t) inner_word] & remaining[inner_word];
                    }
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
    PyMem_Free(best_neighbors);
    PyMem_Free(work);
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
