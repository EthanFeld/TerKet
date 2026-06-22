#include "_schur_native_internal.h"
#include "native_algebra_helpers.inc"

static int adjacency_set(PyObject *neighbors, Py_ssize_t var, long coeff)
{
    PyObject *key = PyLong_FromSsize_t(var);
    PyObject *value;
    int status;
    if (key == NULL) {
        return -1;
    }
    coeff = positive_mod(coeff, 4);
    if (coeff == 0) {
        status = PyDict_DelItem(neighbors, key);
        if (status < 0 && PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            status = 0;
        }
        Py_DECREF(key);
        return status;
    }
    value = PyLong_FromLong(coeff);
    if (value == NULL) {
        Py_DECREF(key);
        return -1;
    }
    status = PyDict_SetItem(neighbors, key, value);
    Py_DECREF(key);
    Py_DECREF(value);
    return status;
}

static long adjacency_get(PyObject *neighbors, Py_ssize_t var)
{
    PyObject *key = PyLong_FromSsize_t(var);
    PyObject *value;
    long coeff;
    if (key == NULL) {
        return -1;
    }
    value = PyDict_GetItemWithError(neighbors, key);
    Py_DECREF(key);
    if (value == NULL) {
        return PyErr_Occurred() ? -1 : 0;
    }
    coeff = PyLong_AsLong(value);
    return coeff;
}

PyObject *elim_sparse_quadratics_batch_terms_native(PyObject *self, PyObject *args)
{
    long q0_residue;
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *candidates_obj;
    PyObject *q1_seq = NULL;
    PyObject *candidates_seq = NULL;
    PyObject *adjacency = NULL;
    PyObject *new_q1_obj = NULL;
    PyObject *new_q2 = NULL;
    PyObject *removed_obj = NULL;
    unsigned char *q1 = NULL;
    unsigned char *candidate = NULL;
    unsigned char *removed = NULL;
    Py_ssize_t *remap = NULL;
    Py_ssize_t nvars;
    Py_ssize_t removed_count = 0;
    Py_ssize_t idx;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;

    (void) self;
    if (!PyArg_ParseTuple(args, "lOOO", &q0_residue, &q1_obj, &q2, &candidates_obj)) {
        return NULL;
    }
    if (!PyDict_Check(q2)) {
        PyErr_SetString(PyExc_TypeError, "q2 must be a dict.");
        return NULL;
    }
    q1_seq = PySequence_Fast(q1_obj, "q1 must be a sequence.");
    candidates_seq = PySequence_Fast(candidates_obj, "candidates must be a sequence.");
    if (q1_seq == NULL || candidates_seq == NULL) {
        goto error;
    }
    nvars = PySequence_Fast_GET_SIZE(q1_seq);
    q1 = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(unsigned char));
    candidate = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(unsigned char));
    removed = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(unsigned char));
    remap = PyMem_Malloc((size_t) (nvars > 0 ? nvars : 1) * sizeof(Py_ssize_t));
    adjacency = PyList_New(nvars);
    if (q1 == NULL || candidate == NULL || removed == NULL || remap == NULL || adjacency == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    for (idx = 0; idx < nvars; ++idx) {
        long coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, idx));
        PyObject *neighbors;
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        q1[idx] = (unsigned char) positive_mod(coeff, 8);
        remap[idx] = -1;
        neighbors = PyDict_New();
        if (neighbors == NULL) {
            goto error;
        }
        PyList_SET_ITEM(adjacency, idx, neighbors);
    }
    for (idx = 0; idx < PySequence_Fast_GET_SIZE(candidates_seq); ++idx) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(candidates_seq, idx));
        if (var == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (var >= 0 && var < nvars) {
            candidate[var] = 1;
        }
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long coeff;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        if (left < 0 || right < 0 || left >= nvars || right >= nvars || left == right) {
            PyErr_SetString(PyExc_IndexError, "q2 key contains invalid variable index.");
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        coeff = positive_mod(coeff, 4);
        if (coeff != 0 && (
            adjacency_set(PyList_GET_ITEM(adjacency, left), right, coeff) < 0 ||
            adjacency_set(PyList_GET_ITEM(adjacency, right), left, coeff) < 0
        )) {
            goto error;
        }
    }

    for (idx = 0; idx < nvars; ++idx) {
        PyObject *neighbors;
        Py_ssize_t degree;
        Py_ssize_t scan_pos = 0;
        PyObject *neighbor_key;
        PyObject *coeff_obj;
        Py_ssize_t coupled_vars[2];
        long coupled_coeffs[2];
        Py_ssize_t coupled_count = 0;
        int reduced_residue;
        int sign;
        if (!candidate[idx] || removed[idx]) {
            continue;
        }
        neighbors = PyList_GET_ITEM(adjacency, idx);
        degree = PyDict_Size(neighbors);
        if (degree < 0) {
            goto error;
        }
        if (degree > 2 || (q1[idx] & 1)) {
            continue;
        }
        reduced_residue = (q1[idx] / 2) & 3;
        if (reduced_residue != 1 && reduced_residue != 3) {
            continue;
        }
        while (PyDict_Next(neighbors, &scan_pos, &neighbor_key, &coeff_obj)) {
            Py_ssize_t neighbor = PyLong_AsSsize_t(neighbor_key);
            long coeff = PyLong_AsLong(coeff_obj);
            if ((neighbor == -1 || coeff == -1) && PyErr_Occurred()) {
                goto error;
            }
            if ((coeff & 1) || coupled_count >= 2) {
                coupled_count = 3;
                break;
            }
            coupled_vars[coupled_count] = neighbor;
            coupled_coeffs[coupled_count] = positive_mod(coeff, 4);
            ++coupled_count;
        }
        if (coupled_count > 2) {
            continue;
        }
        sign = reduced_residue == 1 ? -1 : 1;
        q0_residue = positive_mod(q0_residue + (reduced_residue == 1 ? 1 : 7), 8);
        for (pos = 0; pos < coupled_count; ++pos) {
            Py_ssize_t neighbor = coupled_vars[pos];
            q1[neighbor] = (unsigned char) positive_mod(q1[neighbor] + sign * coupled_coeffs[pos], 8);
        }
        if (coupled_count == 2) {
            Py_ssize_t left = coupled_vars[0];
            Py_ssize_t right = coupled_vars[1];
            long correction = positive_mod((coupled_coeffs[0] * coupled_coeffs[1]) / 2, 4);
            long old_coeff = adjacency_get(PyList_GET_ITEM(adjacency, left), right);
            if (old_coeff < 0 && PyErr_Occurred()) {
                goto error;
            }
            if (
                adjacency_set(PyList_GET_ITEM(adjacency, left), right, old_coeff + correction) < 0 ||
                adjacency_set(PyList_GET_ITEM(adjacency, right), left, old_coeff + correction) < 0
            ) {
                goto error;
            }
        }
        for (pos = 0; pos < coupled_count; ++pos) {
            Py_ssize_t neighbor = coupled_vars[pos];
            if (
                adjacency_set(neighbors, neighbor, 0) < 0 ||
                adjacency_set(PyList_GET_ITEM(adjacency, neighbor), idx, 0) < 0
            ) {
                goto error;
            }
        }
        removed[idx] = 1;
        ++removed_count;
    }

    new_q1_obj = PyList_New(nvars - removed_count);
    removed_obj = PyTuple_New(removed_count);
    new_q2 = PyDict_New();
    if (new_q1_obj == NULL || removed_obj == NULL || new_q2 == NULL) {
        goto error;
    }
    {
        Py_ssize_t out_idx = 0;
        Py_ssize_t removed_idx = 0;
        for (idx = 0; idx < nvars; ++idx) {
            PyObject *item;
            if (removed[idx]) {
                item = PyLong_FromSsize_t(idx);
                if (item == NULL) {
                    goto error;
                }
                PyTuple_SET_ITEM(removed_obj, removed_idx++, item);
                continue;
            }
            remap[idx] = out_idx;
            item = PyLong_FromLong((long) q1[idx]);
            if (item == NULL) {
                goto error;
            }
            PyList_SET_ITEM(new_q1_obj, out_idx++, item);
        }
    }
    for (idx = 0; idx < nvars; ++idx) {
        PyObject *neighbors;
        Py_ssize_t scan_pos = 0;
        PyObject *neighbor_key;
        PyObject *coeff_obj;
        if (removed[idx]) {
            continue;
        }
        neighbors = PyList_GET_ITEM(adjacency, idx);
        while (PyDict_Next(neighbors, &scan_pos, &neighbor_key, &coeff_obj)) {
            Py_ssize_t neighbor = PyLong_AsSsize_t(neighbor_key);
            PyObject *pair;
            if (neighbor == -1 && PyErr_Occurred()) {
                goto error;
            }
            if (neighbor <= idx || removed[neighbor]) {
                continue;
            }
            pair = Py_BuildValue("(nn)", remap[idx], remap[neighbor]);
            if (pair == NULL || PyDict_SetItem(new_q2, pair, coeff_obj) < 0) {
                Py_XDECREF(pair);
                goto error;
            }
            Py_DECREF(pair);
        }
    }

    PyMem_Free(q1);
    PyMem_Free(candidate);
    PyMem_Free(removed);
    PyMem_Free(remap);
    Py_DECREF(q1_seq);
    Py_DECREF(candidates_seq);
    Py_DECREF(adjacency);
    return Py_BuildValue("lNNN", q0_residue, new_q1_obj, new_q2, removed_obj);

error:
    PyMem_Free(q1);
    PyMem_Free(candidate);
    PyMem_Free(removed);
    PyMem_Free(remap);
    Py_XDECREF(q1_seq);
    Py_XDECREF(candidates_seq);
    Py_XDECREF(adjacency);
    Py_XDECREF(new_q1_obj);
    Py_XDECREF(new_q2);
    Py_XDECREF(removed_obj);
    return NULL;
}

PyObject *elim_single_partner_constraint_terms_native(PyObject *self, PyObject *args)
{
    long q0_residue;
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *q3;
    Py_ssize_t k;
    Py_ssize_t p;
    int target;
    PyObject *q1_seq = NULL;
    PyObject *packed_q2 = NULL;
    PyObject *packed_q3 = NULL;
    unsigned char *new_q1 = NULL;
    Py_ssize_t *remap = NULL;
    PyObject *new_q1_obj = NULL;
    PyObject *new_q2_obj = NULL;
    PyObject *new_q3_obj = NULL;
    Py_ssize_t nvars;
    Py_ssize_t nn;
    Py_ssize_t idx;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;
    int q1_p = 0;

    (void) self;

    if (!PyArg_ParseTuple(args, "lOOOnni", &q0_residue, &q1_obj, &q2, &q3, &k, &p, &target)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }

    q1_seq = PySequence_Fast(q1_obj, "q1 must be a sequence.");
    if (q1_seq == NULL) {
        goto error;
    }
    nvars = PySequence_Fast_GET_SIZE(q1_seq);
    if (k < 0 || k >= nvars || p < 0 || p >= nvars || k == p) {
        PyErr_SetString(PyExc_IndexError, "k and p must be distinct in-range indices.");
        goto error;
    }
    if (nvars < 2) {
        PyErr_SetString(PyExc_ValueError, "Need at least two variables to eliminate a partner constraint.");
        goto error;
    }

    nn = nvars - 2;
    remap = PyMem_Malloc((size_t) nvars * sizeof(Py_ssize_t));
    new_q1 = PyMem_Calloc((size_t) (nn > 0 ? nn : 1), sizeof(unsigned char));
    packed_q2 = PyDict_New();
    packed_q3 = PyDict_New();
    if (remap == NULL || new_q1 == NULL || packed_q2 == NULL || packed_q3 == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    for (idx = 0; idx < nvars; ++idx) {
        remap[idx] = -1;
    }

    {
        Py_ssize_t out_idx = 0;
        for (idx = 0; idx < nvars; ++idx) {
            long coeff;
            if (idx == k || idx == p) {
                continue;
            }
            coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, idx));
            if (coeff == -1 && PyErr_Occurred()) {
                goto error;
            }
            remap[idx] = out_idx;
            new_q1[out_idx] = (unsigned char) positive_mod(coeff, 8);
            ++out_idx;
        }
    }

    {
        long coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, p));
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        q1_p = positive_mod(coeff, 8);
    }
    q0_residue = positive_mod(q0_residue + (target ? q1_p : 0), 8);

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long coeff;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        coeff = positive_mod(coeff, 4);
        if (coeff == 0) {
            continue;
        }
        if (left == k || right == k) {
            continue;
        }
        if (left == p || right == p) {
            Py_ssize_t other;
            if (!target) {
                continue;
            }
            other = (left == p) ? right : left;
            if (remap[other] < 0) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to remap q2 neighbor.");
                goto error;
            }
            new_q1[remap[other]] = (unsigned char) ((new_q1[remap[other]] + (2 * coeff)) & 7);
            continue;
        }
        if (update_packed_mod_dict(
            packed_q2,
            (((uint64_t) remap[left]) << 32) | (uint32_t) remap[right],
            coeff,
            4L
        ) < 0) {
            goto error;
        }
    }

    pos = 0;
    while (PyDict_Next(q3, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        long coeff;
        if (parse_triple_key(key, &a, &b, &c) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        coeff = positive_mod(coeff, 2);
        if (coeff == 0) {
            continue;
        }
        if (a == k || b == k || c == k) {
            continue;
        }
        if (a == p || b == p || c == p) {
            Py_ssize_t left = -1;
            Py_ssize_t right = -1;
            Py_ssize_t mapped_left;
            Py_ssize_t mapped_right;
            if (!target) {
                continue;
            }
            if (a != p) {
                left = a;
            } else if (b != p) {
                left = b;
            }
            if (c != p) {
                right = c;
            } else if (b != p && left != b) {
                right = b;
            }
            if (left < 0 || right < 0) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to extract q3 neighbors.");
                goto error;
            }
            mapped_left = remap[left];
            mapped_right = remap[right];
            if (mapped_left < 0 || mapped_right < 0) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to remap q3 neighbors.");
                goto error;
            }
            if (mapped_left > mapped_right) {
                Py_ssize_t tmp = mapped_left;
                mapped_left = mapped_right;
                mapped_right = tmp;
            }
            if (update_packed_mod_dict(
                packed_q2,
                (((uint64_t) mapped_left) << 32) | (uint32_t) mapped_right,
                2L * coeff,
                4L
            ) < 0) {
                goto error;
            }
            continue;
        }
        if (update_packed_mod_dict(
            packed_q3,
            (((uint64_t) remap[a]) << 42) | (((uint64_t) remap[b]) << 21) | (uint64_t) remap[c],
            coeff,
            2L
        ) < 0) {
            goto error;
        }
    }

    new_q1_obj = PyList_New(nn);
    if (new_q1_obj == NULL) {
        goto error;
    }
    for (idx = 0; idx < nn; ++idx) {
        PyObject *value_obj = PyLong_FromLong((long) new_q1[idx]);
        if (value_obj == NULL) {
            goto error;
        }
        PyList_SET_ITEM(new_q1_obj, idx, value_obj);
    }

    new_q2_obj = expand_packed_pairs(packed_q2);
    new_q3_obj = expand_packed_triples(packed_q3);
    if (new_q2_obj == NULL || new_q3_obj == NULL) {
        goto error;
    }

    PyMem_Free(remap);
    PyMem_Free(new_q1);
    Py_DECREF(q1_seq);
    Py_DECREF(packed_q2);
    Py_DECREF(packed_q3);
    return Py_BuildValue("lNNN", q0_residue, new_q1_obj, new_q2_obj, new_q3_obj);

error:
    PyMem_Free(remap);
    PyMem_Free(new_q1);
    Py_XDECREF(q1_seq);
    Py_XDECREF(packed_q2);
    Py_XDECREF(packed_q3);
    Py_XDECREF(new_q1_obj);
    Py_XDECREF(new_q2_obj);
    Py_XDECREF(new_q3_obj);
    return NULL;
}


PyObject *elim_two_partner_constraint_terms_native(PyObject *self, PyObject *args)
{
    long q0_residue;
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *q3;
    Py_ssize_t k;
    Py_ssize_t keep;
    Py_ssize_t remove;
    int target;
    PyObject *q1_seq = NULL;
    PyObject *packed_q2 = NULL;
    PyObject *packed_q3 = NULL;
    unsigned char *new_q1 = NULL;
    Py_ssize_t *remap = NULL;
    PyObject *new_q1_obj = NULL;
    PyObject *new_q2_obj = NULL;
    PyObject *new_q3_obj = NULL;
    Py_ssize_t nvars;
    Py_ssize_t nn;
    Py_ssize_t keep_new;
    Py_ssize_t idx;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;
    int q1_remove = 0;

    (void) self;

    if (!PyArg_ParseTuple(args, "lOOOnnni", &q0_residue, &q1_obj, &q2, &q3, &k, &keep, &remove, &target)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }

    q1_seq = PySequence_Fast(q1_obj, "q1 must be a sequence.");
    if (q1_seq == NULL) {
        goto error;
    }
    nvars = PySequence_Fast_GET_SIZE(q1_seq);
    if (
        k < 0 || k >= nvars ||
        keep < 0 || keep >= nvars ||
        remove < 0 || remove >= nvars ||
        k == keep || k == remove || keep == remove
    ) {
        PyErr_SetString(PyExc_IndexError, "k, keep, and remove must be distinct in-range indices.");
        goto error;
    }
    if (nvars < 2) {
        PyErr_SetString(PyExc_ValueError, "Need at least two variables to eliminate a two-partner constraint.");
        goto error;
    }

    nn = nvars - 2;
    remap = PyMem_Malloc((size_t) nvars * sizeof(Py_ssize_t));
    new_q1 = PyMem_Calloc((size_t) (nn > 0 ? nn : 1), sizeof(unsigned char));
    packed_q2 = PyDict_New();
    packed_q3 = PyDict_New();
    if (remap == NULL || new_q1 == NULL || packed_q2 == NULL || packed_q3 == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    for (idx = 0; idx < nvars; ++idx) {
        remap[idx] = -1;
    }

    {
        Py_ssize_t out_idx = 0;
        for (idx = 0; idx < nvars; ++idx) {
            long coeff;
            if (idx == k || idx == remove) {
                continue;
            }
            coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, idx));
            if (coeff == -1 && PyErr_Occurred()) {
                goto error;
            }
            remap[idx] = out_idx;
            new_q1[out_idx] = (unsigned char) positive_mod(coeff, 8);
            ++out_idx;
        }
    }
    keep_new = remap[keep];
    if (keep_new < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to remap keep variable.");
        goto error;
    }

    {
        long coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, remove));
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        q1_remove = positive_mod(coeff, 8);
    }
    if (target) {
        q0_residue = positive_mod(q0_residue + q1_remove, 8);
        new_q1[keep_new] = (unsigned char) positive_mod(new_q1[keep_new] - q1_remove, 8);
    } else {
        new_q1[keep_new] = (unsigned char) positive_mod(new_q1[keep_new] + q1_remove, 8);
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        Py_ssize_t other;
        Py_ssize_t mapped_left;
        Py_ssize_t mapped_right;
        long coeff;
        long pair_coeff;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        coeff = positive_mod(coeff, 4);
        if (coeff == 0 || left == k || right == k) {
            continue;
        }
        if (left != remove && right != remove) {
            mapped_left = remap[left];
            mapped_right = remap[right];
            if (mapped_left < 0 || mapped_right < 0) {
                continue;
            }
            if (update_packed_mod_dict(
                packed_q2,
                (((uint64_t) mapped_left) << 32) | (uint32_t) mapped_right,
                coeff,
                4L
            ) < 0) {
                goto error;
            }
            continue;
        }

        other = (left == remove) ? right : left;
        if (other == keep) {
            if (!target) {
                new_q1[keep_new] = (unsigned char) ((new_q1[keep_new] + (2 * coeff)) & 7);
            }
            continue;
        }
        if (other == k) {
            continue;
        }
        mapped_right = remap[other];
        if (mapped_right < 0) {
            continue;
        }
        if (target) {
            new_q1[mapped_right] = (unsigned char) ((new_q1[mapped_right] + (2 * coeff)) & 7);
            pair_coeff = positive_mod(-coeff, 4);
        } else {
            pair_coeff = coeff;
        }
        mapped_left = keep_new;
        if (mapped_left > mapped_right) {
            Py_ssize_t tmp = mapped_left;
            mapped_left = mapped_right;
            mapped_right = tmp;
        }
        if (update_packed_mod_dict(
            packed_q2,
            (((uint64_t) mapped_left) << 32) | (uint32_t) mapped_right,
            pair_coeff,
            4L
        ) < 0) {
            goto error;
        }
    }

    pos = 0;
    while (PyDict_Next(q3, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        Py_ssize_t left = -1;
        Py_ssize_t right = -1;
        Py_ssize_t mapped_left;
        Py_ssize_t mapped_right;
        Py_ssize_t mapped_keep;
        long coeff;
        if (parse_triple_key(key, &a, &b, &c) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        coeff = positive_mod(coeff, 2);
        if (coeff == 0 || a == k || b == k || c == k) {
            continue;
        }
        if (a != remove && b != remove && c != remove) {
            if (update_packed_mod_dict(
                packed_q3,
                (((uint64_t) remap[a]) << 42) | (((uint64_t) remap[b]) << 21) | (uint64_t) remap[c],
                coeff,
                2L
            ) < 0) {
                goto error;
            }
            continue;
        }

        if (a != remove) {
            if (left < 0) {
                left = a;
            } else {
                right = a;
            }
        }
        if (b != remove) {
            if (left < 0) {
                left = b;
            } else {
                right = b;
            }
        }
        if (c != remove) {
            if (left < 0) {
                left = c;
            } else {
                right = c;
            }
        }
        if (left < 0 || right < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to extract q3 neighbors.");
            goto error;
        }

        if (left == keep || right == keep) {
            Py_ssize_t other = (left == keep) ? right : left;
            if (target) {
                continue;
            }
            mapped_left = keep_new;
            mapped_right = remap[other];
            if (mapped_right < 0) {
                continue;
            }
            if (mapped_left > mapped_right) {
                Py_ssize_t tmp = mapped_left;
                mapped_left = mapped_right;
                mapped_right = tmp;
            }
            if (update_packed_mod_dict(
                packed_q2,
                (((uint64_t) mapped_left) << 32) | (uint32_t) mapped_right,
                2L * coeff,
                4L
            ) < 0) {
                goto error;
            }
            continue;
        }

        mapped_left = remap[left];
        mapped_right = remap[right];
        mapped_keep = keep_new;
        if (mapped_left < 0 || mapped_right < 0) {
            continue;
        }
        if (target) {
            Py_ssize_t pair_left = mapped_left;
            Py_ssize_t pair_right = mapped_right;
            if (pair_left > pair_right) {
                Py_ssize_t tmp = pair_left;
                pair_left = pair_right;
                pair_right = tmp;
            }
            if (update_packed_mod_dict(
                packed_q2,
                (((uint64_t) pair_left) << 32) | (uint32_t) pair_right,
                2L * coeff,
                4L
            ) < 0) {
                goto error;
            }
        }

        if (mapped_keep > mapped_left) {
            Py_ssize_t tmp = mapped_keep;
            mapped_keep = mapped_left;
            mapped_left = tmp;
        }
        if (mapped_left > mapped_right) {
            Py_ssize_t tmp = mapped_left;
            mapped_left = mapped_right;
            mapped_right = tmp;
        }
        if (mapped_keep > mapped_left) {
            Py_ssize_t tmp = mapped_keep;
            mapped_keep = mapped_left;
            mapped_left = tmp;
        }
        if (update_packed_mod_dict(
            packed_q3,
            (((uint64_t) mapped_keep) << 42) | (((uint64_t) mapped_left) << 21) | (uint64_t) mapped_right,
            coeff,
            2L
        ) < 0) {
            goto error;
        }
    }

    new_q1_obj = PyList_New(nn);
    if (new_q1_obj == NULL) {
        goto error;
    }
    for (idx = 0; idx < nn; ++idx) {
        PyObject *value_obj = PyLong_FromLong((long) new_q1[idx]);
        if (value_obj == NULL) {
            goto error;
        }
        PyList_SET_ITEM(new_q1_obj, idx, value_obj);
    }

    new_q2_obj = expand_packed_pairs(packed_q2);
    new_q3_obj = expand_packed_triples(packed_q3);
    if (new_q2_obj == NULL || new_q3_obj == NULL) {
        goto error;
    }

    PyMem_Free(remap);
    PyMem_Free(new_q1);
    Py_DECREF(q1_seq);
    Py_DECREF(packed_q2);
    Py_DECREF(packed_q3);
    return Py_BuildValue("lNNN", q0_residue, new_q1_obj, new_q2_obj, new_q3_obj);

error:
    PyMem_Free(remap);
    PyMem_Free(new_q1);
    Py_XDECREF(q1_seq);
    Py_XDECREF(packed_q2);
    Py_XDECREF(packed_q3);
    Py_XDECREF(new_q1_obj);
    Py_XDECREF(new_q2_obj);
    Py_XDECREF(new_q3_obj);
    return NULL;
}


