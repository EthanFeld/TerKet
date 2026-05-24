#include "_schur_native_internal.h"
#include "native_algebra_helpers.inc"

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


