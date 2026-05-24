#include "_schur_native_internal.h"
#include "native_algebra_helpers.inc"

PyObject *aff_compose_terms_native(PyObject *self, PyObject *args)
{
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *q3;
    PyObject *shift_mask;
    PyObject *row_masks_obj;
    PyObject *row_masks = NULL;
    PyObject *packed_q2 = NULL;
    PyObject *packed_q3 = NULL;
    PyObject *new_q1_obj = NULL;
    PyObject *new_q2_obj = NULL;
    PyObject *new_q3_obj = NULL;
    PyObject *shift_support = NULL;
    Py_ssize_t k;
    Py_ssize_t m;
    Py_ssize_t idx;
    long mod_q1;
    long mod_q2;
    long mod_q3;
    long *new_q1 = NULL;
    unsigned char *shift_bits = NULL;
    PyObject *q1_seq = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "OOOOOnlll", &q1_obj, &q2, &q3, &shift_mask, &row_masks_obj, &k, &mod_q1, &mod_q2, &mod_q3)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }

    q1_seq = PySequence_Fast(q1_obj, "q1 must be a sequence.");
    row_masks = PySequence_Fast(row_masks_obj, "row_masks must be a sequence.");
    if (q1_seq == NULL || row_masks == NULL) {
        goto error;
    }
    m = PySequence_Fast_GET_SIZE(q1_seq);
    if (PySequence_Fast_GET_SIZE(row_masks) != m) {
        PyErr_SetString(PyExc_ValueError, "row_masks length must match q1 length.");
        goto error;
    }

    packed_q2 = PyDict_New();
    packed_q3 = PyDict_New();
    if (packed_q2 == NULL || packed_q3 == NULL) {
        goto error;
    }

    new_q1 = PyMem_Calloc((size_t) (k > 0 ? k : 1), sizeof(long));
    shift_bits = PyMem_Calloc((size_t) m, sizeof(unsigned char));
    if (new_q1 == NULL || shift_bits == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    shift_support = get_support_from_cache(NULL, shift_mask);
    if (shift_support == NULL) {
        goto error;
    }
    for (idx = 0; idx < PyTuple_GET_SIZE(shift_support); ++idx) {
        Py_ssize_t pos = PyLong_AsSsize_t(PyTuple_GET_ITEM(shift_support, idx));
        if (pos == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (pos >= 0 && pos < m) {
            shift_bits[pos] = 1;
        }
    }

    for (idx = 0; idx < m; ++idx) {
        long coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, idx));
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (coeff != 0) {
            if (
                add_affine_bit_accumulate(
                    coeff,
                    shift_bits[idx],
                    PySequence_Fast_GET_ITEM(row_masks, idx),
                    NULL,
                    new_q1,
                    k,
                    packed_q2,
                    packed_q3,
                    mod_q1,
                    mod_q2,
                    mod_q3
                ) < 0
            ) {
                goto error;
            }
        }
    }

    {
        Py_ssize_t pos = 0;
        PyObject *key;
        PyObject *value;
        while (PyDict_Next(q2, &pos, &key, &value)) {
            Py_ssize_t a;
            Py_ssize_t b;
            long coeff;
            PyObject *xor_mask = NULL;

            if (parse_pair_key(key, &a, &b) < 0) {
                goto error;
            }
            coeff = PyLong_AsLong(value);
            if (coeff == -1 && PyErr_Occurred()) {
                goto error;
            }
            if (coeff == 0) {
                continue;
            }
            if (a < 0 || a >= m || b < 0 || b >= m) {
                PyErr_SetString(PyExc_IndexError, "q2 term index out of range.");
                goto error;
            }
            if (
                add_affine_bit_accumulate(coeff, shift_bits[a], PySequence_Fast_GET_ITEM(row_masks, a), NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0 ||
                add_affine_bit_accumulate(coeff, shift_bits[b], PySequence_Fast_GET_ITEM(row_masks, b), NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0
            ) {
                goto error;
            }
            xor_mask = PyNumber_Xor(PySequence_Fast_GET_ITEM(row_masks, a), PySequence_Fast_GET_ITEM(row_masks, b));
            if (xor_mask == NULL) {
                goto error;
            }
            if (add_affine_bit_accumulate(positive_mod(-coeff, (int) mod_q1), shift_bits[a] ^ shift_bits[b], xor_mask, NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0) {
                Py_DECREF(xor_mask);
                goto error;
            }
            Py_DECREF(xor_mask);
        }
    }

    {
        Py_ssize_t pos = 0;
        PyObject *key;
        PyObject *value;
        while (PyDict_Next(q3, &pos, &key, &value)) {
            Py_ssize_t a;
            Py_ssize_t b;
            Py_ssize_t c;
            long coeff;
            PyObject *ab_mask = NULL;
            PyObject *ac_mask = NULL;
            PyObject *bc_mask = NULL;
            PyObject *abc_mask = NULL;

            if (parse_triple_key(key, &a, &b, &c) < 0) {
                goto error;
            }
            coeff = PyLong_AsLong(value);
            if (coeff == -1 && PyErr_Occurred()) {
                goto error;
            }
            if (coeff == 0) {
                continue;
            }
            if (a < 0 || a >= m || b < 0 || b >= m || c < 0 || c >= m) {
                PyErr_SetString(PyExc_IndexError, "q3 term index out of range.");
                goto error;
            }
            if (
                add_affine_bit_accumulate(coeff, shift_bits[a], PySequence_Fast_GET_ITEM(row_masks, a), NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0 ||
                add_affine_bit_accumulate(coeff, shift_bits[b], PySequence_Fast_GET_ITEM(row_masks, b), NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0 ||
                add_affine_bit_accumulate(coeff, shift_bits[c], PySequence_Fast_GET_ITEM(row_masks, c), NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0
            ) {
                goto error;
            }

            ab_mask = PyNumber_Xor(PySequence_Fast_GET_ITEM(row_masks, a), PySequence_Fast_GET_ITEM(row_masks, b));
            ac_mask = PyNumber_Xor(PySequence_Fast_GET_ITEM(row_masks, a), PySequence_Fast_GET_ITEM(row_masks, c));
            bc_mask = PyNumber_Xor(PySequence_Fast_GET_ITEM(row_masks, b), PySequence_Fast_GET_ITEM(row_masks, c));
            if (ab_mask == NULL || ac_mask == NULL || bc_mask == NULL) {
                Py_XDECREF(ab_mask);
                Py_XDECREF(ac_mask);
                Py_XDECREF(bc_mask);
                goto error;
            }

            if (
                add_affine_bit_accumulate(positive_mod(-coeff, (int) mod_q1), shift_bits[a] ^ shift_bits[b], ab_mask, NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0 ||
                add_affine_bit_accumulate(positive_mod(-coeff, (int) mod_q1), shift_bits[a] ^ shift_bits[c], ac_mask, NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0 ||
                add_affine_bit_accumulate(positive_mod(-coeff, (int) mod_q1), shift_bits[b] ^ shift_bits[c], bc_mask, NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0
            ) {
                Py_DECREF(ab_mask);
                Py_DECREF(ac_mask);
                Py_DECREF(bc_mask);
                goto error;
            }

            abc_mask = PyNumber_Xor(ab_mask, PySequence_Fast_GET_ITEM(row_masks, c));
            Py_DECREF(ab_mask);
            Py_DECREF(ac_mask);
            Py_DECREF(bc_mask);
            if (abc_mask == NULL) {
                goto error;
            }
            if (add_affine_bit_accumulate(coeff, shift_bits[a] ^ shift_bits[b] ^ shift_bits[c], abc_mask, NULL, new_q1, k, packed_q2, packed_q3, mod_q1, mod_q2, mod_q3) < 0) {
                Py_DECREF(abc_mask);
                goto error;
            }
            Py_DECREF(abc_mask);
        }
    }

    new_q1_obj = PyList_New(k);
    if (new_q1_obj == NULL) {
        goto error;
    }
    for (idx = 0; idx < k; ++idx) {
        PyObject *value_obj = PyLong_FromLong(new_q1[idx]);
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

    PyMem_Free(new_q1);
    PyMem_Free(shift_bits);
    Py_DECREF(q1_seq);
    Py_DECREF(row_masks);
    Py_DECREF(packed_q2);
    Py_DECREF(packed_q3);
    Py_XDECREF(shift_support);

    return Py_BuildValue("NNN", new_q1_obj, new_q2_obj, new_q3_obj);

error:
    PyMem_Free(new_q1);
    PyMem_Free(shift_bits);
    Py_XDECREF(q1_seq);
    Py_XDECREF(row_masks);
    Py_XDECREF(packed_q2);
    Py_XDECREF(packed_q3);
    Py_XDECREF(new_q1_obj);
    Py_XDECREF(new_q2_obj);
    Py_XDECREF(new_q3_obj);
    Py_XDECREF(shift_support);
    return NULL;
}


