#include "_schur_native_internal.h"

PyObject *evaluate_q_mask_terms_native(PyObject *self, PyObject *args)
{
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *q3;
    PyObject *mask;
    PyObject *q1_seq = NULL;
    PyObject *support = NULL;
    unsigned char *mask_bits = NULL;
    Py_ssize_t m;
    Py_ssize_t idx;
    long residue = 0;

    (void) self;

    if (!PyArg_ParseTuple(args, "OOOO", &q1_obj, &q2, &q3, &mask)) {
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
    m = PySequence_Fast_GET_SIZE(q1_seq);
    mask_bits = PyMem_Calloc((size_t) m, sizeof(unsigned char));
    if (mask_bits == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    support = get_support_from_cache(NULL, mask);
    if (support == NULL) {
        goto error;
    }
    for (idx = 0; idx < PyTuple_GET_SIZE(support); ++idx) {
        Py_ssize_t pos = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, idx));
        if (pos == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (pos >= 0 && pos < m) {
            mask_bits[pos] = 1;
        }
    }

    for (idx = 0; idx < m; ++idx) {
        long coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, idx));
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (mask_bits[idx]) {
            residue = (residue + coeff) & 7L;
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
            if (parse_pair_key(key, &a, &b) < 0) {
                goto error;
            }
            coeff = PyLong_AsLong(value);
            if (coeff == -1 && PyErr_Occurred()) {
                goto error;
            }
            if (a < 0 || a >= m || b < 0 || b >= m) {
                PyErr_SetString(PyExc_IndexError, "q2 term index out of range.");
                goto error;
            }
            if (mask_bits[a] && mask_bits[b]) {
                residue = (residue + ((2L * coeff) & 7L)) & 7L;
            }
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
            if (parse_triple_key(key, &a, &b, &c) < 0) {
                goto error;
            }
            coeff = PyLong_AsLong(value);
            if (coeff == -1 && PyErr_Occurred()) {
                goto error;
            }
            if (a < 0 || a >= m || b < 0 || b >= m || c < 0 || c >= m) {
                PyErr_SetString(PyExc_IndexError, "q3 term index out of range.");
                goto error;
            }
            if (mask_bits[a] && mask_bits[b] && mask_bits[c]) {
                residue = (residue + ((4L * coeff) & 7L)) & 7L;
            }
        }
    }

    PyMem_Free(mask_bits);
    Py_DECREF(q1_seq);
    Py_DECREF(support);
    return PyLong_FromLong(residue);

error:
    PyMem_Free(mask_bits);
    Py_XDECREF(q1_seq);
    Py_XDECREF(support);
    return NULL;
}

