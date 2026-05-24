static int load_binary_flags(PyObject *seq_obj, const char *message, unsigned char **out_bits, Py_ssize_t *out_len)
{
    PyObject *seq = NULL;
    unsigned char *bits = NULL;
    Py_ssize_t idx;
    Py_ssize_t length;

    seq = PySequence_Fast(seq_obj, message);
    if (seq == NULL) {
        return -1;
    }
    length = PySequence_Fast_GET_SIZE(seq);
    bits = PyMem_Calloc((size_t) (length > 0 ? length : 1), sizeof(unsigned char));
    if (bits == NULL) {
        PyErr_NoMemory();
        Py_DECREF(seq);
        return -1;
    }
    for (idx = 0; idx < length; ++idx) {
        int truthy = PyObject_IsTrue(PySequence_Fast_GET_ITEM(seq, idx));
        if (truthy < 0) {
            PyMem_Free(bits);
            Py_DECREF(seq);
            return -1;
        }
        bits[idx] = truthy ? 1U : 0U;
    }
    Py_DECREF(seq);
    *out_bits = bits;
    *out_len = length;
    return 0;
}


static int load_ssize_array(PyObject *seq_obj, const char *message, Py_ssize_t **out_values, Py_ssize_t expected_len)
{
    PyObject *seq = NULL;
    Py_ssize_t *values = NULL;
    Py_ssize_t idx;
    Py_ssize_t length;

    seq = PySequence_Fast(seq_obj, message);
    if (seq == NULL) {
        return -1;
    }
    length = PySequence_Fast_GET_SIZE(seq);
    if (expected_len >= 0 && length != expected_len) {
        PyErr_SetString(PyExc_ValueError, "Sequence length mismatch.");
        Py_DECREF(seq);
        return -1;
    }
    values = PyMem_Calloc((size_t) (length > 0 ? length : 1), sizeof(Py_ssize_t));
    if (values == NULL) {
        PyErr_NoMemory();
        Py_DECREF(seq);
        return -1;
    }
    for (idx = 0; idx < length; ++idx) {
        values[idx] = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(seq, idx));
        if (values[idx] == -1 && PyErr_Occurred()) {
            PyMem_Free(values);
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    *out_values = values;
    return 0;
}


static int load_u64_array(PyObject *seq_obj, const char *message, uint64_t **out_values, Py_ssize_t expected_len)
{
    PyObject *seq = NULL;
    uint64_t *values = NULL;
    Py_ssize_t idx;
    Py_ssize_t length;

    seq = PySequence_Fast(seq_obj, message);
    if (seq == NULL) {
        return -1;
    }
    length = PySequence_Fast_GET_SIZE(seq);
    if (expected_len >= 0 && length != expected_len) {
        PyErr_SetString(PyExc_ValueError, "Sequence length mismatch.");
        Py_DECREF(seq);
        return -1;
    }
    values = PyMem_Calloc((size_t) (length > 0 ? length : 1), sizeof(uint64_t));
    if (values == NULL) {
        PyErr_NoMemory();
        Py_DECREF(seq);
        return -1;
    }
    for (idx = 0; idx < length; ++idx) {
        unsigned long long value = PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(seq, idx));
        if (PyErr_Occurred()) {
            PyMem_Free(values);
            Py_DECREF(seq);
            return -1;
        }
        values[idx] = (uint64_t) value;
    }
    Py_DECREF(seq);
    *out_values = values;
    return 0;
}


