static int update_packed_mod_dict(PyObject *values, uint64_t packed_key, long delta, long modulus)
{
    PyObject *key_obj = NULL;
    PyObject *current = NULL;
    PyObject *value_obj = NULL;
    long next_value;
    int rc = -1;

    key_obj = PyLong_FromUnsignedLongLong(packed_key);
    if (key_obj == NULL) {
        return -1;
    }

    current = PyDict_GetItemWithError(values, key_obj);
    if (current == NULL && PyErr_Occurred()) {
        goto done;
    }

    if (current != NULL) {
        next_value = PyLong_AsLong(current);
        if (next_value == -1 && PyErr_Occurred()) {
            goto done;
        }
        next_value = (next_value + delta) % modulus;
    } else {
        next_value = delta % modulus;
    }
    if (next_value < 0) {
        next_value += modulus;
    }

    if (next_value == 0) {
        rc = PyDict_DelItem(values, key_obj);
        if (rc < 0 && PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            rc = 0;
        }
        goto done;
    }

    value_obj = PyLong_FromLong(next_value);
    if (value_obj == NULL) {
        goto done;
    }
    rc = PyDict_SetItem(values, key_obj, value_obj);

done:
    Py_XDECREF(key_obj);
    Py_XDECREF(value_obj);
    return rc;
}


static int add_affine_bit_accumulate(
    long alpha,
    int offset,
    PyObject *mask,
    PyObject *support_cache,
    long *new_q1,
    Py_ssize_t q1_len,
    PyObject *new_q2,
    PyObject *new_q3,
    long mod_q1,
    long mod_q2,
    long mod_q3
)
{
    PyObject *support = NULL;
    Py_ssize_t support_len;
    long alpha_mod;
    long linear;
    long pair;
    long cubic;
    Py_ssize_t idx;
    int truthy;

    if (mask == NULL) {
        return 0;
    }
    truthy = PyObject_IsTrue(mask);
    if (truthy < 0) {
        return -1;
    }
    if (!truthy) {
        return 0;
    }

    alpha_mod = positive_mod(alpha, (int) mod_q1);
    if (alpha_mod == 0) {
        return 0;
    }

    support = get_support_from_cache(support_cache, mask);
    if (support == NULL) {
        return -1;
    }
    support_len = PyTuple_GET_SIZE(support);

    linear = offset ? positive_mod(-alpha_mod, (int) mod_q1) : alpha_mod;
    pair = mod_q2 > 1 ? (offset ? positive_mod(alpha_mod, (int) mod_q2) : positive_mod(-alpha_mod, (int) mod_q2)) : 0;
    cubic = mod_q3 > 1 ? (offset ? positive_mod(-alpha_mod, (int) mod_q3) : positive_mod(alpha_mod, (int) mod_q3)) : 0;

    if (linear != 0) {
        for (idx = 0; idx < support_len; ++idx) {
            Py_ssize_t pos = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, idx));
            if (pos == -1 && PyErr_Occurred()) {
                Py_DECREF(support);
                return -1;
            }
            if (pos < 0 || pos >= q1_len) {
                Py_DECREF(support);
                PyErr_SetString(PyExc_IndexError, "Affine support index out of range.");
                return -1;
            }
            new_q1[pos] = positive_mod(new_q1[pos] + linear, (int) mod_q1);
        }
    }

    if (pair != 0) {
        if (support_len == 2) {
            Py_ssize_t a = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 0));
            Py_ssize_t b = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 1));
            if ((a == -1 || b == -1) && PyErr_Occurred()) {
                Py_DECREF(support);
                return -1;
            }
            if (update_packed_mod_dict(new_q2, (((uint64_t) a) << 32) | (uint32_t) b, pair, mod_q2) < 0) {
                Py_DECREF(support);
                return -1;
            }
        } else if (support_len == 3) {
            Py_ssize_t a = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 0));
            Py_ssize_t b = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 1));
            Py_ssize_t c = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 2));
            if ((a == -1 || b == -1 || c == -1) && PyErr_Occurred()) {
                Py_DECREF(support);
                return -1;
            }
            if (
                update_packed_mod_dict(new_q2, (((uint64_t) a) << 32) | (uint32_t) b, pair, mod_q2) < 0 ||
                update_packed_mod_dict(new_q2, (((uint64_t) a) << 32) | (uint32_t) c, pair, mod_q2) < 0 ||
                update_packed_mod_dict(new_q2, (((uint64_t) b) << 32) | (uint32_t) c, pair, mod_q2) < 0
            ) {
                Py_DECREF(support);
                return -1;
            }
        } else {
            Py_ssize_t left_idx;
            for (left_idx = 0; left_idx < support_len; ++left_idx) {
                Py_ssize_t a = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, left_idx));
                Py_ssize_t right_idx;
                if (a == -1 && PyErr_Occurred()) {
                    Py_DECREF(support);
                    return -1;
                }
                for (right_idx = left_idx + 1; right_idx < support_len; ++right_idx) {
                    Py_ssize_t b = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, right_idx));
                    if (b == -1 && PyErr_Occurred()) {
                        Py_DECREF(support);
                        return -1;
                    }
                    if (update_packed_mod_dict(new_q2, (((uint64_t) a) << 32) | (uint32_t) b, pair, mod_q2) < 0) {
                        Py_DECREF(support);
                        return -1;
                    }
                }
            }
        }
    }

    if (cubic != 0 && support_len >= 3) {
        if (support_len == 3) {
            Py_ssize_t a = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 0));
            Py_ssize_t b = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 1));
            Py_ssize_t c = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, 2));
            if ((a == -1 || b == -1 || c == -1) && PyErr_Occurred()) {
                Py_DECREF(support);
                return -1;
            }
            if (a >= (1 << 21) || b >= (1 << 21) || c >= (1 << 21)) {
                Py_DECREF(support);
                PyErr_SetString(PyExc_OverflowError, "q3 support index exceeds native packing width.");
                return -1;
            }
            if (update_packed_mod_dict(new_q3, (((uint64_t) a) << 42) | (((uint64_t) b) << 21) | (uint64_t) c, cubic, mod_q3) < 0) {
                Py_DECREF(support);
                return -1;
            }
        } else {
            Py_ssize_t i;
            for (i = 0; i < support_len; ++i) {
                Py_ssize_t a = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, i));
                Py_ssize_t j;
                if (a == -1 && PyErr_Occurred()) {
                    Py_DECREF(support);
                    return -1;
                }
                for (j = i + 1; j < support_len; ++j) {
                    Py_ssize_t b = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, j));
                    Py_ssize_t l;
                    if (b == -1 && PyErr_Occurred()) {
                        Py_DECREF(support);
                        return -1;
                    }
                    for (l = j + 1; l < support_len; ++l) {
                        Py_ssize_t c = PyLong_AsSsize_t(PyTuple_GET_ITEM(support, l));
                        if (c == -1 && PyErr_Occurred()) {
                            Py_DECREF(support);
                            return -1;
                        }
                        if (a >= (1 << 21) || b >= (1 << 21) || c >= (1 << 21)) {
                            Py_DECREF(support);
                            PyErr_SetString(PyExc_OverflowError, "q3 support index exceeds native packing width.");
                            return -1;
                        }
                        if (update_packed_mod_dict(new_q3, (((uint64_t) a) << 42) | (((uint64_t) b) << 21) | (uint64_t) c, cubic, mod_q3) < 0) {
                            Py_DECREF(support);
                            return -1;
                        }
                    }
                }
            }
        }
    }

    Py_DECREF(support);
    return 0;
}


static PyObject *expand_packed_pairs(PyObject *packed)
{
    PyObject *expanded = PyDict_New();
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;

    if (expanded == NULL) {
        return NULL;
    }

    while (PyDict_Next(packed, &pos, &key, &value)) {
        uint64_t packed_key = PyLong_AsUnsignedLongLong(key);
        PyObject *tuple_key = NULL;
        PyObject *left = NULL;
        PyObject *right = NULL;
        if (packed_key == (uint64_t) -1 && PyErr_Occurred()) {
            goto error;
        }
        left = PyLong_FromUnsignedLongLong(packed_key >> 32);
        right = PyLong_FromUnsignedLongLong(packed_key & 0xffffffffULL);
        if (left == NULL || right == NULL) {
            Py_XDECREF(left);
            Py_XDECREF(right);
            goto error;
        }
        tuple_key = PyTuple_New(2);
        if (tuple_key == NULL) {
            Py_DECREF(left);
            Py_DECREF(right);
            goto error;
        }
        PyTuple_SET_ITEM(tuple_key, 0, left);
        PyTuple_SET_ITEM(tuple_key, 1, right);
        if (PyDict_SetItem(expanded, tuple_key, value) < 0) {
            Py_DECREF(tuple_key);
            goto error;
        }
        Py_DECREF(tuple_key);
    }

    return expanded;

error:
    Py_DECREF(expanded);
    return NULL;
}


static PyObject *expand_packed_triples(PyObject *packed)
{
    PyObject *expanded = PyDict_New();
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;

    if (expanded == NULL) {
        return NULL;
    }

    while (PyDict_Next(packed, &pos, &key, &value)) {
        uint64_t packed_key = PyLong_AsUnsignedLongLong(key);
        PyObject *tuple_key = NULL;
        PyObject *a = NULL;
        PyObject *b = NULL;
        PyObject *c = NULL;
        if (packed_key == (uint64_t) -1 && PyErr_Occurred()) {
            goto error;
        }
        a = PyLong_FromUnsignedLongLong(packed_key >> 42);
        b = PyLong_FromUnsignedLongLong((packed_key >> 21) & 0x1fffffULL);
        c = PyLong_FromUnsignedLongLong(packed_key & 0x1fffffULL);
        if (a == NULL || b == NULL || c == NULL) {
            Py_XDECREF(a);
            Py_XDECREF(b);
            Py_XDECREF(c);
            goto error;
        }
        tuple_key = PyTuple_New(3);
        if (tuple_key == NULL) {
            Py_DECREF(a);
            Py_DECREF(b);
            Py_DECREF(c);
            goto error;
        }
        PyTuple_SET_ITEM(tuple_key, 0, a);
        PyTuple_SET_ITEM(tuple_key, 1, b);
        PyTuple_SET_ITEM(tuple_key, 2, c);
        if (PyDict_SetItem(expanded, tuple_key, value) < 0) {
            Py_DECREF(tuple_key);
            goto error;
        }
        Py_DECREF(tuple_key);
    }

    return expanded;

error:
    Py_DECREF(expanded);
    return NULL;
}


