static PyObject *evaluate_q_mask_terms_native(PyObject *self, PyObject *args)
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


static PyObject *elim_single_partner_constraint_terms_native(PyObject *self, PyObject *args)
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


static PyObject *aff_compose_terms_native(PyObject *self, PyObject *args)
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


static PyObject *build_classification_data_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *q2;
    PyObject *q3;
    PyObject *cubic = NULL;
    PyObject *odd = NULL;
    PyObject *parity = NULL;
    Py_ssize_t idx;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOO", &nvars, &q2, &q3)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }

    cubic = PyList_New(nvars);
    odd = PyList_New(nvars);
    parity = PyList_New(nvars);
    if (cubic == NULL || odd == NULL || parity == NULL) {
        goto error;
    }

    for (idx = 0; idx < nvars; ++idx) {
        PyObject *partners = PyList_New(0);
        if (partners == NULL) {
            goto error;
        }
        Py_INCREF(Py_False);
        PyList_SET_ITEM(cubic, idx, Py_False);
        Py_INCREF(Py_False);
        PyList_SET_ITEM(odd, idx, Py_False);
        PyList_SET_ITEM(parity, idx, partners);
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long coeff;
        long residue;

        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = coeff % 4;
        if (residue < 0) {
            residue += 4;
        }

        if (residue & 1L) {
            if (mark_true(odd, left) < 0 || mark_true(odd, right) < 0) {
                goto error;
            }
        }
        if (residue == 2L) {
            PyObject *right_obj = PyLong_FromSsize_t(right);
            PyObject *left_obj = PyLong_FromSsize_t(left);
            int append_failed = 0;
            if (right_obj == NULL || left_obj == NULL) {
                Py_XDECREF(right_obj);
                Py_XDECREF(left_obj);
                goto error;
            }
            append_failed |= PyList_Append(PyList_GET_ITEM(parity, left), right_obj) < 0;
            append_failed |= PyList_Append(PyList_GET_ITEM(parity, right), left_obj) < 0;
            Py_DECREF(right_obj);
            Py_DECREF(left_obj);
            if (append_failed) {
                goto error;
            }
        }
    }

    pos = 0;
    while (PyDict_Next(q3, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        long coeff;
        long residue;

        if (parse_triple_key(key, &a, &b, &c) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = coeff % 2;
        if (residue < 0) {
            residue += 2;
        }
        if (residue != 0) {
            if (mark_true(cubic, a) < 0 || mark_true(cubic, b) < 0 || mark_true(cubic, c) < 0) {
                goto error;
            }
        }
    }

    for (idx = 0; idx < nvars; ++idx) {
        if (PyList_Sort(PyList_GET_ITEM(parity, idx)) < 0) {
            goto error;
        }
    }

    return Py_BuildValue("NNN", cubic, odd, parity);

error:
    Py_XDECREF(cubic);
    Py_XDECREF(odd);
    Py_XDECREF(parity);
    return NULL;
}


static PyObject *classification_structure_key_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    Py_ssize_t level;
    PyObject *q2;
    PyObject *q3;
    int mod_q2;
    int mod_q3;
    int parity_residue;
    Py_ssize_t q2_capacity;
    Py_ssize_t q3_capacity;
    ClassificationPairRecord *pairs = NULL;
    ClassificationTripleRecord *triples = NULL;
    Py_ssize_t pair_count = 0;
    Py_ssize_t triple_count = 0;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;
    Py_ssize_t payload_size;
    PyObject *payload = NULL;
    unsigned char *buffer;
    unsigned char *cursor;

    (void) self;

    if (!PyArg_ParseTuple(args, "nnOO", &nvars, &level, &q2, &q3)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }
    if (level < 0 || level > 20) {
        PyErr_SetString(PyExc_ValueError, "level out of supported range.");
        return NULL;
    }

    mod_q2 = level > 1 ? (1 << (int) (level - 1)) : 1;
    mod_q3 = level > 2 ? (1 << (int) (level - 2)) : 1;
    parity_residue = mod_q2 > 1 ? (mod_q2 / 2) : 0;
    q2_capacity = PyDict_Size(q2);
    q3_capacity = PyDict_Size(q3);

    if (q2_capacity > 0) {
        pairs = PyMem_Calloc((size_t) q2_capacity, sizeof(ClassificationPairRecord));
        if (pairs == NULL) {
            PyErr_NoMemory();
            goto error;
        }
    }
    if (q3_capacity > 0) {
        triples = PyMem_Calloc((size_t) q3_capacity, sizeof(ClassificationTripleRecord));
        if (triples == NULL) {
            PyErr_NoMemory();
            goto error;
        }
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long coeff;
        int residue;
        unsigned char flags = 0U;

        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = positive_mod(coeff, mod_q2);
        if ((residue & 1) != 0) {
            flags |= 1U;
        }
        if (parity_residue != 0 && residue == parity_residue) {
            flags |= 2U;
        }
        if (flags == 0U) {
            continue;
        }
        pairs[pair_count].left = left;
        pairs[pair_count].right = right;
        pairs[pair_count].flags = flags;
        ++pair_count;
    }

    pos = 0;
    while (PyDict_Next(q3, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        long coeff;
        int residue;

        if (parse_triple_key(key, &a, &b, &c) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = positive_mod(coeff, mod_q3);
        if (residue == 0) {
            continue;
        }
        triples[triple_count].a = a;
        triples[triple_count].b = b;
        triples[triple_count].c = c;
        ++triple_count;
    }

    if (pair_count > 1) {
        qsort(pairs, (size_t) pair_count, sizeof(ClassificationPairRecord), compare_classification_pair_records);
    }
    if (triple_count > 1) {
        qsort(triples, (size_t) triple_count, sizeof(ClassificationTripleRecord), compare_classification_triple_records);
    }

    payload_size = 40 + (17 * pair_count) + (24 * triple_count);
    payload = PyBytes_FromStringAndSize(NULL, payload_size);
    if (payload == NULL) {
        goto error;
    }
    buffer = (unsigned char *) PyBytes_AS_STRING(payload);
    cursor = buffer;

    write_u64_le(cursor, 1ULL);
    cursor += 8;
    write_u64_le(cursor, (uint64_t) nvars);
    cursor += 8;
    write_u64_le(cursor, (uint64_t) level);
    cursor += 8;
    write_u64_le(cursor, (uint64_t) pair_count);
    cursor += 8;
    write_u64_le(cursor, (uint64_t) triple_count);
    cursor += 8;

    for (pos = 0; pos < pair_count; ++pos) {
        write_u64_le(cursor, (uint64_t) pairs[pos].left);
        cursor += 8;
        write_u64_le(cursor, (uint64_t) pairs[pos].right);
        cursor += 8;
        *cursor++ = pairs[pos].flags;
    }
    for (pos = 0; pos < triple_count; ++pos) {
        write_u64_le(cursor, (uint64_t) triples[pos].a);
        cursor += 8;
        write_u64_le(cursor, (uint64_t) triples[pos].b);
        cursor += 8;
        write_u64_le(cursor, (uint64_t) triples[pos].c);
        cursor += 8;
    }

    PyMem_Free(pairs);
    PyMem_Free(triples);
    return payload;

error:
    PyMem_Free(pairs);
    PyMem_Free(triples);
    Py_XDECREF(payload);
    return NULL;
}


static PyObject *build_classification_lookup_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    Py_ssize_t level;
    PyObject *cubic_obj;
    PyObject *odd_obj;
    PyObject *parity_obj;
    unsigned char *cubic = NULL;
    unsigned char *odd = NULL;
    PyObject *parity_seq = NULL;
    PyObject *lookup = NULL;
    PyObject *cubic_entry = NULL;
    PyObject *decoupled_entry = NULL;
    PyObject *zero_entry = NULL;
    Py_ssize_t mod_q1;
    Py_ssize_t threshold;
    Py_ssize_t var;

    (void) self;

    if (!PyArg_ParseTuple(args, "nnOOO", &nvars, &level, &cubic_obj, &odd_obj, &parity_obj)) {
        return NULL;
    }
    if (level < 0 || level > 20) {
        PyErr_SetString(PyExc_ValueError, "level out of supported range.");
        return NULL;
    }

    mod_q1 = (Py_ssize_t) 1 << (int) level;
    threshold = mod_q1 >= 4 ? (mod_q1 / 4) : 1;
    if (load_binary_flags(cubic_obj, "cubic incidence must be a sequence.", &cubic, &var) < 0) {
        goto error;
    }
    if (var != nvars) {
        PyErr_SetString(PyExc_ValueError, "cubic incidence length mismatch.");
        goto error;
    }
    if (load_binary_flags(odd_obj, "odd bilinear must be a sequence.", &odd, &var) < 0) {
        goto error;
    }
    if (var != nvars) {
        PyErr_SetString(PyExc_ValueError, "odd bilinear length mismatch.");
        goto error;
    }

    parity_seq = PySequence_Fast(parity_obj, "parity partners must be a sequence.");
    if (parity_seq == NULL) {
        goto error;
    }
    if (PySequence_Fast_GET_SIZE(parity_seq) != nvars) {
        PyErr_SetString(PyExc_ValueError, "parity partners length mismatch.");
        goto error;
    }

    lookup = PyTuple_New(nvars);
    if (lookup == NULL) {
        goto error;
    }

    cubic_entry = Py_BuildValue("(n)", 0);
    decoupled_entry = Py_BuildValue("(n)", 2);
    zero_entry = Py_BuildValue("(n)", 3);
    if (cubic_entry == NULL || decoupled_entry == NULL || zero_entry == NULL) {
        goto error;
    }

    for (var = 0; var < nvars; ++var) {
        PyObject *partners_obj = PySequence_Fast_GET_ITEM(parity_seq, var);
        PyObject *partners_tuple = NULL;
        PyObject *var_entries = NULL;
        Py_ssize_t coeff;
        Py_ssize_t partner_count;

        if (PyTuple_CheckExact(partners_obj)) {
            partners_tuple = partners_obj;
            Py_INCREF(partners_tuple);
        } else {
            partners_tuple = PySequence_Tuple(partners_obj);
        }
        if (partners_tuple == NULL) {
            goto error;
        }
        partner_count = PyTuple_GET_SIZE(partners_tuple);
        var_entries = PyTuple_New(mod_q1);
        if (var_entries == NULL) {
            Py_DECREF(partners_tuple);
            goto error;
        }

        for (coeff = 0; coeff < mod_q1; ++coeff) {
            PyObject *entry = NULL;

            if ((coeff % threshold) != 0 || cubic[var] != 0U) {
                Py_INCREF(cubic_entry);
                entry = cubic_entry;
            } else {
                Py_ssize_t reduced = (coeff / threshold) % 4;
                if (reduced == 1 || reduced == 3) {
                    entry = Py_BuildValue("(nnO)", 1, coeff, odd[var] ? Py_True : Py_False);
                } else if (odd[var] != 0U) {
                    Py_INCREF(cubic_entry);
                    entry = cubic_entry;
                } else if (reduced == 0 && partner_count == 0) {
                    Py_INCREF(decoupled_entry);
                    entry = decoupled_entry;
                } else if (reduced == 2 && partner_count == 0) {
                    Py_INCREF(zero_entry);
                    entry = zero_entry;
                } else {
                    entry = Py_BuildValue("(nOn)", 4, partners_tuple, coeff);
                }
            }
            if (entry == NULL) {
                Py_DECREF(partners_tuple);
                Py_DECREF(var_entries);
                goto error;
            }
            PyTuple_SET_ITEM(var_entries, coeff, entry);
        }

        Py_DECREF(partners_tuple);
        PyTuple_SET_ITEM(lookup, var, var_entries);
    }

    PyMem_Free(cubic);
    PyMem_Free(odd);
    Py_DECREF(parity_seq);
    Py_DECREF(cubic_entry);
    Py_DECREF(decoupled_entry);
    Py_DECREF(zero_entry);
    return lookup;

error:
    PyMem_Free(cubic);
    PyMem_Free(odd);
    Py_XDECREF(parity_seq);
    Py_XDECREF(lookup);
    Py_XDECREF(cubic_entry);
    Py_XDECREF(decoupled_entry);
    Py_XDECREF(zero_entry);
    return NULL;
}


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


static PyObject *solve_output_shift_mask_u64_native(PyObject *self, PyObject *args)
{
    PyObject *eps0_obj;
    PyObject *pivot_col_obj;
    PyObject *row_ops_obj;
    PyObject *output_bits_obj;
    Py_ssize_t mvars;
    unsigned char *eps0 = NULL;
    Py_ssize_t *pivot_col = NULL;
    uint64_t *row_ops = NULL;
    unsigned char *output_bits = NULL;
    Py_ssize_t nbits;
    Py_ssize_t output_len;
    PyObject *result = NULL;
    uint64_t target_mask = 0ULL;
    uint64_t shift_mask = 0ULL;
    Py_ssize_t row_idx;

    (void) self;

    if (!PyArg_ParseTuple(args, "OOOOn", &eps0_obj, &pivot_col_obj, &row_ops_obj, &output_bits_obj, &mvars)) {
        return NULL;
    }

    if (load_binary_flags(eps0_obj, "eps0 must be a sequence.", &eps0, &nbits) < 0) {
        goto error;
    }
    if (nbits > 64 || mvars > 64) {
        PyErr_SetString(PyExc_ValueError, "Native u64 output solving requires n <= 64 and m <= 64.");
        goto error;
    }
    if (load_ssize_array(pivot_col_obj, "pivot_col must be a sequence.", &pivot_col, nbits) < 0) {
        goto error;
    }
    if (load_u64_array(row_ops_obj, "row_ops must be a sequence.", &row_ops, nbits) < 0) {
        goto error;
    }
    if (load_binary_flags(output_bits_obj, "output_bits must be a sequence.", &output_bits, &output_len) < 0) {
        goto error;
    }
    if (output_len != nbits) {
        PyErr_SetString(PyExc_ValueError, "output_bits length mismatch.");
        goto error;
    }

    for (row_idx = 0; row_idx < nbits; ++row_idx) {
        if (eps0[row_idx] != output_bits[row_idx]) {
            target_mask |= 1ULL << (size_t) row_idx;
        }
    }

    for (row_idx = 0; row_idx < nbits; ++row_idx) {
        if ((popcount_u64(target_mask & row_ops[row_idx]) & 1) == 0) {
            continue;
        }
        if (pivot_col[row_idx] < 0) {
            Py_INCREF(Py_None);
            result = Py_None;
            goto done;
        }
        shift_mask |= 1ULL << (size_t) pivot_col[row_idx];
    }

    result = PyLong_FromUnsignedLongLong((unsigned long long) shift_mask);

done:
    PyMem_Free(eps0);
    PyMem_Free(pivot_col);
    PyMem_Free(row_ops);
    PyMem_Free(output_bits);
    return result;

error:
    PyMem_Free(eps0);
    PyMem_Free(pivot_col);
    PyMem_Free(row_ops);
    PyMem_Free(output_bits);
    return NULL;
}


static PyObject *solve_output_shift_masks_u64_native(PyObject *self, PyObject *args)
{
    PyObject *eps0_obj;
    PyObject *pivot_col_obj;
    PyObject *row_ops_obj;
    PyObject *outputs_obj;
    Py_ssize_t mvars;
    unsigned char *eps0 = NULL;
    Py_ssize_t *pivot_col = NULL;
    uint64_t *row_ops = NULL;
    Py_ssize_t nbits;
    PyObject *outputs_seq = NULL;
    PyObject *result = NULL;
    Py_ssize_t output_count;
    Py_ssize_t output_idx;

    (void) self;

    if (!PyArg_ParseTuple(args, "OOOOn", &eps0_obj, &pivot_col_obj, &row_ops_obj, &outputs_obj, &mvars)) {
        return NULL;
    }

    if (load_binary_flags(eps0_obj, "eps0 must be a sequence.", &eps0, &nbits) < 0) {
        goto error;
    }
    if (nbits > 64 || mvars > 64) {
        PyErr_SetString(PyExc_ValueError, "Native u64 output solving requires n <= 64 and m <= 64.");
        goto error;
    }
    if (load_ssize_array(pivot_col_obj, "pivot_col must be a sequence.", &pivot_col, nbits) < 0) {
        goto error;
    }
    if (load_u64_array(row_ops_obj, "row_ops must be a sequence.", &row_ops, nbits) < 0) {
        goto error;
    }

    outputs_seq = PySequence_Fast(outputs_obj, "outputs must be a sequence.");
    if (outputs_seq == NULL) {
        goto error;
    }
    output_count = PySequence_Fast_GET_SIZE(outputs_seq);
    result = PyTuple_New(output_count);
    if (result == NULL) {
        goto error;
    }

    for (output_idx = 0; output_idx < output_count; ++output_idx) {
        PyObject *output_obj = PySequence_Fast_GET_ITEM(outputs_seq, output_idx);
        PyObject *output_seq = NULL;
        Py_ssize_t output_len;
        Py_ssize_t row_idx;
        uint64_t target_mask = 0ULL;
        uint64_t shift_mask = 0ULL;
        int supported = 1;
        PyObject *item = NULL;

        output_seq = PySequence_Fast(output_obj, "Each output must be a sequence.");
        if (output_seq == NULL) {
            goto error;
        }
        output_len = PySequence_Fast_GET_SIZE(output_seq);
        if (output_len != nbits) {
            Py_DECREF(output_seq);
            PyErr_SetString(PyExc_ValueError, "output_bits length mismatch.");
            goto error;
        }

        for (row_idx = 0; row_idx < nbits; ++row_idx) {
            int truthy = PyObject_IsTrue(PySequence_Fast_GET_ITEM(output_seq, row_idx));
            if (truthy < 0) {
                Py_DECREF(output_seq);
                goto error;
            }
            if (eps0[row_idx] != (truthy ? 1U : 0U)) {
                target_mask |= 1ULL << (size_t) row_idx;
            }
        }

        for (row_idx = 0; row_idx < nbits; ++row_idx) {
            if ((popcount_u64(target_mask & row_ops[row_idx]) & 1) == 0) {
                continue;
            }
            if (pivot_col[row_idx] < 0) {
                supported = 0;
                break;
            }
            shift_mask |= 1ULL << (size_t) pivot_col[row_idx];
        }

        if (supported) {
            item = PyLong_FromUnsignedLongLong((unsigned long long) shift_mask);
        } else {
            Py_INCREF(Py_None);
            item = Py_None;
        }
        Py_DECREF(output_seq);
        if (item == NULL) {
            goto error;
        }
        PyTuple_SET_ITEM(result, output_idx, item);
    }

    PyMem_Free(eps0);
    PyMem_Free(pivot_col);
    PyMem_Free(row_ops);
    Py_DECREF(outputs_seq);
    return result;

error:
    PyMem_Free(eps0);
    PyMem_Free(pivot_col);
    PyMem_Free(row_ops);
    Py_XDECREF(outputs_seq);
    Py_XDECREF(result);
    return NULL;
}


