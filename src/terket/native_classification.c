#include "_schur_native_internal.h"
#include "native_binary_loaders.inc"

PyObject *build_classification_data_native(PyObject *self, PyObject *args)
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


PyObject *classification_structure_key_native(PyObject *self, PyObject *args)
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


PyObject *build_classification_lookup_native(PyObject *self, PyObject *args)
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


