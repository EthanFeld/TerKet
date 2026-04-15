static PyObject *sum_treewidth_dp_level3_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *q3;
    PyObject *order_obj;
    PyObject *q1_seq = NULL;
    PyObject *order_seq = NULL;
    NativeFactor *factors = NULL;
    Py_ssize_t factor_capacity;
    Py_ssize_t factor_count = 0;
    Py_ssize_t *bucket_indexes = NULL;
    Py_ssize_t *union_scope = NULL;
    char *union_marks = NULL;
    NativeComplex scalar = {1.0, 0.0};
    Py_ssize_t max_scope = 0;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;
    Py_ssize_t order_len;
    Py_ssize_t order_idx;
    PyObject *result = NULL;
    PyObject *total_obj = NULL;
    PyObject *width_obj = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOOOO", &nvars, &q1_obj, &q2, &q3, &order_obj)) {
        return NULL;
    }
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return NULL;
    }

    q1_seq = PySequence_Fast(q1_obj, "q1 must be a sequence.");
    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (q1_seq == NULL || order_seq == NULL) {
        goto error;
    }
    if (PySequence_Fast_GET_SIZE(q1_seq) != nvars) {
        PyErr_SetString(PyExc_ValueError, "q1 must have length nvars.");
        goto error;
    }

    factor_capacity = PyDict_Size(q2) + PyDict_Size(q3) + (2 * nvars) + 1;
    if (factor_capacity < 1) {
        factor_capacity = 1;
    }
    factors = PyMem_Calloc((size_t) factor_capacity, sizeof(NativeFactor));
    bucket_indexes = PyMem_Malloc((size_t) factor_capacity * sizeof(Py_ssize_t));
    union_scope = PyMem_Malloc((size_t) nvars * sizeof(Py_ssize_t));
    union_marks = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(char));
    if (factors == NULL || bucket_indexes == NULL || union_scope == NULL || union_marks == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    for (pos = 0; pos < nvars; ++pos) {
        long coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, pos));
        int residue;
        NativeFactor *factor;
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = positive_mod(coeff, 8);
        if (residue == 0) {
            continue;
        }
        if (factor_count >= factor_capacity) {
            PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
            goto error;
        }
        factor = &factors[factor_count++];
        factor->arity = 1;
        factor->scope = PyMem_Malloc(sizeof(Py_ssize_t));
        factor->table = PyMem_Malloc(2 * sizeof(NativeComplex));
        if (factor->scope == NULL || factor->table == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = pos;
        factor->table[0] = complex_make(1.0, 0.0);
        factor->table[1] = omega_level3_value(residue);
        factor->alive = 1;
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long coeff;
        int residue;
        int shift;
        NativeFactor *factor;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = positive_mod(coeff, 4);
        if (residue == 0) {
            continue;
        }
        shift = (2 * residue) & 7;
        if (factor_count >= factor_capacity) {
            PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
            goto error;
        }
        factor = &factors[factor_count++];
        factor->arity = 2;
        factor->scope = PyMem_Malloc(2 * sizeof(Py_ssize_t));
        factor->table = PyMem_Malloc(4 * sizeof(NativeComplex));
        if (factor->scope == NULL || factor->table == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = left;
        factor->scope[1] = right;
        factor->table[0] = complex_make(1.0, 0.0);
        factor->table[1] = complex_make(1.0, 0.0);
        factor->table[2] = complex_make(1.0, 0.0);
        factor->table[3] = omega_level3_value(shift);
        factor->alive = 1;
    }

    pos = 0;
    while (PyDict_Next(q3, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        long coeff;
        int residue;
        NativeFactor *factor;
        Py_ssize_t table_idx;
        if (parse_triple_key(key, &a, &b, &c) < 0) {
            goto error;
        }
        coeff = PyLong_AsLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = positive_mod(coeff, 2);
        if (residue == 0) {
            continue;
        }
        if (factor_count >= factor_capacity) {
            PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
            goto error;
        }
        factor = &factors[factor_count++];
        factor->arity = 3;
        factor->scope = PyMem_Malloc(3 * sizeof(Py_ssize_t));
        factor->table = PyMem_Malloc(8 * sizeof(NativeComplex));
        if (factor->scope == NULL || factor->table == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = a;
        factor->scope[1] = b;
        factor->scope[2] = c;
        for (table_idx = 0; table_idx < 8; ++table_idx) {
            factor->table[table_idx] = complex_make(1.0, 0.0);
        }
        factor->table[7] = omega_level3_value(4);
        factor->alive = 1;
    }

    order_len = PySequence_Fast_GET_SIZE(order_seq);
    for (order_idx = 0; order_idx < order_len; ++order_idx) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(order_seq, order_idx));
        Py_ssize_t bucket_count = 0;
        Py_ssize_t union_arity = 0;
        Py_ssize_t var_pos = -1;
        Py_ssize_t factor_idx;

        if (var == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (var < 0 || var >= nvars) {
            PyErr_SetString(PyExc_ValueError, "order entries must lie in [0, nvars).");
            goto error;
        }

        if (nvars > 0) {
            memset(union_marks, 0, (size_t) nvars * sizeof(char));
        }

        for (factor_idx = 0; factor_idx < factor_count; ++factor_idx) {
            Py_ssize_t scope_idx;
            NativeFactor *factor = &factors[factor_idx];
            if (!factor->alive || !factor_contains_var(factor, var)) {
                continue;
            }
            bucket_indexes[bucket_count++] = factor_idx;
            for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                union_marks[factor->scope[scope_idx]] = 1;
            }
        }

        if (bucket_count == 0) {
            scalar.real *= 2.0;
            scalar.imag *= 2.0;
            if (max_scope < 1) {
                max_scope = 1;
            }
            continue;
        }

        for (factor_idx = 0; factor_idx < nvars; ++factor_idx) {
            if (union_marks[factor_idx]) {
                if (factor_idx == var) {
                    var_pos = union_arity;
                }
                union_scope[union_arity++] = factor_idx;
            }
        }
        if (var_pos < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Elimination variable missing from union scope.");
            goto error;
        }
        if (union_arity > max_scope) {
            max_scope = union_arity;
        }

        {
            Py_ssize_t new_arity = union_arity - 1;
            size_t table_size;
            NativeComplex *new_table = NULL;
            Py_ssize_t *new_scope = NULL;
            Py_ssize_t **positions = NULL;
            Py_ssize_t *position_buffer = NULL;
            Py_ssize_t bucket_idx;
            Py_ssize_t insert_idx = -1;

            if (new_arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
                PyErr_SetString(PyExc_OverflowError, "Union scope too wide for native treewidth DP.");
                goto error;
            }
            table_size = ((size_t) 1) << (size_t) new_arity;

            new_table = PyMem_Calloc(table_size > 0 ? table_size : 1, sizeof(NativeComplex));
            positions = PyMem_Malloc((size_t) bucket_count * sizeof(Py_ssize_t *));
            if (new_arity > 0) {
                new_scope = PyMem_Malloc((size_t) new_arity * sizeof(Py_ssize_t));
            }
            if (new_table == NULL || positions == NULL || (new_arity > 0 && new_scope == NULL)) {
                PyMem_Free(new_table);
                PyMem_Free(positions);
                PyMem_Free(new_scope);
                PyErr_NoMemory();
                goto error;
            }

            if (new_arity > 0) {
                Py_ssize_t out_idx = 0;
                for (factor_idx = 0; factor_idx < union_arity; ++factor_idx) {
                    if (factor_idx == var_pos) {
                        continue;
                    }
                    new_scope[out_idx++] = union_scope[factor_idx];
                }
            }

            for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                NativeFactor *factor = &factors[bucket_indexes[bucket_idx]];
                Py_ssize_t scope_idx;
                positions[bucket_idx] = PyMem_Malloc((size_t) factor->arity * sizeof(Py_ssize_t));
                if (positions[bucket_idx] == NULL) {
                    for (scope_idx = 0; scope_idx < bucket_idx; ++scope_idx) {
                        PyMem_Free(positions[scope_idx]);
                    }
                    PyMem_Free(new_table);
                    PyMem_Free(positions);
                    PyMem_Free(new_scope);
                    PyErr_NoMemory();
                    goto error;
                }
                for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                    Py_ssize_t union_idx;
                    positions[bucket_idx][scope_idx] = -1;
                    for (union_idx = 0; union_idx < union_arity; ++union_idx) {
                        if (union_scope[union_idx] == factor->scope[scope_idx]) {
                            positions[bucket_idx][scope_idx] = union_idx;
                            break;
                        }
                    }
                    if (positions[bucket_idx][scope_idx] < 0) {
                        for (union_idx = 0; union_idx <= bucket_idx; ++union_idx) {
                            PyMem_Free(positions[union_idx]);
                        }
                        PyMem_Free(new_table);
                        PyMem_Free(positions);
                        PyMem_Free(new_scope);
                        PyErr_SetString(PyExc_RuntimeError, "Failed to map factor scope into union scope.");
                        goto error;
                    }
                }
            }

            for (factor_idx = 0; factor_idx < (Py_ssize_t) table_size; ++factor_idx) {
                uint64_t reduced_assignment = (uint64_t) factor_idx;
                NativeComplex total = complex_make(0.0, 0.0);
                int fixed_value;
                for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                    uint64_t low_mask = (var_pos == 0) ? 0ULL : ((((uint64_t) 1) << (size_t) var_pos) - 1ULL);
                    uint64_t full_assignment =
                        (reduced_assignment & low_mask) |
                        (((uint64_t) fixed_value) << (size_t) var_pos) |
                        ((reduced_assignment & ~low_mask) << 1);
                    NativeComplex weight = complex_make(1.0, 0.0);
                    for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                        NativeFactor *factor = &factors[bucket_indexes[bucket_idx]];
                        Py_ssize_t table_idx = project_assignment_bits_u64(
                            full_assignment,
                            positions[bucket_idx],
                            factor->arity
                        );
                        weight = complex_mul(weight, factor->table[table_idx]);
                    }
                    total = complex_add(total, weight);
                }
                new_table[factor_idx] = total;
            }

            for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                PyMem_Free(positions[bucket_idx]);
            }
            PyMem_Free(positions);
            positions = NULL;

            for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                free_native_factor(&factors[bucket_indexes[bucket_idx]]);
            }

            if (new_arity == 0) {
                scalar = complex_mul(scalar, new_table[0]);
                PyMem_Free(new_table);
                PyMem_Free(new_scope);
                new_table = NULL;
                new_scope = NULL;
                continue;
            }

            for (factor_idx = 0; factor_idx < factor_count; ++factor_idx) {
                NativeFactor *factor = &factors[factor_idx];
                if (!factor->alive) {
                    if (insert_idx < 0) {
                        insert_idx = factor_idx;
                    }
                    continue;
                }
                if (scopes_equal(factor, new_scope, new_arity)) {
                    for (bucket_idx = 0; bucket_idx < (Py_ssize_t) table_size; ++bucket_idx) {
                        factor->table[bucket_idx] = complex_mul(factor->table[bucket_idx], new_table[bucket_idx]);
                    }
                    PyMem_Free(new_table);
                    PyMem_Free(new_scope);
                    new_table = NULL;
                    new_scope = NULL;
                    insert_idx = -1;
                    break;
                }
            }

            if (new_scope == NULL) {
                continue;
            }

            if (insert_idx < 0) {
                if (factor_count >= factor_capacity) {
                    PyMem_Free(new_table);
                    PyMem_Free(new_scope);
                    PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
                    goto error;
                }
                insert_idx = factor_count++;
            }

            factors[insert_idx].arity = new_arity;
            factors[insert_idx].scope = new_scope;
            factors[insert_idx].table = new_table;
            factors[insert_idx].alive = 1;
        }
    }

    for (pos = 0; pos < factor_count; ++pos) {
        if (factors[pos].alive) {
            PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the DP order.");
            goto error;
        }
    }

    total_obj = PyComplex_FromDoubles(scalar.real, scalar.imag);
    width_obj = PyLong_FromSsize_t(max_scope);
    if (total_obj == NULL || width_obj == NULL) {
        goto error;
    }
    result = PyTuple_Pack(2, total_obj, width_obj);

error:
    if (factors != NULL) {
        for (pos = 0; pos < factor_count; ++pos) {
            free_native_factor(&factors[pos]);
        }
    }
    PyMem_Free(factors);
    PyMem_Free(bucket_indexes);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    Py_XDECREF(q1_seq);
    Py_XDECREF(order_seq);
    Py_XDECREF(total_obj);
    Py_XDECREF(width_obj);
    return result;
}


static PyObject *sum_factor_tables_scaled_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *factors_obj;
    PyObject *order_obj;
    PyObject *scalar_obj;
    PyObject *order_seq = NULL;
    NativeScaledFactor *factors = NULL;
    Py_ssize_t factor_capacity = 0;
    Py_ssize_t factor_count = 0;
    Py_ssize_t *bucket_indexes = NULL;
    Py_ssize_t *union_scope = NULL;
    char *union_marks = NULL;
    NativeScaledComplex scalar;
    Py_ssize_t max_scope = 0;
    PyObject *result = NULL;
    PyObject *total_obj = NULL;
    PyObject *exp_obj = NULL;
    PyObject *width_obj = NULL;
    Py_ssize_t order_len;
    Py_ssize_t order_idx;
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOOO", &nvars, &factors_obj, &order_obj, &scalar_obj)) {
        return NULL;
    }
    if (!PyDict_Check(factors_obj)) {
        PyErr_SetString(PyExc_TypeError, "factors must be a dict keyed by scope tuples.");
        return NULL;
    }
    if (parse_scaled_complex_entry(scalar_obj, &scalar) < 0) {
        return NULL;
    }

    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (order_seq == NULL) {
        goto error;
    }

    factor_capacity = PyDict_Size(factors_obj);
    if (factor_capacity < 1) {
        factor_capacity = 1;
    }
    factors = PyMem_Calloc((size_t) factor_capacity, sizeof(NativeScaledFactor));
    bucket_indexes = PyMem_Malloc((size_t) factor_capacity * sizeof(Py_ssize_t));
    union_scope = PyMem_Malloc((size_t) (nvars > 0 ? nvars : 1) * sizeof(Py_ssize_t));
    union_marks = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(char));
    if (factors == NULL || bucket_indexes == NULL || union_scope == NULL || union_marks == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    while (PyDict_Next(factors_obj, &pos, &key, &value)) {
        PyObject *scope_seq = NULL;
        PyObject *table_seq = NULL;
        Py_ssize_t arity;
        Py_ssize_t table_size;
        Py_ssize_t idx;
        NativeScaledFactor *factor;

        scope_seq = PySequence_Fast(key, "Factor scopes must be tuples/sequences.");
        table_seq = PySequence_Fast(value, "Factor tables must be sequences.");
        if (scope_seq == NULL || table_seq == NULL) {
            Py_XDECREF(scope_seq);
            Py_XDECREF(table_seq);
            goto error;
        }

        arity = PySequence_Fast_GET_SIZE(scope_seq);
        if (arity < 0) {
            Py_DECREF(scope_seq);
            Py_DECREF(table_seq);
            goto error;
        }
        if (arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
            PyErr_SetString(PyExc_OverflowError, "Factor scope too wide for native factor DP.");
            Py_DECREF(scope_seq);
            Py_DECREF(table_seq);
            goto error;
        }
        table_size = ((Py_ssize_t) 1) << (size_t) arity;
        if (PySequence_Fast_GET_SIZE(table_seq) != table_size) {
            PyErr_SetString(PyExc_ValueError, "Factor table length must equal 2**len(scope).");
            Py_DECREF(scope_seq);
            Py_DECREF(table_seq);
            goto error;
        }
        if (factor_count >= factor_capacity) {
            PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
            Py_DECREF(scope_seq);
            Py_DECREF(table_seq);
            goto error;
        }

        factor = &factors[factor_count++];
        factor->arity = arity;
        factor->alive = 1;
        factor->scope = NULL;
        factor->table = NULL;
        if (arity > 0) {
            factor->scope = PyMem_Malloc((size_t) arity * sizeof(Py_ssize_t));
        }
        factor->table = PyMem_Malloc((size_t) table_size * sizeof(NativeScaledComplex));
        if ((arity > 0 && factor->scope == NULL) || factor->table == NULL) {
            Py_DECREF(scope_seq);
            Py_DECREF(table_seq);
            PyErr_NoMemory();
            goto error;
        }

        for (idx = 0; idx < arity; ++idx) {
            Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(scope_seq, idx));
            if (var == -1 && PyErr_Occurred()) {
                Py_DECREF(scope_seq);
                Py_DECREF(table_seq);
                goto error;
            }
            if (var < 0 || var >= nvars) {
                PyErr_SetString(PyExc_ValueError, "Factor scope contains an out-of-range variable.");
                Py_DECREF(scope_seq);
                Py_DECREF(table_seq);
                goto error;
            }
            factor->scope[idx] = var;
        }
        for (idx = 0; idx < table_size; ++idx) {
            if (parse_scaled_complex_entry(PySequence_Fast_GET_ITEM(table_seq, idx), &factor->table[idx]) < 0) {
                Py_DECREF(scope_seq);
                Py_DECREF(table_seq);
                goto error;
            }
        }

        Py_DECREF(scope_seq);
        Py_DECREF(table_seq);
    }

    order_len = PySequence_Fast_GET_SIZE(order_seq);
    for (order_idx = 0; order_idx < order_len; ++order_idx) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(order_seq, order_idx));
        Py_ssize_t bucket_count = 0;
        Py_ssize_t union_arity = 0;
        Py_ssize_t var_pos = -1;
        Py_ssize_t factor_idx;

        if (var == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (var < 0 || var >= nvars) {
            PyErr_SetString(PyExc_ValueError, "order entries must lie in [0, nvars).");
            goto error;
        }

        if (nvars > 0) {
            memset(union_marks, 0, (size_t) nvars * sizeof(char));
        }

        for (factor_idx = 0; factor_idx < factor_count; ++factor_idx) {
            Py_ssize_t scope_idx;
            NativeScaledFactor *factor = &factors[factor_idx];
            if (!factor->alive) {
                continue;
            }
            for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                if (factor->scope[scope_idx] == var) {
                    bucket_indexes[bucket_count++] = factor_idx;
                    for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                        union_marks[factor->scope[scope_idx]] = 1;
                    }
                    break;
                }
            }
        }

        if (bucket_count == 0) {
            scalar = renormalize_scaled_complex_if_needed(
                scalar.value,
                scalar.half_pow2_exp + 2
            );
            if (max_scope < 1) {
                max_scope = 1;
            }
            continue;
        }

        for (factor_idx = 0; factor_idx < nvars; ++factor_idx) {
            if (union_marks[factor_idx]) {
                if (factor_idx == var) {
                    var_pos = union_arity;
                }
                union_scope[union_arity++] = factor_idx;
            }
        }
        if (var_pos < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Elimination variable missing from union scope.");
            goto error;
        }
        if (union_arity > max_scope) {
            max_scope = union_arity;
        }

        {
            Py_ssize_t new_arity = union_arity - 1;
            size_t table_size;
            NativeScaledComplex *new_table = NULL;
            Py_ssize_t *new_scope = NULL;
            Py_ssize_t **positions = NULL;
            Py_ssize_t bucket_idx;
            Py_ssize_t insert_idx = -1;

            if (new_arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
                PyErr_SetString(PyExc_OverflowError, "Union scope too wide for native factor DP.");
                goto error;
            }
            table_size = ((size_t) 1) << (size_t) new_arity;

            new_table = PyMem_Malloc((table_size > 0 ? table_size : 1) * sizeof(NativeScaledComplex));
            positions = PyMem_Malloc((size_t) bucket_count * sizeof(Py_ssize_t *));
            if (new_arity > 0) {
                new_scope = PyMem_Malloc((size_t) new_arity * sizeof(Py_ssize_t));
            }
            if (new_table == NULL || positions == NULL || (new_arity > 0 && new_scope == NULL)) {
                PyMem_Free(new_table);
                PyMem_Free(positions);
                PyMem_Free(new_scope);
                PyErr_NoMemory();
                goto error;
            }

            if (new_arity > 0) {
                Py_ssize_t out_idx = 0;
                for (factor_idx = 0; factor_idx < union_arity; ++factor_idx) {
                    if (factor_idx == var_pos) {
                        continue;
                    }
                    new_scope[out_idx++] = union_scope[factor_idx];
                }
            }

            for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                NativeScaledFactor *factor = &factors[bucket_indexes[bucket_idx]];
                Py_ssize_t scope_idx;
                positions[bucket_idx] = PyMem_Malloc((size_t) factor->arity * sizeof(Py_ssize_t));
                if (positions[bucket_idx] == NULL) {
                    for (scope_idx = 0; scope_idx < bucket_idx; ++scope_idx) {
                        PyMem_Free(positions[scope_idx]);
                    }
                    PyMem_Free(new_table);
                    PyMem_Free(positions);
                    PyMem_Free(new_scope);
                    PyErr_NoMemory();
                    goto error;
                }
                for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                    Py_ssize_t union_idx;
                    positions[bucket_idx][scope_idx] = -1;
                    for (union_idx = 0; union_idx < union_arity; ++union_idx) {
                        if (union_scope[union_idx] == factor->scope[scope_idx]) {
                            positions[bucket_idx][scope_idx] = union_idx;
                            break;
                        }
                    }
                    if (positions[bucket_idx][scope_idx] < 0) {
                        for (union_idx = 0; union_idx <= bucket_idx; ++union_idx) {
                            PyMem_Free(positions[union_idx]);
                        }
                        PyMem_Free(new_table);
                        PyMem_Free(positions);
                        PyMem_Free(new_scope);
                        PyErr_SetString(PyExc_RuntimeError, "Failed to map factor scope into union scope.");
                        goto error;
                    }
                }
            }

            for (factor_idx = 0; factor_idx < (Py_ssize_t) table_size; ++factor_idx) {
                uint64_t reduced_assignment = (uint64_t) factor_idx;
                NativeScaledComplex total = scaled_complex_make(0.0, 0.0, 0);
                int fixed_value;
                for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                    uint64_t low_mask = (var_pos == 0) ? 0ULL : ((((uint64_t) 1) << (size_t) var_pos) - 1ULL);
                    uint64_t full_assignment =
                        (reduced_assignment & low_mask) |
                        (((uint64_t) fixed_value) << (size_t) var_pos) |
                        ((reduced_assignment & ~low_mask) << 1);
                    NativeScaledComplex weight = scaled_complex_make(1.0, 0.0, 0);
                    for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                        NativeScaledFactor *factor = &factors[bucket_indexes[bucket_idx]];
                        Py_ssize_t table_idx = project_assignment_bits_u64(
                            full_assignment,
                            positions[bucket_idx],
                            factor->arity
                        );
                        weight = mul_scaled_complex(weight, factor->table[table_idx]);
                    }
                    total = add_scaled_complex(total, weight);
                }
                new_table[factor_idx] = total;
            }

            for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                PyMem_Free(positions[bucket_idx]);
            }
            PyMem_Free(positions);
            positions = NULL;

            for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                free_native_scaled_factor(&factors[bucket_indexes[bucket_idx]]);
            }

            if (new_arity == 0) {
                scalar = mul_scaled_complex(scalar, new_table[0]);
                PyMem_Free(new_table);
                PyMem_Free(new_scope);
                continue;
            }

            for (factor_idx = 0; factor_idx < factor_count; ++factor_idx) {
                NativeScaledFactor *factor = &factors[factor_idx];
                Py_ssize_t table_idx;
                if (!scaled_scopes_equal(factor, new_scope, new_arity)) {
                    continue;
                }
                for (table_idx = 0; table_idx < (Py_ssize_t) table_size; ++table_idx) {
                    factor->table[table_idx] = mul_scaled_complex(factor->table[table_idx], new_table[table_idx]);
                }
                PyMem_Free(new_table);
                PyMem_Free(new_scope);
                new_table = NULL;
                new_scope = NULL;
                insert_idx = -1;
                break;
            }

            if (new_scope == NULL) {
                continue;
            }

            for (factor_idx = 0; factor_idx < factor_count; ++factor_idx) {
                if (!factors[factor_idx].alive) {
                    insert_idx = factor_idx;
                    break;
                }
            }
            if (insert_idx < 0) {
                if (factor_count >= factor_capacity) {
                    PyMem_Free(new_table);
                    PyMem_Free(new_scope);
                    PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
                    goto error;
                }
                insert_idx = factor_count++;
            }

            factors[insert_idx].arity = new_arity;
            factors[insert_idx].scope = new_scope;
            factors[insert_idx].table = new_table;
            factors[insert_idx].alive = 1;
        }
    }

    for (pos = 0; pos < factor_count; ++pos) {
        if (factors[pos].alive) {
            PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
            goto error;
        }
    }

    total_obj = PyComplex_FromDoubles(scalar.value.real, scalar.value.imag);
    exp_obj = PyLong_FromLongLong(scalar.half_pow2_exp);
    width_obj = PyLong_FromSsize_t(max_scope);
    if (total_obj == NULL || exp_obj == NULL || width_obj == NULL) {
        goto error;
    }
    result = PyTuple_Pack(3, total_obj, exp_obj, width_obj);

error:
    if (factors != NULL) {
        for (pos = 0; pos < factor_count; ++pos) {
            free_native_scaled_factor(&factors[pos]);
        }
    }
    PyMem_Free(factors);
    PyMem_Free(bucket_indexes);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    Py_XDECREF(order_seq);
    Py_XDECREF(total_obj);
    Py_XDECREF(exp_obj);
    Py_XDECREF(width_obj);
    return result;
}


static int build_q3_free_treewidth_plan_internal(
    Py_ssize_t nvars,
    long level,
    PyObject *q2,
    PyObject *order_seq,
    int with_tables,
    NativeQ3FreeTreewidthPlan **out_plan
)
{
    NativeQ3FreeTreewidthPlan *plan = NULL;
    NativePlanScopeFactor *states = NULL;
    Q2PhaseRecord *q2_records = NULL;
    Py_ssize_t *bucket_slots = NULL;
    Py_ssize_t *union_scope = NULL;
    char *union_marks = NULL;
    Py_ssize_t q2_count = 0;
    Py_ssize_t q2_capacity = 0;
    Py_ssize_t factor_capacity;
    Py_ssize_t order_len;
    Py_ssize_t pos;
    PyObject *key;
    PyObject *value;
    long long mod_q1;
    long long mod_q2;

    *out_plan = NULL;
    if (!PyDict_Check(q2)) {
        PyErr_SetString(PyExc_TypeError, "q2 must be a dict.");
        return -1;
    }
    if (level < 1 || level > 30) {
        PyErr_SetString(PyExc_ValueError, "level must lie in [1, 30] for native q3-free batch DP.");
        return -1;
    }

    mod_q1 = 1LL << level;
    mod_q2 = (level > 1) ? (1LL << (level - 1)) : 1LL;
    order_len = PySequence_Fast_GET_SIZE(order_seq);
    q2_capacity = PyDict_Size(q2);
    factor_capacity = q2_capacity + order_len + 1;
    if (factor_capacity < 1) {
        factor_capacity = 1;
    }

    plan = PyMem_Calloc(1, sizeof(NativeQ3FreeTreewidthPlan));
    states = PyMem_Calloc((size_t) factor_capacity, sizeof(NativePlanScopeFactor));
    bucket_slots = PyMem_Malloc((size_t) factor_capacity * sizeof(Py_ssize_t));
    union_scope = PyMem_Malloc((size_t) (nvars > 0 ? nvars : 1) * sizeof(Py_ssize_t));
    union_marks = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(char));
    if (plan == NULL || states == NULL || bucket_slots == NULL || union_scope == NULL || union_marks == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    if (q2_capacity > 0) {
        q2_records = PyMem_Malloc((size_t) q2_capacity * sizeof(Q2PhaseRecord));
        if (q2_records == NULL) {
            PyErr_NoMemory();
            goto error;
        }
    }

    plan->nvars = nvars;
    plan->level = level;
    plan->mod_q1 = mod_q1;
    plan->slot_arities = PyMem_Calloc((size_t) factor_capacity, sizeof(Py_ssize_t));
    plan->slot_table_sizes = PyMem_Calloc((size_t) factor_capacity, sizeof(size_t));
    plan->steps = PyMem_Calloc((size_t) (order_len > 0 ? order_len : 1), sizeof(NativeQ3FreePlanStep));
    if (plan->slot_arities == NULL || plan->slot_table_sizes == NULL || plan->steps == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    if (with_tables && q2_capacity > 0) {
        plan->initial_tables = PyMem_Calloc((size_t) q2_capacity, sizeof(NativeScaledComplex *));
        if (plan->initial_tables == NULL) {
            PyErr_NoMemory();
            goto error;
        }
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long long coeff;
        long long residue;
        NativePlanScopeFactor *factor;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        if (left < 0 || left >= nvars || right < 0 || right >= nvars) {
            PyErr_SetString(PyExc_ValueError, "q2 key contains an out-of-range variable.");
            goto error;
        }
        coeff = PyLong_AsLongLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = positive_mod_ll(coeff, mod_q2);
        if (residue == 0) {
            continue;
        }
        factor = &states[q2_count];
        factor->arity = 2;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(2 * sizeof(Py_ssize_t));
        if (factor->scope == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = left;
        factor->scope[1] = right;
        q2_records[q2_count].left = left;
        q2_records[q2_count].right = right;
        q2_records[q2_count].shift = positive_mod_ll(2LL * residue, mod_q1);
        plan->slot_arities[q2_count] = 2;
        plan->slot_table_sizes[q2_count] = 4;
        if (with_tables) {
            NativeScaledComplex *table = PyMem_Malloc(4 * sizeof(NativeScaledComplex));
            if (table == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            table[0] = scaled_complex_make(1.0, 0.0, 0);
            table[1] = scaled_complex_make(1.0, 0.0, 0);
            table[2] = scaled_complex_make(1.0, 0.0, 0);
            {
                NativeComplex phase = omega_dyadic_value(q2_records[q2_count].shift, mod_q1);
                table[3] = scaled_complex_make(phase.real, phase.imag, 0);
            }
            plan->initial_tables[q2_count] = table;
        }
        q2_count += 1;
    }

    plan->initial_slot_count = q2_count;
    plan->slot_count = q2_count;

    for (pos = 0; pos < order_len; ++pos) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(order_seq, pos));
        NativeQ3FreePlanStep *step;
        Py_ssize_t bucket_count = 0;
        Py_ssize_t union_arity = 0;
        Py_ssize_t var_pos = -1;
        Py_ssize_t slot_idx;
        Py_ssize_t position_cursor = 0;

        if (var == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (var < 0 || var >= nvars) {
            PyErr_SetString(PyExc_ValueError, "order entries must lie in [0, nvars).");
            goto error;
        }

        step = &plan->steps[plan->step_count];
        step->var = var;
        step->output_slot = -1;

        if (nvars > 0) {
            memset(union_marks, 0, (size_t) nvars * sizeof(char));
        }
        for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
            Py_ssize_t scope_idx;
            NativePlanScopeFactor *factor = &states[slot_idx];
            if (!factor->alive) {
                continue;
            }
            for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                if (factor->scope[scope_idx] == var) {
                    bucket_slots[bucket_count++] = slot_idx;
                    for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                        union_marks[factor->scope[scope_idx]] = 1;
                    }
                    break;
                }
            }
        }

        if (bucket_count == 0) {
            if (plan->max_scope < 1) {
                plan->max_scope = 1;
            }
            plan->step_count += 1;
            continue;
        }

        for (slot_idx = 0; slot_idx < nvars; ++slot_idx) {
            if (union_marks[slot_idx]) {
                if (slot_idx == var) {
                    var_pos = union_arity;
                }
                union_scope[union_arity++] = slot_idx;
            }
        }
        if (var_pos < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Elimination variable missing from union scope.");
            goto error;
        }
        if (union_arity > plan->max_scope) {
            plan->max_scope = union_arity;
        }

        step->bucket_count = bucket_count;
        step->union_arity = union_arity;
        step->var_pos = var_pos;
        step->bucket_slot_ids = PyMem_Malloc((size_t) bucket_count * sizeof(Py_ssize_t));
        step->bucket_arities = PyMem_Malloc((size_t) bucket_count * sizeof(Py_ssize_t));
        step->bucket_pos_offsets = PyMem_Malloc((size_t) (bucket_count + 1) * sizeof(Py_ssize_t));
        if (step->bucket_slot_ids == NULL || step->bucket_arities == NULL || step->bucket_pos_offsets == NULL) {
            PyErr_NoMemory();
            goto error;
        }

        for (slot_idx = 0; slot_idx < bucket_count; ++slot_idx) {
            position_cursor += states[bucket_slots[slot_idx]].arity;
        }
        step->bucket_positions = PyMem_Malloc((size_t) (position_cursor > 0 ? position_cursor : 1) * sizeof(Py_ssize_t));
        if (step->bucket_positions == NULL) {
            PyErr_NoMemory();
            goto error;
        }

        position_cursor = 0;
        for (slot_idx = 0; slot_idx < bucket_count; ++slot_idx) {
            Py_ssize_t source_slot = bucket_slots[slot_idx];
            NativePlanScopeFactor *factor = &states[source_slot];
            Py_ssize_t scope_idx;
            step->bucket_slot_ids[slot_idx] = source_slot;
            step->bucket_arities[slot_idx] = factor->arity;
            step->bucket_pos_offsets[slot_idx] = position_cursor;
            for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                Py_ssize_t union_idx;
                Py_ssize_t mapped = -1;
                for (union_idx = 0; union_idx < union_arity; ++union_idx) {
                    if (union_scope[union_idx] == factor->scope[scope_idx]) {
                        mapped = union_idx;
                        break;
                    }
                }
                if (mapped < 0) {
                    PyErr_SetString(PyExc_RuntimeError, "Failed to map factor scope into union scope.");
                    goto error;
                }
                step->bucket_positions[position_cursor++] = mapped;
            }
        }
        step->bucket_pos_offsets[bucket_count] = position_cursor;

        if (union_arity == 1) {
            for (slot_idx = 0; slot_idx < bucket_count; ++slot_idx) {
                NativePlanScopeFactor *factor = &states[bucket_slots[slot_idx]];
                factor->alive = 0;
                PyMem_Free(factor->scope);
                factor->scope = NULL;
                factor->arity = 0;
            }
            step->output_mode = 0;
            step->output_slot = -1;
            plan->step_count += 1;
            continue;
        }

        {
            Py_ssize_t new_arity = union_arity - 1;
            Py_ssize_t *new_scope = PyMem_Malloc((size_t) new_arity * sizeof(Py_ssize_t));
            Py_ssize_t out_idx = 0;
            if (new_arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
                PyErr_SetString(PyExc_OverflowError, "Union scope too wide for native q3-free batch DP.");
                PyMem_Free(new_scope);
                goto error;
            }
            if (new_scope == NULL) {
                PyErr_NoMemory();
                goto error;
            }

            for (slot_idx = 0; slot_idx < union_arity; ++slot_idx) {
                if (slot_idx == var_pos) {
                    continue;
                }
                new_scope[out_idx++] = union_scope[slot_idx];
            }

            for (slot_idx = 0; slot_idx < bucket_count; ++slot_idx) {
                NativePlanScopeFactor *factor = &states[bucket_slots[slot_idx]];
                factor->alive = 0;
                PyMem_Free(factor->scope);
                factor->scope = NULL;
                factor->arity = 0;
            }

            step->output_mode = 1;
            for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
                if (plan_scopes_equal(&states[slot_idx], new_scope, new_arity)) {
                    step->output_mode = 2;
                    step->output_slot = slot_idx;
                    PyMem_Free(new_scope);
                    new_scope = NULL;
                    break;
                }
            }
            if (step->output_mode == 1) {
                Py_ssize_t new_slot = plan->slot_count;
                states[new_slot].arity = new_arity;
                states[new_slot].scope = new_scope;
                states[new_slot].alive = 1;
                plan->slot_arities[new_slot] = new_arity;
                plan->slot_table_sizes[new_slot] = ((size_t) 1) << (size_t) new_arity;
                step->output_slot = new_slot;
                plan->slot_count += 1;
            }
        }

        plan->step_count += 1;
    }

    for (pos = 0; pos < plan->slot_count; ++pos) {
        if (states[pos].alive) {
            PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
            goto error;
        }
    }

    for (pos = 0; pos < factor_capacity; ++pos) {
        PyMem_Free(states[pos].scope);
    }
    PyMem_Free(states);
    PyMem_Free(q2_records);
    PyMem_Free(bucket_slots);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    *out_plan = plan;
    return 0;

error:
    if (states != NULL) {
        for (pos = 0; pos < factor_capacity; ++pos) {
            PyMem_Free(states[pos].scope);
        }
    }
    PyMem_Free(states);
    PyMem_Free(q2_records);
    PyMem_Free(bucket_slots);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    free_q3_free_treewidth_plan(plan);
    return -1;
}


static PyObject *evaluate_q3_free_treewidth_plan_batch(
    NativeQ3FreeTreewidthPlan *plan,
    PyObject *q1_batch_obj
)
{
    PyObject *batch_seq = NULL;
    PyObject *results = NULL;
    Py_ssize_t batch_size;
    Py_ssize_t row_idx;

    batch_seq = PySequence_Fast(q1_batch_obj, "q1_batch must be a sequence of q1 rows.");
    if (batch_seq == NULL) {
        return NULL;
    }
    batch_size = PySequence_Fast_GET_SIZE(batch_seq);
    results = PyList_New(batch_size);
    if (results == NULL) {
        Py_DECREF(batch_seq);
        return NULL;
    }

    for (row_idx = 0; row_idx < batch_size; ++row_idx) {
        PyObject *q1_seq = NULL;
        long long *q1_row = NULL;
        NativeScaledComplex **slot_tables = NULL;
        NativeScaledComplex scalar = scaled_complex_make(1.0, 0.0, 0);
        PyObject *row_result = NULL;
        PyObject *total_obj = NULL;
        PyObject *exp_obj = NULL;
        PyObject *width_obj = NULL;
        int row_failed = 0;
        Py_ssize_t slot_idx;
        Py_ssize_t step_idx;

        q1_seq = PySequence_Fast(
            PySequence_Fast_GET_ITEM(batch_seq, row_idx),
            "Each q1 row must be a sequence."
        );
        if (q1_seq == NULL) {
            goto error;
        }
        if (PySequence_Fast_GET_SIZE(q1_seq) != plan->nvars) {
            PyErr_SetString(PyExc_ValueError, "Each q1 row must have length nvars.");
            Py_DECREF(q1_seq);
            goto error;
        }

        q1_row = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(long long));
        slot_tables = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
        if (q1_row == NULL || slot_tables == NULL) {
            Py_DECREF(q1_seq);
            PyErr_NoMemory();
            row_failed = 1;
            goto row_cleanup;
        }

        for (slot_idx = 0; slot_idx < plan->nvars; ++slot_idx) {
            long long coeff = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(q1_seq, slot_idx));
            if (coeff == -1 && PyErr_Occurred()) {
                row_failed = 1;
                break;
            }
            q1_row[slot_idx] = positive_mod_ll(coeff, plan->mod_q1);
        }
        Py_DECREF(q1_seq);
        q1_seq = NULL;
        if (row_failed) {
            goto row_cleanup;
        }

        for (slot_idx = 0; slot_idx < plan->initial_slot_count; ++slot_idx) {
            size_t table_size_bytes = plan->slot_table_sizes[slot_idx] * sizeof(NativeScaledComplex);
            slot_tables[slot_idx] = PyMem_Malloc(table_size_bytes > 0 ? table_size_bytes : sizeof(NativeScaledComplex));
            if (slot_tables[slot_idx] == NULL) {
                PyErr_NoMemory();
                row_failed = 1;
                break;
            }
            memcpy(slot_tables[slot_idx], plan->initial_tables[slot_idx], table_size_bytes);
        }
        if (row_failed) {
            goto row_cleanup;
        }

        for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
            NativeQ3FreePlanStep *step = &plan->steps[step_idx];
            long long unary_coeff = q1_row[step->var];

            if (step->bucket_count == 0) {
                if (unary_coeff == 0) {
                    scalar = renormalize_scaled_complex_if_needed(
                        scalar.value,
                        scalar.half_pow2_exp + 2
                    );
                } else {
                    NativeComplex phase = omega_dyadic_value(unary_coeff, plan->mod_q1);
                    NativeScaledComplex unary_total = add_scaled_complex(
                        scaled_complex_make(1.0, 0.0, 0),
                        scaled_complex_make(phase.real, phase.imag, 0)
                    );
                    scalar = mul_scaled_complex(scalar, unary_total);
                }
                continue;
            }

            {
                Py_ssize_t new_arity = step->union_arity - 1;
                size_t table_size = ((size_t) 1) << (size_t) new_arity;
                NativeScaledComplex *new_table = PyMem_Malloc(
                    (table_size > 0 ? table_size : 1) * sizeof(NativeScaledComplex)
                );
                NativeScaledComplex unary_phase = scaled_complex_make(1.0, 0.0, 0);
                int has_unary = unary_coeff != 0;
                size_t assignment_idx;
                if (new_table == NULL) {
                    PyErr_NoMemory();
                    row_failed = 1;
                    break;
                }
                if (has_unary) {
                    NativeComplex phase = omega_dyadic_value(unary_coeff, plan->mod_q1);
                    unary_phase = scaled_complex_make(phase.real, phase.imag, 0);
                }

                for (assignment_idx = 0; assignment_idx < table_size; ++assignment_idx) {
                    uint64_t reduced_assignment = (uint64_t) assignment_idx;
                    NativeScaledComplex total = scaled_complex_make(0.0, 0.0, 0);
                    int fixed_value;
                    for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                        uint64_t low_mask = (step->var_pos == 0) ? 0ULL : ((((uint64_t) 1) << (size_t) step->var_pos) - 1ULL);
                        uint64_t full_assignment =
                            (reduced_assignment & low_mask) |
                            (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                            ((reduced_assignment & ~low_mask) << 1);
                        NativeScaledComplex weight = scaled_complex_make(1.0, 0.0, 0);
                        Py_ssize_t bucket_idx;
                        if (has_unary && fixed_value) {
                            weight = unary_phase;
                        }
                        for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                            Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                            Py_ssize_t table_idx = project_assignment_bits_u64(
                                full_assignment,
                                step->bucket_positions + step->bucket_pos_offsets[bucket_idx],
                                step->bucket_arities[bucket_idx]
                            );
                            if (slot_tables[slot_id] == NULL) {
                                PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native q3-free treewidth plan.");
                                row_failed = 1;
                                break;
                            }
                            weight = mul_scaled_complex(weight, slot_tables[slot_id][table_idx]);
                        }
                        if (row_failed) {
                            break;
                        }
                        total = add_scaled_complex(total, weight);
                    }
                    if (row_failed) {
                        break;
                    }
                    new_table[assignment_idx] = total;
                }
                if (row_failed) {
                    PyMem_Free(new_table);
                    break;
                }

                for (slot_idx = 0; slot_idx < step->bucket_count; ++slot_idx) {
                    Py_ssize_t source_slot = step->bucket_slot_ids[slot_idx];
                    PyMem_Free(slot_tables[source_slot]);
                    slot_tables[source_slot] = NULL;
                }

                if (step->output_mode == 0) {
                    scalar = mul_scaled_complex(scalar, new_table[0]);
                    PyMem_Free(new_table);
                } else if (step->output_mode == 1) {
                    slot_tables[step->output_slot] = new_table;
                } else {
                    size_t merge_idx;
                    NativeScaledComplex *target = slot_tables[step->output_slot];
                    if (target == NULL) {
                        PyErr_SetString(PyExc_RuntimeError, "Missing merge target for native q3-free treewidth plan.");
                        PyMem_Free(new_table);
                        row_failed = 1;
                        break;
                    }
                    for (merge_idx = 0; merge_idx < table_size; ++merge_idx) {
                        target[merge_idx] = mul_scaled_complex(target[merge_idx], new_table[merge_idx]);
                    }
                    PyMem_Free(new_table);
                }
            }
        }

        if (!row_failed) {
            for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
                if (slot_tables[slot_idx] != NULL) {
                    PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
                    row_failed = 1;
                    break;
                }
            }
        }

        if (!row_failed) {
            total_obj = PyComplex_FromDoubles(scalar.value.real, scalar.value.imag);
            exp_obj = PyLong_FromLongLong(scalar.half_pow2_exp);
            width_obj = PyLong_FromSsize_t(plan->max_scope);
            if (total_obj == NULL || exp_obj == NULL || width_obj == NULL) {
                row_failed = 1;
            } else {
                row_result = PyTuple_Pack(3, total_obj, exp_obj, width_obj);
                if (row_result == NULL) {
                    row_failed = 1;
                }
            }
        }

row_cleanup:
        Py_XDECREF(total_obj);
        Py_XDECREF(exp_obj);
        Py_XDECREF(width_obj);
        if (slot_tables != NULL) {
            for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
                PyMem_Free(slot_tables[slot_idx]);
            }
        }
        PyMem_Free(slot_tables);
        PyMem_Free(q1_row);

        if (row_failed) {
            Py_XDECREF(row_result);
            Py_DECREF(results);
            Py_DECREF(batch_seq);
            return NULL;
        }
        PyList_SET_ITEM(results, row_idx, row_result);
    }

    Py_DECREF(batch_seq);
    return results;

error:
    Py_XDECREF(results);
    Py_XDECREF(batch_seq);
    return NULL;
}


static PyObject *build_q3_free_treewidth_plan_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    long level;
    PyObject *q2;
    PyObject *order_obj;
    PyObject *order_seq = NULL;
    NativeQ3FreeTreewidthPlan *plan = NULL;
    PyObject *capsule = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "nlOO", &nvars, &level, &q2, &order_obj)) {
        return NULL;
    }
    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (order_seq == NULL) {
        return NULL;
    }
    if (build_q3_free_treewidth_plan_internal(nvars, level, q2, order_seq, 1, &plan) < 0) {
        Py_DECREF(order_seq);
        return NULL;
    }
    capsule = PyCapsule_New(plan, q3_free_treewidth_plan_capsule_name, q3_free_treewidth_plan_capsule_destructor);
    Py_DECREF(order_seq);
    if (capsule == NULL) {
        free_q3_free_treewidth_plan(plan);
        return NULL;
    }
    return capsule;
}


static PyObject *sum_q3_free_treewidth_preplanned_batch_scaled_native(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    PyObject *q1_batch_obj;
    NativeQ3FreeTreewidthPlan *plan;

    (void) self;

    if (!PyArg_ParseTuple(args, "OO", &capsule, &q1_batch_obj)) {
        return NULL;
    }
    plan = (NativeQ3FreeTreewidthPlan *) PyCapsule_GetPointer(capsule, q3_free_treewidth_plan_capsule_name);
    if (plan == NULL) {
        return NULL;
    }
    return evaluate_q3_free_treewidth_plan_batch(plan, q1_batch_obj);
}


static PyObject *q3_free_treewidth_dp_work_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    long level;
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *order_obj;
    PyObject *q1_seq = NULL;
    PyObject *order_seq = NULL;
    NativePlanScopeFactor *factors = NULL;
    Py_ssize_t *bucket_slots = NULL;
    Py_ssize_t *union_scope = NULL;
    char *union_marks = NULL;
    Py_ssize_t factor_capacity;
    Py_ssize_t factor_count = 0;
    unsigned long long work = 0ULL;
    Py_ssize_t order_len;
    Py_ssize_t pos;
    long long mod_q1;
    long long mod_q2;
    PyObject *key;
    PyObject *value;
    PyObject *result = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "nlOOO", &nvars, &level, &q1_obj, &q2, &order_obj)) {
        return NULL;
    }
    q1_seq = PySequence_Fast(q1_obj, "q1 must be a sequence.");
    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (q1_seq == NULL || order_seq == NULL) {
        goto cleanup;
    }
    if (PySequence_Fast_GET_SIZE(q1_seq) != nvars) {
        PyErr_SetString(PyExc_ValueError, "q1 must have length nvars.");
        goto cleanup;
    }
    if (!PyDict_Check(q2)) {
        PyErr_SetString(PyExc_TypeError, "q2 must be a dict.");
        goto cleanup;
    }
    if (level < 1 || level > 30) {
        PyErr_SetString(PyExc_ValueError, "level must lie in [1, 30] for native q3-free work estimation.");
        goto cleanup;
    }

    mod_q1 = 1LL << level;
    mod_q2 = (level > 1) ? (1LL << (level - 1)) : 1LL;
    order_len = PySequence_Fast_GET_SIZE(order_seq);
    factor_capacity = nvars + PyDict_Size(q2) + order_len + 1;
    if (factor_capacity < 1) {
        factor_capacity = 1;
    }

    factors = PyMem_Calloc((size_t) factor_capacity, sizeof(NativePlanScopeFactor));
    bucket_slots = PyMem_Malloc((size_t) factor_capacity * sizeof(Py_ssize_t));
    union_scope = PyMem_Malloc((size_t) (nvars > 0 ? nvars : 1) * sizeof(Py_ssize_t));
    union_marks = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(char));
    if (factors == NULL || bucket_slots == NULL || union_scope == NULL || union_marks == NULL) {
        PyErr_NoMemory();
        goto cleanup;
    }

    for (pos = 0; pos < nvars; ++pos) {
        long long coeff = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(q1_seq, pos));
        NativePlanScopeFactor *factor;
        if (coeff == -1 && PyErr_Occurred()) {
            goto cleanup;
        }
        if (positive_mod_ll(coeff, mod_q1) == 0) {
            continue;
        }
        factor = &factors[factor_count++];
        factor->arity = 1;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(sizeof(Py_ssize_t));
        if (factor->scope == NULL) {
            PyErr_NoMemory();
            goto cleanup;
        }
        factor->scope[0] = pos;
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long long coeff;
        NativePlanScopeFactor *factor;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto cleanup;
        }
        coeff = PyLong_AsLongLong(value);
        if (coeff == -1 && PyErr_Occurred()) {
            goto cleanup;
        }
        if (positive_mod_ll(coeff, mod_q2) == 0) {
            continue;
        }
        factor = &factors[factor_count++];
        factor->arity = 2;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(2 * sizeof(Py_ssize_t));
        if (factor->scope == NULL) {
            PyErr_NoMemory();
            goto cleanup;
        }
        factor->scope[0] = left;
        factor->scope[1] = right;
    }

    for (pos = 0; pos < order_len; ++pos) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(order_seq, pos));
        Py_ssize_t bucket_count = 0;
        Py_ssize_t union_arity = 0;
        Py_ssize_t var_pos = -1;
        Py_ssize_t factor_idx;
        unsigned long long bucket_cost;
        if (var == -1 && PyErr_Occurred()) {
            goto cleanup;
        }
        if (nvars > 0) {
            memset(union_marks, 0, (size_t) nvars * sizeof(char));
        }

        for (factor_idx = 0; factor_idx < factor_count; ++factor_idx) {
            Py_ssize_t scope_idx;
            NativePlanScopeFactor *factor = &factors[factor_idx];
            if (!factor->alive) {
                continue;
            }
            for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                if (factor->scope[scope_idx] == var) {
                    bucket_slots[bucket_count++] = factor_idx;
                    for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                        union_marks[factor->scope[scope_idx]] = 1;
                    }
                    break;
                }
            }
        }

        if (bucket_count == 0) {
            if (work != (unsigned long long) -1) {
                work += 1ULL;
            }
            continue;
        }

        for (factor_idx = 0; factor_idx < nvars; ++factor_idx) {
            if (union_marks[factor_idx]) {
                if (factor_idx == var) {
                    var_pos = union_arity;
                }
                union_scope[union_arity++] = factor_idx;
            }
        }
        bucket_cost = (unsigned long long) bucket_count;
        if (union_arity >= (Py_ssize_t) (8 * sizeof(unsigned long long))) {
            work = (unsigned long long) -1;
            break;
        }
        bucket_cost <<= (unsigned long long) union_arity;
        if ((((unsigned long long) -1) - work) < bucket_cost) {
            work = (unsigned long long) -1;
            break;
        }
        work += bucket_cost;

        for (factor_idx = 0; factor_idx < bucket_count; ++factor_idx) {
            NativePlanScopeFactor *factor = &factors[bucket_slots[factor_idx]];
            factor->alive = 0;
            PyMem_Free(factor->scope);
            factor->scope = NULL;
            factor->arity = 0;
        }

        if (union_arity > 1) {
            Py_ssize_t new_arity = union_arity - 1;
            Py_ssize_t *new_scope = PyMem_Malloc((size_t) new_arity * sizeof(Py_ssize_t));
            Py_ssize_t out_idx = 0;
            int found = 0;
            if (new_scope == NULL) {
                PyErr_NoMemory();
                goto cleanup;
            }
            for (factor_idx = 0; factor_idx < union_arity; ++factor_idx) {
                if (factor_idx == var_pos) {
                    continue;
                }
                new_scope[out_idx++] = union_scope[factor_idx];
            }
            for (factor_idx = 0; factor_idx < factor_count; ++factor_idx) {
                if (plan_scopes_equal(&factors[factor_idx], new_scope, new_arity)) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                factors[factor_count].arity = new_arity;
                factors[factor_count].scope = new_scope;
                factors[factor_count].alive = 1;
                factor_count += 1;
                new_scope = NULL;
            }
            PyMem_Free(new_scope);
        }
    }

    result = PyLong_FromUnsignedLongLong(work);

cleanup:
    if (factors != NULL) {
        for (pos = 0; pos < factor_count; ++pos) {
            PyMem_Free(factors[pos].scope);
        }
    }
    PyMem_Free(factors);
    PyMem_Free(bucket_slots);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    Py_XDECREF(q1_seq);
    Py_XDECREF(order_seq);
    return result;
}


static PyObject *sum_q3_free_treewidth_batch_scaled_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    long level;
    PyObject *q1_batch_obj;
    PyObject *q2;
    PyObject *order_obj;
    PyObject *order_seq = NULL;
    NativeQ3FreeTreewidthPlan *plan = NULL;
    PyObject *results = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "nlOOO", &nvars, &level, &q1_batch_obj, &q2, &order_obj)) {
        return NULL;
    }
    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (order_seq == NULL) {
        return NULL;
    }
    if (build_q3_free_treewidth_plan_internal(nvars, level, q2, order_seq, 1, &plan) < 0) {
        Py_DECREF(order_seq);
        return NULL;
    }
    results = evaluate_q3_free_treewidth_plan_batch(plan, q1_batch_obj);
    free_q3_free_treewidth_plan(plan);
    Py_DECREF(order_seq);
    return results;
}


