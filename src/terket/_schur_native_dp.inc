enum {
    LEVEL3_PEELED_FAST_MAX_BUCKET = 8,
    LEVEL3_PEELED_FAST_MAX_LOCAL_SUPPORT = 4,
};


typedef struct {
    const NativeFactor *factor;
    unsigned char factor_arity;
    signed char var_scope_pos;
    unsigned char local_positions[LEVEL3_PEELED_FAST_MAX_LOCAL_SUPPORT];
} NativeLevel3LocalFactorView;


static NativeDeferredDominantLocal *build_level3_dominant_local_deferred(
    Py_ssize_t var,
    const Py_ssize_t *union_scope,
    Py_ssize_t union_arity,
    const Py_ssize_t *new_scope,
    Py_ssize_t new_arity,
    const Py_ssize_t *bucket_indexes,
    Py_ssize_t bucket_count,
    NativeFactor *factors,
    Py_ssize_t *out_dominant_bucket
)
{
    const NativeFactor *dominant = NULL;
    Py_ssize_t dominant_bucket = -1;
    Py_ssize_t dominant_var_pos = -1;
    Py_ssize_t missing_var = -1;
    Py_ssize_t missing_pos = -1;
    Py_ssize_t local_support[LEVEL3_PEELED_FAST_MAX_LOCAL_SUPPORT];
    NativeLevel3LocalFactorView views[LEVEL3_PEELED_FAST_MAX_BUCKET];
    NativeDeferredDominantLocal *deferred = NULL;
    Py_ssize_t local_arity = 0;
    Py_ssize_t view_count = 0;
    size_t local_table_size;
    Py_ssize_t bucket_idx;
    Py_ssize_t local_idx;

    if (
        bucket_count < 2
        || bucket_count > LEVEL3_PEELED_FAST_MAX_BUCKET
        || new_arity < 1
        || union_arity != (new_arity + 1)
    ) {
        return NULL;
    }

    for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
        const NativeFactor *candidate = &factors[bucket_indexes[bucket_idx]];
        Py_ssize_t scope_idx = 0;
        Py_ssize_t union_idx = 0;
        Py_ssize_t local_missing = -1;

        if (!candidate->alive || candidate->arity != union_arity - 1) {
            continue;
        }
        while (scope_idx < candidate->arity && candidate->scope[scope_idx] != var) {
            ++scope_idx;
        }
        if (scope_idx >= candidate->arity) {
            continue;
        }
        while (union_idx < union_arity && scope_idx < candidate->arity) {
            if (union_scope[union_idx] == candidate->scope[scope_idx]) {
                ++union_idx;
                ++scope_idx;
                continue;
            }
            if (local_missing >= 0) {
                local_missing = -1;
                break;
            }
            local_missing = union_scope[union_idx];
            ++union_idx;
        }
        if (local_missing < 0) {
            if (scope_idx != candidate->arity || union_idx + 1 != union_arity) {
                continue;
            }
            local_missing = union_scope[union_idx];
        } else if (scope_idx != candidate->arity || union_idx != union_arity) {
            continue;
        }

        dominant = candidate;
        dominant_bucket = bucket_idx;
        missing_var = local_missing;
        break;
    }

    if (dominant == NULL || missing_var < 0) {
        return NULL;
    }

    for (local_idx = 0; local_idx < dominant->arity; ++local_idx) {
        if (dominant->scope[local_idx] == var) {
            dominant_var_pos = local_idx;
            break;
        }
    }
    if (dominant_var_pos < 0) {
        return NULL;
    }

    for (local_idx = 0; local_idx < new_arity; ++local_idx) {
        if (new_scope[local_idx] == missing_var) {
            missing_pos = local_idx;
            break;
        }
    }
    if (missing_pos < 0) {
        return NULL;
    }

    for (local_idx = 0; local_idx < new_arity; ++local_idx) {
        int appears = 0;
        for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
            const NativeFactor *factor;
            Py_ssize_t scope_idx;
            if (bucket_idx == dominant_bucket) {
                continue;
            }
            factor = &factors[bucket_indexes[bucket_idx]];
            for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                if (factor->scope[scope_idx] == new_scope[local_idx]) {
                    appears = 1;
                    break;
                }
            }
            if (appears) {
                break;
            }
        }
        if (appears) {
            if (local_arity >= LEVEL3_PEELED_FAST_MAX_LOCAL_SUPPORT) {
                return NULL;
            }
            local_support[local_arity++] = new_scope[local_idx];
        }
    }

    if (local_arity < 1 || local_arity > LEVEL3_PEELED_FAST_MAX_LOCAL_SUPPORT) {
        return NULL;
    }

    for (local_idx = 0; local_idx < local_arity; ++local_idx) {
        if (local_support[local_idx] == missing_var) {
            break;
        }
    }
    if (local_idx >= local_arity) {
        return NULL;
    }

    deferred = PyMem_Calloc(1, sizeof(NativeDeferredDominantLocal));
    if (deferred == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    deferred->dominant_result_positions = PyMem_Malloc(
        (size_t) (dominant->arity > 0 ? dominant->arity : 1) * sizeof(signed char)
    );
    if (deferred->dominant_result_positions == NULL) {
        PyMem_Free(deferred);
        PyErr_NoMemory();
        return NULL;
    }
    for (local_idx = 0; local_idx < dominant->arity; ++local_idx) {
        deferred->dominant_result_positions[local_idx] = -1;
    }
    deferred->missing_pos = (unsigned char) missing_pos;
    deferred->dominant_var_pos = (unsigned char) dominant_var_pos;
    deferred->dominant_arity = (unsigned char) dominant->arity;
    deferred->local_support_arity = (unsigned char) local_arity;
    deferred->dominant_factor = &factors[bucket_indexes[dominant_bucket]];

    for (local_idx = 0; local_idx < local_arity; ++local_idx) {
        Py_ssize_t new_scope_pos;
        if (local_support[local_idx] == missing_var) {
            deferred->local_result_positions[local_idx] = (unsigned char) missing_pos;
            continue;
        }
        for (new_scope_pos = 0; new_scope_pos < new_arity; ++new_scope_pos) {
            if (new_scope[new_scope_pos] == local_support[local_idx]) {
                deferred->local_result_positions[local_idx] = (unsigned char) new_scope_pos;
                break;
            }
        }
        if (new_scope_pos >= new_arity) {
            PyMem_Free(deferred->dominant_result_positions);
            PyMem_Free(deferred);
            return NULL;
        }
    }

    for (local_idx = 0; local_idx < dominant->arity; ++local_idx) {
        Py_ssize_t new_scope_pos;
        Py_ssize_t scope_var;
        if (local_idx == dominant_var_pos) {
            continue;
        }
        scope_var = dominant->scope[local_idx];
        for (new_scope_pos = 0; new_scope_pos < new_arity; ++new_scope_pos) {
            if (new_scope[new_scope_pos] == scope_var) {
                deferred->dominant_result_positions[local_idx] = (signed char) new_scope_pos;
                break;
            }
        }
        if (new_scope_pos >= new_arity) {
            PyMem_Free(deferred->dominant_result_positions);
            PyMem_Free(deferred);
            return NULL;
        }
    }

    for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
        const NativeFactor *factor;
        NativeLevel3LocalFactorView *view;
        Py_ssize_t scope_idx;
        Py_ssize_t local_pos_cursor = 0;
        if (bucket_idx == dominant_bucket) {
            continue;
        }
        factor = &factors[bucket_indexes[bucket_idx]];
        if (view_count >= LEVEL3_PEELED_FAST_MAX_BUCKET) {
            PyMem_Free(deferred->dominant_result_positions);
            PyMem_Free(deferred);
            return NULL;
        }
        view = &views[view_count];
        view->factor = factor;
        view->factor_arity = (unsigned char) factor->arity;
        view->var_scope_pos = -1;
        for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
            if (factor->scope[scope_idx] == var) {
                view->var_scope_pos = (signed char) scope_idx;
                continue;
            }
            for (local_idx = 0; local_idx < local_arity; ++local_idx) {
                if (local_support[local_idx] == factor->scope[scope_idx]) {
                    view->local_positions[local_pos_cursor++] = (unsigned char) local_idx;
                    break;
                }
            }
            if (local_idx >= local_arity) {
                PyMem_Free(deferred->dominant_result_positions);
                PyMem_Free(deferred);
                return NULL;
            }
        }
        if (view->var_scope_pos < 0) {
            PyMem_Free(deferred->dominant_result_positions);
            PyMem_Free(deferred);
            return NULL;
        }
        ++view_count;
    }

    local_table_size = ((size_t) 1) << (size_t) local_arity;
    for (local_idx = 0; local_idx < (Py_ssize_t) local_table_size; ++local_idx) {
        deferred->local_products[0][local_idx] = complex_make(1.0, 0.0);
        deferred->local_products[1][local_idx] = complex_make(1.0, 0.0);
    }

    for (bucket_idx = 0; bucket_idx < view_count; ++bucket_idx) {
        const NativeLevel3LocalFactorView *view = &views[bucket_idx];
        int fixed_value;
        for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
            for (local_idx = 0; local_idx < (Py_ssize_t) local_table_size; ++local_idx) {
                Py_ssize_t table_idx = 0;
                Py_ssize_t scope_idx;
                Py_ssize_t local_pos_idx = 0;
                for (scope_idx = 0; scope_idx < view->factor_arity; ++scope_idx) {
                    int bit;
                    if (scope_idx == view->var_scope_pos) {
                        bit = fixed_value;
                    } else {
                        bit = (int) (
                            (((size_t) local_idx) >> (size_t) view->local_positions[local_pos_idx]) & 1U
                        );
                        ++local_pos_idx;
                    }
                    table_idx |= ((Py_ssize_t) bit) << (size_t) scope_idx;
                }
                deferred->local_products[fixed_value][local_idx] = complex_mul(
                    deferred->local_products[fixed_value][local_idx],
                    view->factor->kind == NATIVE_FACTOR_KIND_CONCRETE
                        ? view->factor->table[table_idx]
                        : native_factor_value_at_index(view->factor, table_idx)
                );
            }
        }
    }

    *out_dominant_bucket = dominant_bucket;
    return deferred;
}


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
    Py_ssize_t **positions = NULL;
    Py_ssize_t *position_buffer = NULL;
    size_t position_buffer_capacity = 0;
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
    positions = PyMem_Malloc((size_t) factor_capacity * sizeof(Py_ssize_t *));
    if (factors == NULL || bucket_indexes == NULL || union_scope == NULL || union_marks == NULL || positions == NULL) {
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
            NativeDeferredDominantLocal *deferred_factor = NULL;
            Py_ssize_t deferred_dominant_bucket = -1;
            size_t required_positions;
            Py_ssize_t bucket_idx;
            Py_ssize_t insert_idx = -1;
            uint64_t low_mask;

            if (new_arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
                PyErr_SetString(PyExc_OverflowError, "Union scope too wide for native treewidth DP.");
                goto error;
            }
            table_size = ((size_t) 1) << (size_t) new_arity;
            low_mask = (var_pos == 0) ? 0ULL : ((((uint64_t) 1) << (size_t) var_pos) - 1ULL);

            new_table = PyMem_Calloc(table_size > 0 ? table_size : 1, sizeof(NativeComplex));
            if (new_arity > 0) {
                new_scope = PyMem_Malloc((size_t) new_arity * sizeof(Py_ssize_t));
            }
            if (new_table == NULL || (new_arity > 0 && new_scope == NULL)) {
                PyMem_Free(new_table);
                PyMem_Free(new_scope);
                PyErr_NoMemory();
                goto error;
            }
            required_positions = (size_t) bucket_count * (size_t) union_arity;
            if (required_positions > position_buffer_capacity) {
                Py_ssize_t *resized_positions = PyMem_Realloc(
                    position_buffer,
                    (required_positions > 0 ? required_positions : 1) * sizeof(Py_ssize_t)
                );
                if (resized_positions == NULL) {
                    PyMem_Free(new_table);
                    PyMem_Free(new_scope);
                    PyErr_NoMemory();
                    goto error;
                }
                position_buffer = resized_positions;
                position_buffer_capacity = required_positions;
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

            if (new_arity > 0) {
                deferred_factor = build_level3_dominant_local_deferred(
                    var,
                    union_scope,
                    union_arity,
                    new_scope,
                    new_arity,
                    bucket_indexes,
                    bucket_count,
                    factors,
                    &deferred_dominant_bucket
                );
                if (deferred_factor != NULL) {
                    NativeFactor *owned_dominant = PyMem_Calloc(1, sizeof(NativeFactor));
                    NativeFactor *source = &factors[bucket_indexes[deferred_dominant_bucket]];
                    if (owned_dominant == NULL) {
                        PyMem_Free(deferred_factor->dominant_result_positions);
                        PyMem_Free(deferred_factor);
                        PyMem_Free(new_table);
                        PyMem_Free(new_scope);
                        PyErr_NoMemory();
                        goto error;
                    }
                    *owned_dominant = *source;
                    source->scope = NULL;
                    source->table = NULL;
                    source->aux = NULL;
                    source->alive = 0;
                    source->arity = 0;
                    source->kind = NATIVE_FACTOR_KIND_CONCRETE;
                    deferred_factor->dominant_factor = owned_dominant;
                }
            }

            if (deferred_factor == NULL) {
                for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                    NativeFactor *factor = &factors[bucket_indexes[bucket_idx]];
                    Py_ssize_t scope_idx;
                    positions[bucket_idx] = position_buffer + (((size_t) bucket_idx) * (size_t) union_arity);
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
                            PyMem_Free(new_table);
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
                            weight = complex_mul(
                                weight,
                                factor->kind == NATIVE_FACTOR_KIND_CONCRETE
                                    ? factor->table[table_idx]
                                    : native_factor_value_at_index(factor, table_idx)
                            );
                        }
                        total = complex_add(total, weight);
                    }
                    new_table[factor_idx] = total;
                }
            }

            for (bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
                if (deferred_factor != NULL && bucket_idx == deferred_dominant_bucket) {
                    continue;
                }
                free_native_factor(&factors[bucket_indexes[bucket_idx]]);
            }

            if (new_arity == 0) {
                if (deferred_factor != NULL) {
                    scalar = complex_mul(scalar, native_factor_value_at_index((const NativeFactor *) &(NativeFactor){
                        .arity = new_arity,
                        .scope = new_scope,
                        .table = NULL,
                        .alive = 1,
                        .kind = NATIVE_FACTOR_KIND_DEFERRED_DOMLOCAL,
                        .aux = deferred_factor
                    }, 0));
                    if (deferred_factor->dominant_factor != NULL) {
                        free_native_factor(deferred_factor->dominant_factor);
                        PyMem_Free(deferred_factor->dominant_factor);
                    }
                    PyMem_Free(deferred_factor->dominant_result_positions);
                    PyMem_Free(deferred_factor);
                } else {
                    scalar = complex_mul(scalar, new_table[0]);
                    PyMem_Free(new_table);
                }
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
                    if (factor->kind != NATIVE_FACTOR_KIND_CONCRETE && materialize_native_factor(factor) < 0) {
                        PyMem_Free(new_table);
                        PyMem_Free(new_scope);
                        goto error;
                    }
                    if (deferred_factor != NULL) {
                        size_t merge_idx;
                        for (merge_idx = 0; merge_idx < table_size; ++merge_idx) {
                            factor->table[merge_idx] = complex_mul(
                                factor->table[merge_idx],
                                native_factor_value_at_index((const NativeFactor *) &(NativeFactor){
                                    .arity = new_arity,
                                    .scope = new_scope,
                                    .table = NULL,
                                    .alive = 1,
                                    .kind = NATIVE_FACTOR_KIND_DEFERRED_DOMLOCAL,
                                    .aux = deferred_factor
                                }, (Py_ssize_t) merge_idx)
                            );
                        }
                        if (deferred_factor->dominant_factor != NULL) {
                            free_native_factor(deferred_factor->dominant_factor);
                            PyMem_Free(deferred_factor->dominant_factor);
                        }
                        PyMem_Free(deferred_factor->dominant_result_positions);
                        PyMem_Free(deferred_factor);
                        PyMem_Free(new_table);
                    } else {
                        for (bucket_idx = 0; bucket_idx < (Py_ssize_t) table_size; ++bucket_idx) {
                            factor->table[bucket_idx] = complex_mul(factor->table[bucket_idx], new_table[bucket_idx]);
                        }
                        PyMem_Free(new_table);
                    }
                    PyMem_Free(new_scope);
                    new_table = NULL;
                    new_scope = NULL;
                    deferred_factor = NULL;
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
            factors[insert_idx].kind = (
                deferred_factor != NULL
                ? NATIVE_FACTOR_KIND_DEFERRED_DOMLOCAL
                : NATIVE_FACTOR_KIND_CONCRETE
            );
            factors[insert_idx].aux = deferred_factor;
            if (deferred_factor != NULL) {
                PyMem_Free(new_table);
                factors[insert_idx].table = NULL;
            }
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
    PyMem_Free(positions);
    PyMem_Free(position_buffer);
    Py_XDECREF(q1_seq);
    Py_XDECREF(order_seq);
    Py_XDECREF(total_obj);
    Py_XDECREF(width_obj);
    return result;
}


static int build_level3_treewidth_plan_internal(
    Py_ssize_t nvars,
    PyObject *q1_obj,
    PyObject *q2,
    PyObject *q3,
    PyObject *order_seq,
    NativeLevel3TreewidthPlan **out_plan
)
{
    NativeLevel3TreewidthPlan *plan = NULL;
    NativePlanScopeFactor *states = NULL;
    Py_ssize_t *bucket_slots = NULL;
    Py_ssize_t *union_scope = NULL;
    char *union_marks = NULL;
    PyObject *q1_seq = NULL;
    Py_ssize_t factor_capacity;
    Py_ssize_t factor_count = 0;
    Py_ssize_t order_len;
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;

    *out_plan = NULL;
    if (!PyDict_Check(q2) || !PyDict_Check(q3)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 must be dicts.");
        return -1;
    }
    q1_seq = PySequence_Fast(q1_obj, "q1 must be a sequence.");
    if (q1_seq == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(q1_seq) != nvars) {
        PyErr_SetString(PyExc_ValueError, "q1 must have length nvars.");
        Py_DECREF(q1_seq);
        return -1;
    }

    order_len = PySequence_Fast_GET_SIZE(order_seq);
    factor_capacity = PyDict_Size(q2) + PyDict_Size(q3) + (2 * nvars) + 1;
    if (factor_capacity < 1) {
        factor_capacity = 1;
    }

    plan = PyMem_Calloc(1, sizeof(NativeLevel3TreewidthPlan));
    states = PyMem_Calloc((size_t) factor_capacity, sizeof(NativePlanScopeFactor));
    bucket_slots = PyMem_Malloc((size_t) factor_capacity * sizeof(Py_ssize_t));
    union_scope = PyMem_Malloc((size_t) (nvars > 0 ? nvars : 1) * sizeof(Py_ssize_t));
    union_marks = PyMem_Calloc((size_t) (nvars > 0 ? nvars : 1), sizeof(char));
    if (plan == NULL || states == NULL || bucket_slots == NULL || union_scope == NULL || union_marks == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    plan->nvars = nvars;
    plan->slot_arities = PyMem_Calloc((size_t) factor_capacity, sizeof(Py_ssize_t));
    plan->slot_table_sizes = PyMem_Calloc((size_t) factor_capacity, sizeof(size_t));
    plan->slot_workspace_offsets = PyMem_Calloc((size_t) factor_capacity, sizeof(size_t));
    plan->initial_tables = PyMem_Calloc((size_t) factor_capacity, sizeof(NativeComplex *));
    plan->steps = PyMem_Calloc((size_t) (order_len > 0 ? order_len : 1), sizeof(NativeLevel3PlanStep));
    if (
        plan->slot_arities == NULL
        || plan->slot_table_sizes == NULL
        || plan->slot_workspace_offsets == NULL
        || plan->initial_tables == NULL
        || plan->steps == NULL
    ) {
        PyErr_NoMemory();
        goto error;
    }

    for (pos = 0; pos < nvars; ++pos) {
        long coeff = PyLong_AsLong(PySequence_Fast_GET_ITEM(q1_seq, pos));
        int residue;
        NativePlanScopeFactor *factor;
        NativeComplex *table;
        if (coeff == -1 && PyErr_Occurred()) {
            goto error;
        }
        residue = positive_mod(coeff, 8);
        if (residue == 0) {
            continue;
        }
        factor = &states[factor_count];
        factor->arity = 1;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(sizeof(Py_ssize_t));
        table = PyMem_Malloc(2 * sizeof(NativeComplex));
        if (factor->scope == NULL || table == NULL) {
            PyMem_Free(table);
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = pos;
        table[0] = complex_make(1.0, 0.0);
        table[1] = omega_level3_value(residue);
        plan->slot_arities[factor_count] = 1;
        plan->slot_table_sizes[factor_count] = 2;
        plan->initial_tables[factor_count] = table;
        factor_count += 1;
    }

    pos = 0;
    while (PyDict_Next(q2, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        long coeff;
        int residue;
        int shift;
        NativePlanScopeFactor *factor;
        NativeComplex *table;
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
        factor = &states[factor_count];
        factor->arity = 2;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(2 * sizeof(Py_ssize_t));
        table = PyMem_Malloc(4 * sizeof(NativeComplex));
        if (factor->scope == NULL || table == NULL) {
            PyMem_Free(table);
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = left;
        factor->scope[1] = right;
        table[0] = complex_make(1.0, 0.0);
        table[1] = complex_make(1.0, 0.0);
        table[2] = complex_make(1.0, 0.0);
        table[3] = omega_level3_value(shift);
        plan->slot_arities[factor_count] = 2;
        plan->slot_table_sizes[factor_count] = 4;
        plan->initial_tables[factor_count] = table;
        factor_count += 1;
    }

    pos = 0;
    while (PyDict_Next(q3, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        long coeff;
        int residue;
        NativePlanScopeFactor *factor;
        NativeComplex *table;
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
        factor = &states[factor_count];
        factor->arity = 3;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(3 * sizeof(Py_ssize_t));
        table = PyMem_Malloc(8 * sizeof(NativeComplex));
        if (factor->scope == NULL || table == NULL) {
            PyMem_Free(table);
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = a;
        factor->scope[1] = b;
        factor->scope[2] = c;
        for (table_idx = 0; table_idx < 8; ++table_idx) {
            table[table_idx] = complex_make(1.0, 0.0);
        }
        table[7] = omega_level3_value(4);
        plan->slot_arities[factor_count] = 3;
        plan->slot_table_sizes[factor_count] = 8;
        plan->initial_tables[factor_count] = table;
        factor_count += 1;
    }

    plan->initial_slot_count = factor_count;
    plan->slot_count = factor_count;

    for (pos = 0; pos < order_len; ++pos) {
        Py_ssize_t var = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(order_seq, pos));
        NativeLevel3PlanStep *step;
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
        step->low_mask = 0ULL;
        step->reduced_table_size = 0;
        step->bucket_table_indexes = NULL;

        memset(union_marks, 0, (size_t) (nvars > 0 ? nvars : 1) * sizeof(char));
        for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
            NativePlanScopeFactor *factor = &states[slot_idx];
            if (!factor->alive || !plan_scope_contains_var(factor, var)) {
                continue;
            }
            bucket_slots[bucket_count++] = slot_idx;
            {
                Py_ssize_t scope_idx;
                for (scope_idx = 0; scope_idx < factor->arity; ++scope_idx) {
                    union_marks[factor->scope[scope_idx]] = 1;
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
        step->low_mask = (var_pos == 0) ? 0ULL : ((((uint64_t) 1) << (size_t) var_pos) - 1ULL);
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
            step->reduced_table_size = 1;
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
            if (new_scope == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            if (new_arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
                PyErr_SetString(PyExc_OverflowError, "Union scope too wide for native level-3 treewidth plan.");
                PyMem_Free(new_scope);
                goto error;
            }
            step->reduced_table_size = ((size_t) 1) << (size_t) new_arity;
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

        if (
            step->reduced_table_size > 0
            && bucket_count > 0
            && ((size_t) bucket_count) <= (SIZE_MAX / (2 * step->reduced_table_size))
        ) {
            size_t total_index_count = ((size_t) bucket_count) * (2 * step->reduced_table_size);
            if (total_index_count <= (((size_t) 1) << 20)) {
                step->bucket_table_indexes = PyMem_Malloc(
                    (total_index_count > 0 ? total_index_count : 1) * sizeof(Py_ssize_t)
                );
                if (step->bucket_table_indexes == NULL) {
                    PyErr_NoMemory();
                    goto error;
                }
                for (slot_idx = 0; slot_idx < bucket_count; ++slot_idx) {
                    size_t assignment_idx;
                    size_t bucket_offset = ((size_t) slot_idx) * (2 * step->reduced_table_size);
                    for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                        uint64_t reduced_assignment = (uint64_t) assignment_idx;
                        int fixed_value;
                        for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                            uint64_t full_assignment =
                                (reduced_assignment & step->low_mask) |
                                (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                                ((reduced_assignment & ~step->low_mask) << 1);
                            step->bucket_table_indexes[
                                bucket_offset + (((size_t) fixed_value) * step->reduced_table_size) + assignment_idx
                            ] = project_assignment_bits_u64(
                                full_assignment,
                                step->bucket_positions + step->bucket_pos_offsets[slot_idx],
                                step->bucket_arities[slot_idx]
                            );
                        }
                    }
                }
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

    {
        size_t workspace_size = 0;
        size_t max_table = 1;
        for (pos = 0; pos < plan->slot_count; ++pos) {
            plan->slot_workspace_offsets[pos] = workspace_size;
            workspace_size += plan->slot_table_sizes[pos];
            if (plan->slot_table_sizes[pos] > max_table) {
                max_table = plan->slot_table_sizes[pos];
            }
        }
        plan->workspace_size = workspace_size;
        plan->workspace = PyMem_Malloc((workspace_size > 0 ? workspace_size : 1) * sizeof(NativeComplex));
        plan->merge_scratch_size = max_table;
        plan->merge_scratch = PyMem_Malloc((max_table > 0 ? max_table : 1) * sizeof(NativeComplex));
        plan->slot_views = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeComplex *));
        plan->slot_owned = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
        if (plan->workspace == NULL || plan->merge_scratch == NULL || plan->slot_views == NULL || plan->slot_owned == NULL) {
            PyErr_NoMemory();
            goto error;
        }
    }

    for (pos = 0; pos < factor_capacity; ++pos) {
        PyMem_Free(states[pos].scope);
    }
    PyMem_Free(states);
    PyMem_Free(bucket_slots);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    Py_DECREF(q1_seq);
    *out_plan = plan;
    return 0;

error:
    if (states != NULL) {
        for (pos = 0; pos < factor_capacity; ++pos) {
            PyMem_Free(states[pos].scope);
        }
    }
    PyMem_Free(states);
    PyMem_Free(bucket_slots);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    Py_XDECREF(q1_seq);
    free_level3_treewidth_plan(plan);
    return -1;
}


static void initialize_level3_plan_slot_views(NativeLevel3TreewidthPlan *plan)
{
    Py_ssize_t slot_idx;
    for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
        plan->slot_views[slot_idx] = NULL;
        plan->slot_owned[slot_idx] = 0U;
    }
    for (slot_idx = 0; slot_idx < plan->initial_slot_count; ++slot_idx) {
        plan->slot_views[slot_idx] = plan->initial_tables[slot_idx];
    }
}


static int ensure_owned_level3_plan_slot_table(
    NativeLevel3TreewidthPlan *plan,
    Py_ssize_t slot_id
)
{
    NativeComplex *dest;
    size_t table_size;
    if (plan->slot_owned[slot_id]) {
        return 0;
    }
    if (plan->slot_views[slot_id] == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Missing merge target for native level-3 treewidth plan.");
        return -1;
    }
    table_size = plan->slot_table_sizes[slot_id];
    dest = plan->workspace + plan->slot_workspace_offsets[slot_id];
    memcpy(dest, plan->slot_views[slot_id], table_size * sizeof(NativeComplex));
    plan->slot_views[slot_id] = dest;
    plan->slot_owned[slot_id] = 1U;
    return 0;
}


static int evaluate_level3_treewidth_plan(
    NativeLevel3TreewidthPlan *plan,
    NativeComplex *out_total
)
{
    NativeComplex scalar = complex_make(1.0, 0.0);
    Py_ssize_t step_idx;

    initialize_level3_plan_slot_views(plan);

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        NativeLevel3PlanStep *step = &plan->steps[step_idx];
        if (step->bucket_count == 0) {
            scalar.real *= 2.0;
            scalar.imag *= 2.0;
            continue;
        }

        if (step->output_mode == 0) {
            NativeComplex total = complex_make(0.0, 0.0);
            int fixed_value;
            for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                NativeComplex weight = complex_make(1.0, 0.0);
                Py_ssize_t bucket_idx;
                for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                    Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                    size_t table_idx = (size_t) fixed_value;
                    weight = complex_mul(weight, plan->slot_views[slot_id][table_idx]);
                }
                total = complex_add(total, weight);
            }
            {
                Py_ssize_t bucket_idx;
                for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                    Py_ssize_t source_slot = step->bucket_slot_ids[bucket_idx];
                    plan->slot_views[source_slot] = NULL;
                    plan->slot_owned[source_slot] = 0U;
                }
            }
            scalar = complex_mul(scalar, total);
            continue;
        }

        {
            NativeComplex *target_table;
            size_t assignment_idx;
            if (step->output_mode == 1) {
                target_table = plan->workspace + plan->slot_workspace_offsets[step->output_slot];
            } else {
                target_table = plan->merge_scratch;
            }

            for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                uint64_t reduced_assignment = (uint64_t) assignment_idx;
                NativeComplex total = complex_make(0.0, 0.0);
                int fixed_value;
                for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                    uint64_t full_assignment =
                        (reduced_assignment & step->low_mask) |
                        (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                        ((reduced_assignment & ~step->low_mask) << 1);
                    NativeComplex weight = complex_make(1.0, 0.0);
                    Py_ssize_t bucket_idx;
                    for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                        Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                        Py_ssize_t table_idx;
                        if (step->bucket_table_indexes != NULL) {
                            size_t bucket_offset = ((size_t) bucket_idx) * (2 * step->reduced_table_size);
                            table_idx = step->bucket_table_indexes[
                                bucket_offset + (((size_t) fixed_value) * step->reduced_table_size) + assignment_idx
                            ];
                        } else {
                            table_idx = project_assignment_bits_u64(
                                full_assignment,
                                step->bucket_positions + step->bucket_pos_offsets[bucket_idx],
                                step->bucket_arities[bucket_idx]
                            );
                        }
                        weight = complex_mul(weight, plan->slot_views[slot_id][table_idx]);
                    }
                    total = complex_add(total, weight);
                }
                target_table[assignment_idx] = total;
            }

            {
                Py_ssize_t bucket_idx;
                for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                    Py_ssize_t source_slot = step->bucket_slot_ids[bucket_idx];
                    plan->slot_views[source_slot] = NULL;
                    plan->slot_owned[source_slot] = 0U;
                }
            }

            if (step->output_mode == 1) {
                plan->slot_views[step->output_slot] = target_table;
                plan->slot_owned[step->output_slot] = 1U;
            } else {
                size_t merge_idx;
                if (ensure_owned_level3_plan_slot_table(plan, step->output_slot) < 0) {
                    return -1;
                }
                for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                    plan->slot_views[step->output_slot][merge_idx] = complex_mul(
                        plan->slot_views[step->output_slot][merge_idx],
                        target_table[merge_idx]
                    );
                }
            }
        }
    }

    for (step_idx = 0; step_idx < plan->slot_count; ++step_idx) {
        if (plan->slot_views[step_idx] != NULL) {
            PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
            return -1;
        }
    }
    *out_total = scalar;
    return 0;
}


static PyObject *build_level3_treewidth_plan_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *q1_obj;
    PyObject *q2;
    PyObject *q3;
    PyObject *order_obj;
    PyObject *order_seq = NULL;
    NativeLevel3TreewidthPlan *plan = NULL;
    PyObject *capsule = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOOOO", &nvars, &q1_obj, &q2, &q3, &order_obj)) {
        return NULL;
    }
    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (order_seq == NULL) {
        return NULL;
    }
    if (build_level3_treewidth_plan_internal(nvars, q1_obj, q2, q3, order_seq, &plan) < 0) {
        Py_DECREF(order_seq);
        return NULL;
    }
    Py_DECREF(order_seq);

    capsule = PyCapsule_New(plan, level3_treewidth_plan_capsule_name, level3_treewidth_plan_capsule_destructor);
    if (capsule == NULL) {
        free_level3_treewidth_plan(plan);
        return NULL;
    }
    return capsule;
}


static PyObject *sum_level3_treewidth_preplanned_native(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    NativeLevel3TreewidthPlan *plan;
    NativeComplex total;
    PyObject *total_obj = NULL;
    PyObject *width_obj = NULL;
    PyObject *result = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    plan = (NativeLevel3TreewidthPlan *) PyCapsule_GetPointer(capsule, level3_treewidth_plan_capsule_name);
    if (plan == NULL) {
        return NULL;
    }
    if (evaluate_level3_treewidth_plan(plan, &total) < 0) {
        return NULL;
    }

    total_obj = PyComplex_FromDoubles(total.real, total.imag);
    width_obj = PyLong_FromSsize_t(plan->max_scope);
    if (total_obj == NULL || width_obj == NULL) {
        Py_XDECREF(total_obj);
        Py_XDECREF(width_obj);
        return NULL;
    }
    result = PyTuple_Pack(2, total_obj, width_obj);
    Py_DECREF(total_obj);
    Py_DECREF(width_obj);
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

static inline Py_ssize_t q3_free_plan_step_table_index(
    const NativeQ3FreePlanStep *step,
    Py_ssize_t bucket_idx,
    int fixed_value,
    uint64_t full_assignment,
    size_t assignment_idx
);
static void initialize_q3_free_plan_slot_views(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables,
    unsigned char *slot_owned
);
static void free_owned_q3_free_plan_slot_views(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables,
    unsigned char *slot_owned
);
static int ensure_owned_q3_free_plan_slot_table(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables,
    unsigned char *slot_owned,
    Py_ssize_t slot_id
);


static int build_scaled_factor_treewidth_plan_internal(
    Py_ssize_t nvars,
    PyObject *factors_obj,
    PyObject *order_seq,
    NativeQ3FreeTreewidthPlan **out_plan
)
{
    NativeQ3FreeTreewidthPlan *plan = NULL;
    NativePlanScopeFactor *states = NULL;
    Py_ssize_t *bucket_slots = NULL;
    Py_ssize_t *union_scope = NULL;
    char *union_marks = NULL;
    Py_ssize_t factor_capacity;
    Py_ssize_t factor_count = 0;
    Py_ssize_t order_len;
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *value;

    *out_plan = NULL;
    if (!PyDict_Check(factors_obj)) {
        PyErr_SetString(PyExc_TypeError, "factors must be a dict keyed by scope tuples.");
        return -1;
    }

    order_len = PySequence_Fast_GET_SIZE(order_seq);
    factor_capacity = PyDict_Size(factors_obj) + order_len + 1;
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

    plan->nvars = nvars;
    plan->level = 0;
    plan->mod_q1 = 1;
    plan->slot_arities = PyMem_Calloc((size_t) factor_capacity, sizeof(Py_ssize_t));
    plan->slot_table_sizes = PyMem_Calloc((size_t) factor_capacity, sizeof(size_t));
    plan->steps = PyMem_Calloc((size_t) (order_len > 0 ? order_len : 1), sizeof(NativeQ3FreePlanStep));
    plan->initial_tables = PyMem_Calloc((size_t) factor_capacity, sizeof(NativeScaledComplex *));
    if (plan->slot_arities == NULL || plan->slot_table_sizes == NULL || plan->steps == NULL || plan->initial_tables == NULL) {
        PyErr_NoMemory();
        goto error;
    }

    while (PyDict_Next(factors_obj, &pos, &key, &value)) {
        PyObject *scope_seq = NULL;
        PyObject *table_seq = NULL;
        Py_ssize_t arity;
        Py_ssize_t table_size;
        Py_ssize_t idx;
        NativePlanScopeFactor *factor;
        NativeScaledComplex *table = NULL;

        scope_seq = PySequence_Fast(key, "Factor scopes must be tuples/sequences.");
        table_seq = PySequence_Fast(value, "Factor tables must be sequences.");
        if (scope_seq == NULL || table_seq == NULL) {
            Py_XDECREF(scope_seq);
            Py_XDECREF(table_seq);
            goto error;
        }

        arity = PySequence_Fast_GET_SIZE(scope_seq);
        if (arity < 0 || arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
            PyErr_SetString(PyExc_OverflowError, "Factor scope too wide for native treewidth plan.");
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

        factor = &states[factor_count];
        factor->arity = arity;
        factor->alive = 1;
        factor->scope = PyMem_Malloc((size_t) (arity > 0 ? arity : 1) * sizeof(Py_ssize_t));
        table = PyMem_Malloc((size_t) (table_size > 0 ? table_size : 1) * sizeof(NativeScaledComplex));
        if (factor->scope == NULL || table == NULL) {
            PyMem_Free(table);
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
                PyMem_Free(table);
                goto error;
            }
            if (var < 0 || var >= nvars) {
                PyErr_SetString(PyExc_ValueError, "factor scope contains an out-of-range variable.");
                Py_DECREF(scope_seq);
                Py_DECREF(table_seq);
                PyMem_Free(table);
                goto error;
            }
            factor->scope[idx] = var;
        }
        for (idx = 0; idx < table_size; ++idx) {
            if (parse_scaled_complex_entry(PySequence_Fast_GET_ITEM(table_seq, idx), &table[idx]) < 0) {
                Py_DECREF(scope_seq);
                Py_DECREF(table_seq);
                PyMem_Free(table);
                goto error;
            }
        }

        plan->slot_arities[factor_count] = arity;
        plan->slot_table_sizes[factor_count] = (size_t) table_size;
        plan->initial_tables[factor_count] = table;
        factor_count += 1;
        Py_DECREF(scope_seq);
        Py_DECREF(table_seq);
    }

    plan->initial_slot_count = factor_count;
    plan->slot_count = factor_count;

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
        step->low_mask = 0ULL;
        step->reduced_table_size = 0;
        step->bucket_table_indexes = NULL;

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
        step->low_mask = (var_pos == 0) ? 0ULL : ((((uint64_t) 1) << (size_t) var_pos) - 1ULL);

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
            step->reduced_table_size = 1;
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
            step->reduced_table_size = ((size_t) 1) << (size_t) new_arity;
            if (new_arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
                PyErr_SetString(PyExc_OverflowError, "Union scope too wide for native fixed-factor treewidth DP.");
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

        if (
            step->reduced_table_size > 0
            && bucket_count > 0
            && ((size_t) bucket_count) <= (SIZE_MAX / (2 * step->reduced_table_size))
        ) {
            size_t total_index_count = ((size_t) bucket_count) * (2 * step->reduced_table_size);
            if (total_index_count <= (((size_t) 1) << 20)) {
                step->bucket_table_indexes = PyMem_Malloc(
                    (total_index_count > 0 ? total_index_count : 1) * sizeof(Py_ssize_t)
                );
                if (step->bucket_table_indexes == NULL) {
                    PyErr_NoMemory();
                    goto error;
                }
                for (slot_idx = 0; slot_idx < bucket_count; ++slot_idx) {
                    size_t assignment_idx;
                    size_t bucket_offset = ((size_t) slot_idx) * (2 * step->reduced_table_size);
                    for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                        uint64_t reduced_assignment = (uint64_t) assignment_idx;
                        int fixed_value;
                        for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                            uint64_t full_assignment =
                                (reduced_assignment & step->low_mask) |
                                (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                                ((reduced_assignment & ~step->low_mask) << 1);
                            step->bucket_table_indexes[
                                bucket_offset + (((size_t) fixed_value) * step->reduced_table_size) + assignment_idx
                            ] = project_assignment_bits_u64(
                                full_assignment,
                                step->bucket_positions + step->bucket_pos_offsets[slot_idx],
                                step->bucket_arities[slot_idx]
                            );
                        }
                    }
                }
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
    PyMem_Free(bucket_slots);
    PyMem_Free(union_scope);
    PyMem_Free(union_marks);
    free_q3_free_treewidth_plan(plan);
    return -1;
}


static int evaluate_fixed_scaled_treewidth_plan(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex *out_scalar
)
{
    NativeScaledComplex **slot_tables = NULL;
    unsigned char *slot_owned = NULL;
    NativeScaledComplex scalar = scaled_complex_make(1.0, 0.0, 0);
    Py_ssize_t step_idx;
    int failed = 0;

    slot_tables = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_owned = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    if (slot_tables == NULL || slot_owned == NULL) {
        PyErr_NoMemory();
        failed = 1;
        goto cleanup;
    }
    initialize_q3_free_plan_slot_views(plan, slot_tables, slot_owned);

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        const NativeQ3FreePlanStep *step = &plan->steps[step_idx];
        if (step->bucket_count == 0) {
            scalar = renormalize_scaled_complex_if_needed(scalar.value, scalar.half_pow2_exp + 2);
            continue;
        }

        {
            size_t assignment_idx;
            NativeScaledComplex *new_table = PyMem_Malloc((step->reduced_table_size > 0 ? step->reduced_table_size : 1) * sizeof(NativeScaledComplex));
            if (new_table == NULL) {
                PyErr_NoMemory();
                failed = 1;
                break;
            }

            for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                uint64_t reduced_assignment = (uint64_t) assignment_idx;
                NativeScaledComplex total = scaled_complex_make(0.0, 0.0, 0);
                int fixed_value;
                for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                    uint64_t full_assignment =
                        (reduced_assignment & step->low_mask) |
                        (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                        ((reduced_assignment & ~step->low_mask) << 1);
                    NativeScaledComplex weight = scaled_complex_make(1.0, 0.0, 0);
                    Py_ssize_t bucket_idx;
                    for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                        Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                        Py_ssize_t table_idx = q3_free_plan_step_table_index(
                            step,
                            bucket_idx,
                            fixed_value,
                            full_assignment,
                            assignment_idx
                        );
                        if (slot_tables[slot_id] == NULL) {
                            PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native fixed-factor treewidth plan.");
                            failed = 1;
                            break;
                        }
                        weight = mul_scaled_complex(weight, slot_tables[slot_id][table_idx]);
                    }
                    if (failed) {
                        break;
                    }
                    total = add_scaled_complex(total, weight);
                }
                if (failed) {
                    break;
                }
                new_table[assignment_idx] = total;
            }
            if (failed) {
                PyMem_Free(new_table);
                break;
            }

            {
                Py_ssize_t slot_idx;
                for (slot_idx = 0; slot_idx < step->bucket_count; ++slot_idx) {
                    Py_ssize_t source_slot = step->bucket_slot_ids[slot_idx];
                    if (slot_owned[source_slot] && slot_tables[source_slot] != NULL) {
                        PyMem_Free(slot_tables[source_slot]);
                    }
                    slot_tables[source_slot] = NULL;
                    slot_owned[source_slot] = 0U;
                }
            }

            if (step->output_mode == 0) {
                scalar = mul_scaled_complex(scalar, new_table[0]);
                PyMem_Free(new_table);
            } else if (step->output_mode == 1) {
                slot_tables[step->output_slot] = new_table;
                slot_owned[step->output_slot] = 1U;
            } else {
                size_t merge_idx;
                if (ensure_owned_q3_free_plan_slot_table(plan, slot_tables, slot_owned, step->output_slot) < 0) {
                    PyMem_Free(new_table);
                    failed = 1;
                    break;
                }
                for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                    slot_tables[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables[step->output_slot][merge_idx], new_table[merge_idx]);
                }
                PyMem_Free(new_table);
            }
        }
    }

    if (!failed) {
        Py_ssize_t slot_idx;
        for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
            if (slot_tables[slot_idx] != NULL) {
                PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
                failed = 1;
                break;
            }
        }
    }

cleanup:
    free_owned_q3_free_plan_slot_views(plan, slot_tables, slot_owned);
    PyMem_Free(slot_tables);
    PyMem_Free(slot_owned);
    if (failed) {
        return -1;
    }
    *out_scalar = scalar;
    return 0;
}


static const size_t q3_free_treewidth_precomputed_bucket_index_limit = ((size_t) 1) << 20;


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
                NativeComplex phase = (
                    level == 3
                    ? omega_level3_value((int) (q2_records[q2_count].shift & 7LL))
                    : omega_dyadic_value(q2_records[q2_count].shift, mod_q1)
                );
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
        step->low_mask = 0ULL;
        step->reduced_table_size = 0;
        step->bucket_table_indexes = NULL;

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
        step->low_mask = (var_pos == 0) ? 0ULL : ((((uint64_t) 1) << (size_t) var_pos) - 1ULL);












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
            step->reduced_table_size = 1;
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
            step->reduced_table_size = ((size_t) 1) << (size_t) new_arity;
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

        if (
            step->reduced_table_size > 0
            && bucket_count > 0
            && ((size_t) bucket_count) <= (SIZE_MAX / (2 * step->reduced_table_size))
        ) {
            size_t total_index_count = ((size_t) bucket_count) * (2 * step->reduced_table_size);
            if (total_index_count <= q3_free_treewidth_precomputed_bucket_index_limit) {
                step->bucket_table_indexes = PyMem_Malloc(
                    (total_index_count > 0 ? total_index_count : 1) * sizeof(Py_ssize_t)
                );
                if (step->bucket_table_indexes == NULL) {
                    PyErr_NoMemory();
                    goto error;
                }
                for (slot_idx = 0; slot_idx < bucket_count; ++slot_idx) {
                    size_t assignment_idx;
                    size_t bucket_offset = ((size_t) slot_idx) * (2 * step->reduced_table_size);
                    for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                        uint64_t reduced_assignment = (uint64_t) assignment_idx;
                        int fixed_value;
                        for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                            uint64_t full_assignment =
                                (reduced_assignment & step->low_mask) |
                                (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                                ((reduced_assignment & ~step->low_mask) << 1);
                            step->bucket_table_indexes[
                                bucket_offset + (((size_t) fixed_value) * step->reduced_table_size) + assignment_idx
                            ] = project_assignment_bits_u64(
                                full_assignment,
                                step->bucket_positions + step->bucket_pos_offsets[slot_idx],
                                step->bucket_arities[slot_idx]
                            );
                        }
                    }
                }
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


static inline Py_ssize_t q3_free_plan_step_table_index(
    const NativeQ3FreePlanStep *step,
    Py_ssize_t bucket_idx,
    int fixed_value,
    uint64_t full_assignment,
    size_t assignment_idx
)
{
    if (step->bucket_table_indexes != NULL) {
        size_t bucket_offset = ((size_t) bucket_idx) * (2 * step->reduced_table_size);
        return step->bucket_table_indexes[
            bucket_offset + (((size_t) fixed_value) * step->reduced_table_size) + assignment_idx
        ];
    }
    return project_assignment_bits_u64(
        full_assignment,
        step->bucket_positions + step->bucket_pos_offsets[bucket_idx],
        step->bucket_arities[bucket_idx]
    );
}


static inline NativeScaledComplex level3_unary_phase_scaled(unsigned char residue)
{
    NativeComplex phase = omega_level3_value((int) (residue & 7U));
    return scaled_complex_make(phase.real, phase.imag, 0);
}


static inline NativeScaledComplex level3_unary_total_scaled(unsigned char residue)
{
    if ((residue & 7U) == 0U) {
        return scaled_complex_make(1.0, 0.0, 2);
    }
    return add_scaled_complex(
        scaled_complex_make(1.0, 0.0, 0),
        level3_unary_phase_scaled(residue)
    );
}


static int fill_q1_row_generic(
    PyObject *q1_seq,
    Py_ssize_t nvars,
    long long mod_q1,
    long long *out_row
)
{
    Py_ssize_t idx;
    for (idx = 0; idx < nvars; ++idx) {
        long long coeff = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(q1_seq, idx));
        if (coeff == -1 && PyErr_Occurred()) {
            return -1;
        }
        out_row[idx] = positive_mod_ll(coeff, mod_q1);
    }
    return 0;
}


static int fill_q1_row_level3(
    PyObject *q1_seq,
    Py_ssize_t nvars,
    unsigned char *out_row
)
{
    Py_ssize_t idx;
    for (idx = 0; idx < nvars; ++idx) {
        long long coeff = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(q1_seq, idx));
        if (coeff == -1 && PyErr_Occurred()) {
            return -1;
        }
        out_row[idx] = (unsigned char) (positive_mod_ll(coeff, 8LL) & 7LL);
    }
    return 0;
}


static void initialize_q3_free_plan_slot_views(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables,
    unsigned char *slot_owned
)
{
    Py_ssize_t slot_idx;
    for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
        slot_tables[slot_idx] = NULL;
        slot_owned[slot_idx] = 0U;
    }
    for (slot_idx = 0; slot_idx < plan->initial_slot_count; ++slot_idx) {
        slot_tables[slot_idx] = plan->initial_tables[slot_idx];
    }
}


static void free_owned_q3_free_plan_slot_views(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables,
    unsigned char *slot_owned
)
{
    Py_ssize_t slot_idx;
    if (slot_tables == NULL || slot_owned == NULL) {
        return;
    }
    for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
        if (slot_owned[slot_idx] && slot_tables[slot_idx] != NULL) {
            PyMem_Free(slot_tables[slot_idx]);
        }
    }
}


static int ensure_owned_q3_free_plan_slot_table(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables,
    unsigned char *slot_owned,
    Py_ssize_t slot_id
)
{
    NativeScaledComplex *copy;
    size_t table_size_bytes;
    if (slot_owned[slot_id]) {
        return 0;
    }
    if (slot_tables[slot_id] == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Missing merge target for native q3-free treewidth plan.");
        return -1;
    }
    table_size_bytes = plan->slot_table_sizes[slot_id] * sizeof(NativeScaledComplex);
    copy = PyMem_Malloc(table_size_bytes > 0 ? table_size_bytes : sizeof(NativeScaledComplex));
    if (copy == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    memcpy(copy, slot_tables[slot_id], table_size_bytes);
    slot_tables[slot_id] = copy;
    slot_owned[slot_id] = 1U;
    return 0;
}


static PyObject *make_q3_free_treewidth_row_result(
    NativeScaledComplex scalar,
    Py_ssize_t max_scope
)
{
    PyObject *total_obj = NULL;
    PyObject *exp_obj = NULL;
    PyObject *width_obj = NULL;
    PyObject *row_result = NULL;

    total_obj = PyComplex_FromDoubles(scalar.value.real, scalar.value.imag);
    exp_obj = PyLong_FromLongLong(scalar.half_pow2_exp);
    width_obj = PyLong_FromSsize_t(max_scope);
    if (total_obj == NULL || exp_obj == NULL || width_obj == NULL) {
        Py_XDECREF(total_obj);
        Py_XDECREF(exp_obj);
        Py_XDECREF(width_obj);
        return NULL;
    }
    row_result = PyTuple_Pack(3, total_obj, exp_obj, width_obj);
    Py_DECREF(total_obj);
    Py_DECREF(exp_obj);
    Py_DECREF(width_obj);
    return row_result;
}


static int q3_free_pair_step_inputs_match(
    const NativeQ3FreePlanStep *step,
    NativeScaledComplex **slot_tables0,
    NativeScaledComplex **slot_tables1
)
{
    Py_ssize_t bucket_idx;
    for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
        Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
        if (slot_tables0[slot_id] != slot_tables1[slot_id]) {
            return 0;
        }
    }
    return 1;
}


static void clear_q3_free_pair_bucket_slots(
    const NativeQ3FreePlanStep *step,
    NativeScaledComplex **slot_tables0,
    NativeScaledComplex **slot_tables1,
    unsigned char *slot_owned0,
    unsigned char *slot_owned1
)
{
    Py_ssize_t slot_idx;
    for (slot_idx = 0; slot_idx < step->bucket_count; ++slot_idx) {
        Py_ssize_t source_slot = step->bucket_slot_ids[slot_idx];
        if (slot_tables0[source_slot] == slot_tables1[source_slot]) {
            if (slot_owned0[source_slot] && slot_tables0[source_slot] != NULL) {
                PyMem_Free(slot_tables0[source_slot]);
            } else if (slot_owned1[source_slot] && slot_tables1[source_slot] != NULL) {
                PyMem_Free(slot_tables1[source_slot]);
            }
        } else {
            if (slot_owned0[source_slot] && slot_tables0[source_slot] != NULL) {
                PyMem_Free(slot_tables0[source_slot]);
            }
            if (slot_owned1[source_slot] && slot_tables1[source_slot] != NULL) {
                PyMem_Free(slot_tables1[source_slot]);
            }
        }
        slot_tables0[source_slot] = NULL;
        slot_tables1[source_slot] = NULL;
        slot_owned0[source_slot] = 0U;
        slot_owned1[source_slot] = 0U;
    }
}


static int evaluate_q3_free_treewidth_plan_row_generic(
    const NativeQ3FreeTreewidthPlan *plan,
    const long long *q1_row,
    NativeScaledComplex *out_scalar
)
{
    NativeScaledComplex **slot_tables = NULL;
    unsigned char *slot_owned = NULL;
    NativeScaledComplex scalar = scaled_complex_make(1.0, 0.0, 0);
    Py_ssize_t step_idx;
    int failed = 0;

    slot_tables = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_owned = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    if (slot_tables == NULL || slot_owned == NULL) {
        PyErr_NoMemory();
        failed = 1;
        goto cleanup;
    }
    initialize_q3_free_plan_slot_views(plan, slot_tables, slot_owned);

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        const NativeQ3FreePlanStep *step = &plan->steps[step_idx];
        long long unary_coeff = q1_row[step->var];

        if (step->bucket_count == 0) {
            if (unary_coeff == 0) {
                scalar = renormalize_scaled_complex_if_needed(scalar.value, scalar.half_pow2_exp + 2);
            } else {
                NativeComplex phase = omega_dyadic_value(unary_coeff, plan->mod_q1);
                scalar = mul_scaled_complex(scalar, add_scaled_complex(scaled_complex_make(1.0, 0.0, 0), scaled_complex_make(phase.real, phase.imag, 0)));
            }
            continue;
        }

        {
            size_t assignment_idx;
            NativeScaledComplex *new_table = PyMem_Malloc((step->reduced_table_size > 0 ? step->reduced_table_size : 1) * sizeof(NativeScaledComplex));
            NativeScaledComplex unary_phase = scaled_complex_make(1.0, 0.0, 0);
            int has_unary = unary_coeff != 0;
            if (new_table == NULL) {
                PyErr_NoMemory();
                failed = 1;
                break;
            }
            if (has_unary) {
                NativeComplex phase = omega_dyadic_value(unary_coeff, plan->mod_q1);
                unary_phase = scaled_complex_make(phase.real, phase.imag, 0);
            }

            for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                uint64_t reduced_assignment = (uint64_t) assignment_idx;
                NativeScaledComplex total = scaled_complex_make(0.0, 0.0, 0);
                int fixed_value;
                for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                    uint64_t full_assignment =
                        (reduced_assignment & step->low_mask) |
                        (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                        ((reduced_assignment & ~step->low_mask) << 1);
                    NativeScaledComplex weight = scaled_complex_make(1.0, 0.0, 0);
                    Py_ssize_t bucket_idx;
                    if (has_unary && fixed_value) {
                        weight = unary_phase;
                    }
                    for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                        Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                        Py_ssize_t table_idx = q3_free_plan_step_table_index(
                            step,
                            bucket_idx,
                            fixed_value,
                            full_assignment,
                            assignment_idx
                        );
                        if (slot_tables[slot_id] == NULL) {
                            PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native q3-free treewidth plan.");
                            failed = 1;
                            break;
                        }
                        weight = mul_scaled_complex(weight, slot_tables[slot_id][table_idx]);
                    }
                    if (failed) {
                        break;
                    }
                    total = add_scaled_complex(total, weight);
                }
                if (failed) {
                    break;
                }
                new_table[assignment_idx] = total;
            }
            if (failed) {
                PyMem_Free(new_table);
                break;
            }

            {
                Py_ssize_t slot_idx;
                for (slot_idx = 0; slot_idx < step->bucket_count; ++slot_idx) {
                    Py_ssize_t source_slot = step->bucket_slot_ids[slot_idx];
                    if (slot_owned[source_slot] && slot_tables[source_slot] != NULL) {
                        PyMem_Free(slot_tables[source_slot]);
                    }
                    slot_tables[source_slot] = NULL;
                    slot_owned[source_slot] = 0U;
                }
            }

            if (step->output_mode == 0) {
                scalar = mul_scaled_complex(scalar, new_table[0]);
                PyMem_Free(new_table);
            } else if (step->output_mode == 1) {
                slot_tables[step->output_slot] = new_table;
                slot_owned[step->output_slot] = 1U;
            } else {
                size_t merge_idx;
                if (ensure_owned_q3_free_plan_slot_table(plan, slot_tables, slot_owned, step->output_slot) < 0) {
                    PyMem_Free(new_table);
                    failed = 1;
                    break;
                }
                for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                    slot_tables[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables[step->output_slot][merge_idx], new_table[merge_idx]);
                }
                PyMem_Free(new_table);
            }
        }
    }

    if (!failed) {
        Py_ssize_t slot_idx;
        for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
            if (slot_tables[slot_idx] != NULL) {
                PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
                failed = 1;
                break;
            }
        }
    }

cleanup:
    free_owned_q3_free_plan_slot_views(plan, slot_tables, slot_owned);
    PyMem_Free(slot_tables);
    PyMem_Free(slot_owned);
    if (failed) {
        return -1;
    }
    *out_scalar = scalar;
    return 0;
}


static int evaluate_q3_free_treewidth_plan_row_level3(
    const NativeQ3FreeTreewidthPlan *plan,
    const unsigned char *q1_row,
    NativeScaledComplex *out_scalar
)
{
    NativeScaledComplex **slot_tables = NULL;
    unsigned char *slot_owned = NULL;
    NativeScaledComplex scalar = scaled_complex_make(1.0, 0.0, 0);
    Py_ssize_t step_idx;
    int failed = 0;

    slot_tables = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_owned = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    if (slot_tables == NULL || slot_owned == NULL) {
        PyErr_NoMemory();
        failed = 1;
        goto cleanup;
    }
    initialize_q3_free_plan_slot_views(plan, slot_tables, slot_owned);

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        const NativeQ3FreePlanStep *step = &plan->steps[step_idx];
        unsigned char unary_coeff = (unsigned char) (q1_row[step->var] & 7U);

        if (step->bucket_count == 0) {
            scalar = mul_scaled_complex(scalar, level3_unary_total_scaled(unary_coeff));
            continue;
        }

        {
            size_t assignment_idx;
            NativeScaledComplex *new_table = PyMem_Malloc((step->reduced_table_size > 0 ? step->reduced_table_size : 1) * sizeof(NativeScaledComplex));
            NativeScaledComplex unary_phase = scaled_complex_make(1.0, 0.0, 0);
            int has_unary = unary_coeff != 0U;
            if (new_table == NULL) {
                PyErr_NoMemory();
                failed = 1;
                break;
            }
            if (has_unary) {
                unary_phase = level3_unary_phase_scaled(unary_coeff);
            }

            for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                uint64_t reduced_assignment = (uint64_t) assignment_idx;
                NativeScaledComplex total = scaled_complex_make(0.0, 0.0, 0);
                int fixed_value;
                for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                    uint64_t full_assignment =
                        (reduced_assignment & step->low_mask) |
                        (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                        ((reduced_assignment & ~step->low_mask) << 1);
                    NativeScaledComplex weight = scaled_complex_make(1.0, 0.0, 0);
                    Py_ssize_t bucket_idx;
                    if (has_unary && fixed_value) {
                        weight = unary_phase;
                    }
                    for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                        Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                        Py_ssize_t table_idx = q3_free_plan_step_table_index(
                            step,
                            bucket_idx,
                            fixed_value,
                            full_assignment,
                            assignment_idx
                        );
                        if (slot_tables[slot_id] == NULL) {
                            PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native q3-free treewidth plan.");
                            failed = 1;
                            break;
                        }
                        weight = mul_scaled_complex(weight, slot_tables[slot_id][table_idx]);
                    }
                    if (failed) {
                        break;
                    }
                    total = add_scaled_complex(total, weight);
                }
                if (failed) {
                    break;
                }
                new_table[assignment_idx] = total;
            }
            if (failed) {
                PyMem_Free(new_table);
                break;
            }

            {
                Py_ssize_t slot_idx;
                for (slot_idx = 0; slot_idx < step->bucket_count; ++slot_idx) {
                    Py_ssize_t source_slot = step->bucket_slot_ids[slot_idx];
                    if (slot_owned[source_slot] && slot_tables[source_slot] != NULL) {
                        PyMem_Free(slot_tables[source_slot]);
                    }
                    slot_tables[source_slot] = NULL;
                    slot_owned[source_slot] = 0U;
                }
            }

            if (step->output_mode == 0) {
                scalar = mul_scaled_complex(scalar, new_table[0]);
                PyMem_Free(new_table);
            } else if (step->output_mode == 1) {
                slot_tables[step->output_slot] = new_table;
                slot_owned[step->output_slot] = 1U;
            } else {
                size_t merge_idx;
                if (ensure_owned_q3_free_plan_slot_table(plan, slot_tables, slot_owned, step->output_slot) < 0) {
                    PyMem_Free(new_table);
                    failed = 1;
                    break;
                }
                for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                    slot_tables[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables[step->output_slot][merge_idx], new_table[merge_idx]);
                }
                PyMem_Free(new_table);
            }
        }
    }

    if (!failed) {
        Py_ssize_t slot_idx;
        for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
            if (slot_tables[slot_idx] != NULL) {
                PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
                failed = 1;
                break;
            }
        }
    }

cleanup:
    free_owned_q3_free_plan_slot_views(plan, slot_tables, slot_owned);
    PyMem_Free(slot_tables);
    PyMem_Free(slot_owned);
    if (failed) {
        return -1;
    }
    *out_scalar = scalar;
    return 0;
}


static int evaluate_q3_free_treewidth_plan_pair_generic(
    const NativeQ3FreeTreewidthPlan *plan,
    const long long *q1_row0,
    const long long *q1_row1,
    NativeScaledComplex *out_scalar0,
    NativeScaledComplex *out_scalar1
)
{
    NativeScaledComplex **slot_tables0 = NULL;
    NativeScaledComplex **slot_tables1 = NULL;
    unsigned char *slot_owned0 = NULL;
    unsigned char *slot_owned1 = NULL;
    NativeScaledComplex scalar0 = scaled_complex_make(1.0, 0.0, 0);
    NativeScaledComplex scalar1 = scaled_complex_make(1.0, 0.0, 0);
    Py_ssize_t step_idx;
    int failed = 0;

    slot_tables0 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_tables1 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_owned0 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    slot_owned1 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    if (slot_tables0 == NULL || slot_tables1 == NULL || slot_owned0 == NULL || slot_owned1 == NULL) {
        PyErr_NoMemory();
        failed = 1;
        goto cleanup;
    }
    initialize_q3_free_plan_slot_views(plan, slot_tables0, slot_owned0);
    initialize_q3_free_plan_slot_views(plan, slot_tables1, slot_owned1);

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        const NativeQ3FreePlanStep *step = &plan->steps[step_idx];
        long long unary_coeff0 = q1_row0[step->var];
        long long unary_coeff1 = q1_row1[step->var];

        if (step->bucket_count == 0) {
            if (unary_coeff0 == 0) {
                scalar0 = renormalize_scaled_complex_if_needed(scalar0.value, scalar0.half_pow2_exp + 2);
            } else {
                NativeComplex phase0 = omega_dyadic_value(unary_coeff0, plan->mod_q1);
                scalar0 = mul_scaled_complex(scalar0, add_scaled_complex(scaled_complex_make(1.0, 0.0, 0), scaled_complex_make(phase0.real, phase0.imag, 0)));
            }
            if (unary_coeff1 == 0) {
                scalar1 = renormalize_scaled_complex_if_needed(scalar1.value, scalar1.half_pow2_exp + 2);
            } else {
                NativeComplex phase1 = omega_dyadic_value(unary_coeff1, plan->mod_q1);
                scalar1 = mul_scaled_complex(scalar1, add_scaled_complex(scaled_complex_make(1.0, 0.0, 0), scaled_complex_make(phase1.real, phase1.imag, 0)));
            }
            continue;
        }

        {
            size_t assignment_idx;
            NativeScaledComplex *new_table0 = PyMem_Malloc((step->reduced_table_size > 0 ? step->reduced_table_size : 1) * sizeof(NativeScaledComplex));
            NativeScaledComplex *new_table1 = NULL;
            NativeScaledComplex unary_phase0 = scaled_complex_make(1.0, 0.0, 0);
            NativeScaledComplex unary_phase1 = scaled_complex_make(1.0, 0.0, 0);
            int has_unary0 = unary_coeff0 != 0;
            int has_unary1 = unary_coeff1 != 0;
            int identical_step = (
                unary_coeff0 == unary_coeff1
                && q3_free_pair_step_inputs_match(step, slot_tables0, slot_tables1)
            );
            if (!identical_step) {
                new_table1 = PyMem_Malloc((step->reduced_table_size > 0 ? step->reduced_table_size : 1) * sizeof(NativeScaledComplex));
            }
            if (new_table0 == NULL || (!identical_step && new_table1 == NULL)) {
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
                PyErr_NoMemory();
                failed = 1;
                break;
            }
            if (has_unary0) {
                NativeComplex phase0 = omega_dyadic_value(unary_coeff0, plan->mod_q1);
                unary_phase0 = scaled_complex_make(phase0.real, phase0.imag, 0);
            }
            if (has_unary1) {
                NativeComplex phase1 = omega_dyadic_value(unary_coeff1, plan->mod_q1);
                unary_phase1 = scaled_complex_make(phase1.real, phase1.imag, 0);
            }

            if (identical_step) {
                for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                    uint64_t reduced_assignment = (uint64_t) assignment_idx;
                    NativeScaledComplex total = scaled_complex_make(0.0, 0.0, 0);
                    int fixed_value;
                    for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                        uint64_t full_assignment =
                            (reduced_assignment & step->low_mask) |
                            (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                            ((reduced_assignment & ~step->low_mask) << 1);
                        NativeScaledComplex weight = scaled_complex_make(1.0, 0.0, 0);
                        Py_ssize_t bucket_idx;
                        if (has_unary0 && fixed_value) {
                            weight = unary_phase0;
                        }
                        for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                            Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                            Py_ssize_t table_idx = q3_free_plan_step_table_index(
                                step,
                                bucket_idx,
                                fixed_value,
                                full_assignment,
                                assignment_idx
                            );
                            if (slot_tables0[slot_id] == NULL) {
                                PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native q3-free treewidth plan.");
                                failed = 1;
                                break;
                            }
                            weight = mul_scaled_complex(weight, slot_tables0[slot_id][table_idx]);
                        }
                        if (failed) {
                            break;
                        }
                        total = add_scaled_complex(total, weight);
                    }
                    if (failed) {
                        break;
                    }
                    new_table0[assignment_idx] = total;
                }
            } else {
                for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                    uint64_t reduced_assignment = (uint64_t) assignment_idx;
                    NativeScaledComplex total0 = scaled_complex_make(0.0, 0.0, 0);
                    NativeScaledComplex total1 = scaled_complex_make(0.0, 0.0, 0);
                    int fixed_value;
                    for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                        uint64_t full_assignment =
                            (reduced_assignment & step->low_mask) |
                            (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                            ((reduced_assignment & ~step->low_mask) << 1);
                        NativeScaledComplex weight0 = scaled_complex_make(1.0, 0.0, 0);
                        NativeScaledComplex weight1 = scaled_complex_make(1.0, 0.0, 0);
                        Py_ssize_t bucket_idx;
                        if (has_unary0 && fixed_value) {
                            weight0 = unary_phase0;
                        }
                        if (has_unary1 && fixed_value) {
                            weight1 = unary_phase1;
                        }
                        for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                            Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                            Py_ssize_t table_idx = q3_free_plan_step_table_index(
                                step,
                                bucket_idx,
                                fixed_value,
                                full_assignment,
                                assignment_idx
                            );
                            if (slot_tables0[slot_id] == NULL || slot_tables1[slot_id] == NULL) {
                                PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native q3-free treewidth plan.");
                                failed = 1;
                                break;
                            }
                            weight0 = mul_scaled_complex(weight0, slot_tables0[slot_id][table_idx]);
                            weight1 = mul_scaled_complex(weight1, slot_tables1[slot_id][table_idx]);
                        }
                        if (failed) {
                            break;
                        }
                        total0 = add_scaled_complex(total0, weight0);
                        total1 = add_scaled_complex(total1, weight1);
                    }
                    if (failed) {
                        break;
                    }
                    new_table0[assignment_idx] = total0;
                    new_table1[assignment_idx] = total1;
                }
            }
            if (failed) {
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
                break;
            }

            clear_q3_free_pair_bucket_slots(step, slot_tables0, slot_tables1, slot_owned0, slot_owned1);

            if (step->output_mode == 0) {
                scalar0 = mul_scaled_complex(scalar0, new_table0[0]);
                scalar1 = mul_scaled_complex(scalar1, identical_step ? new_table0[0] : new_table1[0]);
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
            } else if (step->output_mode == 1) {
                slot_tables0[step->output_slot] = new_table0;
                slot_tables1[step->output_slot] = identical_step ? new_table0 : new_table1;
                slot_owned0[step->output_slot] = 1U;
                slot_owned1[step->output_slot] = identical_step ? 0U : 1U;
            } else {
                size_t merge_idx;
                if (ensure_owned_q3_free_plan_slot_table(plan, slot_tables0, slot_owned0, step->output_slot) < 0) {
                    PyMem_Free(new_table0);
                    PyMem_Free(new_table1);
                    failed = 1;
                    break;
                }
                if (
                    identical_step
                    && slot_tables0[step->output_slot] == slot_tables1[step->output_slot]
                ) {
                    for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                        slot_tables0[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables0[step->output_slot][merge_idx], new_table0[merge_idx]);
                    }
                    slot_tables1[step->output_slot] = slot_tables0[step->output_slot];
                    slot_owned1[step->output_slot] = 0U;
                } else {
                    if (ensure_owned_q3_free_plan_slot_table(plan, slot_tables1, slot_owned1, step->output_slot) < 0) {
                        PyMem_Free(new_table0);
                        PyMem_Free(new_table1);
                        failed = 1;
                        break;
                    }
                    for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                        slot_tables0[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables0[step->output_slot][merge_idx], new_table0[merge_idx]);
                        slot_tables1[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables1[step->output_slot][merge_idx], identical_step ? new_table0[merge_idx] : new_table1[merge_idx]);
                    }
                }
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
            }
        }
    }

    if (!failed) {
        Py_ssize_t slot_idx;
        for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
            if (slot_tables0[slot_idx] != NULL || slot_tables1[slot_idx] != NULL) {
                PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
                failed = 1;
                break;
            }
        }
    }

cleanup:
    free_owned_q3_free_plan_slot_views(plan, slot_tables0, slot_owned0);
    free_owned_q3_free_plan_slot_views(plan, slot_tables1, slot_owned1);
    PyMem_Free(slot_tables0);
    PyMem_Free(slot_tables1);
    PyMem_Free(slot_owned0);
    PyMem_Free(slot_owned1);
    if (failed) {
        return -1;
    }
    *out_scalar0 = scalar0;
    *out_scalar1 = scalar1;
    return 0;
}


static int evaluate_q3_free_treewidth_plan_pair_level3(
    const NativeQ3FreeTreewidthPlan *plan,
    const unsigned char *q1_row0,
    const unsigned char *q1_row1,
    NativeScaledComplex *out_scalar0,
    NativeScaledComplex *out_scalar1
)
{
    NativeScaledComplex **slot_tables0 = NULL;
    NativeScaledComplex **slot_tables1 = NULL;
    unsigned char *slot_owned0 = NULL;
    unsigned char *slot_owned1 = NULL;
    NativeScaledComplex scalar0 = scaled_complex_make(1.0, 0.0, 0);
    NativeScaledComplex scalar1 = scaled_complex_make(1.0, 0.0, 0);
    Py_ssize_t step_idx;
    int failed = 0;

    slot_tables0 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_tables1 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_owned0 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    slot_owned1 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    if (slot_tables0 == NULL || slot_tables1 == NULL || slot_owned0 == NULL || slot_owned1 == NULL) {
        PyErr_NoMemory();
        failed = 1;
        goto cleanup;
    }
    initialize_q3_free_plan_slot_views(plan, slot_tables0, slot_owned0);
    initialize_q3_free_plan_slot_views(plan, slot_tables1, slot_owned1);

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        const NativeQ3FreePlanStep *step = &plan->steps[step_idx];
        unsigned char unary_coeff0 = (unsigned char) (q1_row0[step->var] & 7U);
        unsigned char unary_coeff1 = (unsigned char) (q1_row1[step->var] & 7U);

        if (step->bucket_count == 0) {
            scalar0 = mul_scaled_complex(scalar0, level3_unary_total_scaled(unary_coeff0));
            scalar1 = mul_scaled_complex(scalar1, level3_unary_total_scaled(unary_coeff1));
            continue;
        }

        {
            size_t assignment_idx;
            NativeScaledComplex *new_table0 = PyMem_Malloc((step->reduced_table_size > 0 ? step->reduced_table_size : 1) * sizeof(NativeScaledComplex));
            NativeScaledComplex *new_table1 = NULL;
            NativeScaledComplex unary_phase0 = scaled_complex_make(1.0, 0.0, 0);
            NativeScaledComplex unary_phase1 = scaled_complex_make(1.0, 0.0, 0);
            int has_unary0 = unary_coeff0 != 0U;
            int has_unary1 = unary_coeff1 != 0U;
            int identical_step = (
                unary_coeff0 == unary_coeff1
                && q3_free_pair_step_inputs_match(step, slot_tables0, slot_tables1)
            );
            if (!identical_step) {
                new_table1 = PyMem_Malloc((step->reduced_table_size > 0 ? step->reduced_table_size : 1) * sizeof(NativeScaledComplex));
            }
            if (new_table0 == NULL || (!identical_step && new_table1 == NULL)) {
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
                PyErr_NoMemory();
                failed = 1;
                break;
            }
            if (has_unary0) {
                unary_phase0 = level3_unary_phase_scaled(unary_coeff0);
            }
            if (has_unary1) {
                unary_phase1 = level3_unary_phase_scaled(unary_coeff1);
            }

            if (identical_step) {
                for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                    uint64_t reduced_assignment = (uint64_t) assignment_idx;
                    NativeScaledComplex total = scaled_complex_make(0.0, 0.0, 0);
                    int fixed_value;
                    for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                        uint64_t full_assignment =
                            (reduced_assignment & step->low_mask) |
                            (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                            ((reduced_assignment & ~step->low_mask) << 1);
                        NativeScaledComplex weight = scaled_complex_make(1.0, 0.0, 0);
                        Py_ssize_t bucket_idx;
                        if (has_unary0 && fixed_value) {
                            weight = unary_phase0;
                        }
                        for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                            Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                            Py_ssize_t table_idx = q3_free_plan_step_table_index(
                                step,
                                bucket_idx,
                                fixed_value,
                                full_assignment,
                                assignment_idx
                            );
                            if (slot_tables0[slot_id] == NULL) {
                                PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native q3-free treewidth plan.");
                                failed = 1;
                                break;
                            }
                            weight = mul_scaled_complex(weight, slot_tables0[slot_id][table_idx]);
                        }
                        if (failed) {
                            break;
                        }
                        total = add_scaled_complex(total, weight);
                    }
                    if (failed) {
                        break;
                    }
                    new_table0[assignment_idx] = total;
                }
            } else {
                for (assignment_idx = 0; assignment_idx < step->reduced_table_size; ++assignment_idx) {
                    uint64_t reduced_assignment = (uint64_t) assignment_idx;
                    NativeScaledComplex total0 = scaled_complex_make(0.0, 0.0, 0);
                    NativeScaledComplex total1 = scaled_complex_make(0.0, 0.0, 0);
                    int fixed_value;
                    for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                        uint64_t full_assignment =
                            (reduced_assignment & step->low_mask) |
                            (((uint64_t) fixed_value) << (size_t) step->var_pos) |
                            ((reduced_assignment & ~step->low_mask) << 1);
                        NativeScaledComplex weight0 = scaled_complex_make(1.0, 0.0, 0);
                        NativeScaledComplex weight1 = scaled_complex_make(1.0, 0.0, 0);
                        Py_ssize_t bucket_idx;
                        if (has_unary0 && fixed_value) {
                            weight0 = unary_phase0;
                        }
                        if (has_unary1 && fixed_value) {
                            weight1 = unary_phase1;
                        }
                        for (bucket_idx = 0; bucket_idx < step->bucket_count; ++bucket_idx) {
                            Py_ssize_t slot_id = step->bucket_slot_ids[bucket_idx];
                            Py_ssize_t table_idx = q3_free_plan_step_table_index(
                                step,
                                bucket_idx,
                                fixed_value,
                                full_assignment,
                                assignment_idx
                            );
                            if (slot_tables0[slot_id] == NULL || slot_tables1[slot_id] == NULL) {
                                PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native q3-free treewidth plan.");
                                failed = 1;
                                break;
                            }
                            weight0 = mul_scaled_complex(weight0, slot_tables0[slot_id][table_idx]);
                            weight1 = mul_scaled_complex(weight1, slot_tables1[slot_id][table_idx]);
                        }
                        if (failed) {
                            break;
                        }
                        total0 = add_scaled_complex(total0, weight0);
                        total1 = add_scaled_complex(total1, weight1);
                    }
                    if (failed) {
                        break;
                    }
                    new_table0[assignment_idx] = total0;
                    new_table1[assignment_idx] = total1;
                }
            }
            if (failed) {
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
                break;
            }

            clear_q3_free_pair_bucket_slots(step, slot_tables0, slot_tables1, slot_owned0, slot_owned1);

            if (step->output_mode == 0) {
                scalar0 = mul_scaled_complex(scalar0, new_table0[0]);
                scalar1 = mul_scaled_complex(scalar1, identical_step ? new_table0[0] : new_table1[0]);
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
            } else if (step->output_mode == 1) {
                slot_tables0[step->output_slot] = new_table0;
                slot_tables1[step->output_slot] = identical_step ? new_table0 : new_table1;
                slot_owned0[step->output_slot] = 1U;
                slot_owned1[step->output_slot] = identical_step ? 0U : 1U;
            } else {
                size_t merge_idx;
                if (ensure_owned_q3_free_plan_slot_table(plan, slot_tables0, slot_owned0, step->output_slot) < 0) {
                    PyMem_Free(new_table0);
                    PyMem_Free(new_table1);
                    failed = 1;
                    break;
                }
                if (
                    identical_step
                    && slot_tables0[step->output_slot] == slot_tables1[step->output_slot]
                ) {
                    for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                        slot_tables0[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables0[step->output_slot][merge_idx], new_table0[merge_idx]);
                    }
                    slot_tables1[step->output_slot] = slot_tables0[step->output_slot];
                    slot_owned1[step->output_slot] = 0U;
                } else {
                    if (ensure_owned_q3_free_plan_slot_table(plan, slot_tables1, slot_owned1, step->output_slot) < 0) {
                        PyMem_Free(new_table0);
                        PyMem_Free(new_table1);
                        failed = 1;
                        break;
                    }
                    for (merge_idx = 0; merge_idx < step->reduced_table_size; ++merge_idx) {
                        slot_tables0[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables0[step->output_slot][merge_idx], new_table0[merge_idx]);
                        slot_tables1[step->output_slot][merge_idx] = mul_scaled_complex(slot_tables1[step->output_slot][merge_idx], identical_step ? new_table0[merge_idx] : new_table1[merge_idx]);
                    }
                }
                PyMem_Free(new_table0);
                PyMem_Free(new_table1);
            }
        }
    }

    if (!failed) {
        Py_ssize_t slot_idx;
        for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
            if (slot_tables0[slot_idx] != NULL || slot_tables1[slot_idx] != NULL) {
                PyErr_SetString(PyExc_RuntimeError, "All factors should be eliminated by the supplied order.");
                failed = 1;
                break;
            }
        }
    }

cleanup:
    free_owned_q3_free_plan_slot_views(plan, slot_tables0, slot_owned0);
    free_owned_q3_free_plan_slot_views(plan, slot_tables1, slot_owned1);
    PyMem_Free(slot_tables0);
    PyMem_Free(slot_tables1);
    PyMem_Free(slot_owned0);
    PyMem_Free(slot_owned1);
    if (failed) {
        return -1;
    }
    *out_scalar0 = scalar0;
    *out_scalar1 = scalar1;
    return 0;
}


static int validate_q3_free_q1_batch_buffer(
    PyObject *q1_batch_obj,
    Py_ssize_t expected_nvars,
    Py_buffer *view
)
{
    if (PyObject_GetBuffer(q1_batch_obj, view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return -1;
    }
    if (view->ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "q1_batch buffer must be 2-dimensional.");
        PyBuffer_Release(view);
        return -1;
    }
    if (view->shape == NULL || view->strides == NULL) {
        PyErr_SetString(PyExc_ValueError, "q1_batch buffer is missing shape/stride metadata.");
        PyBuffer_Release(view);
        return -1;
    }
    if (view->itemsize != (Py_ssize_t) sizeof(long long)) {
        PyErr_SetString(PyExc_ValueError, "q1_batch buffer must use 64-bit signed integers.");
        PyBuffer_Release(view);
        return -1;
    }
    if (view->shape[1] != expected_nvars) {
        PyErr_SetString(PyExc_ValueError, "Each q1 row must have length nvars.");
        PyBuffer_Release(view);
        return -1;
    }
    if (view->strides[1] != (Py_ssize_t) sizeof(long long)) {
        PyErr_SetString(PyExc_ValueError, "q1_batch rows must be contiguous int64 arrays.");
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}


static PyObject *evaluate_q3_free_treewidth_plan_batch_buffer(
    NativeQ3FreeTreewidthPlan *plan,
    PyObject *q1_batch_obj
)
{
    Py_buffer view;
    PyObject *results = NULL;
    Py_ssize_t batch_size;
    Py_ssize_t row_stride_elems;
    Py_ssize_t row_idx;

    memset(&view, 0, sizeof(Py_buffer));
    if (validate_q3_free_q1_batch_buffer(q1_batch_obj, plan->nvars, &view) < 0) {
        return NULL;
    }

    batch_size = view.shape[0];
    row_stride_elems = view.strides[0] / (Py_ssize_t) sizeof(long long);
    results = PyList_New(batch_size);
    if (results == NULL) {
        PyBuffer_Release(&view);
        return NULL;
    }

    if (batch_size == 1 || batch_size == 2) {
        const long long *base = (const long long *) view.buf;
        PyObject *row_result0 = NULL;
        PyObject *row_result1 = NULL;
        NativeScaledComplex scalar0;
        NativeScaledComplex scalar1;

        if (plan->level == 3) {
            unsigned char *q1_row0 = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(unsigned char));
            unsigned char *q1_row1 = NULL;
            Py_ssize_t idx;
            if (q1_row0 == NULL) {
                PyBuffer_Release(&view);
                Py_DECREF(results);
                PyErr_NoMemory();
                return NULL;
            }
            for (idx = 0; idx < plan->nvars; ++idx) {
                q1_row0[idx] = (unsigned char) (base[idx] & 7LL);
            }
            if (batch_size == 1) {
                if (evaluate_q3_free_treewidth_plan_row_level3(plan, q1_row0, &scalar0) < 0) {
                    PyMem_Free(q1_row0);
                    PyBuffer_Release(&view);
                    Py_DECREF(results);
                    return NULL;
                }
            } else {
                const long long *row1_base = base + row_stride_elems;
                q1_row1 = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(unsigned char));
                if (q1_row1 == NULL) {
                    PyMem_Free(q1_row0);
                    PyBuffer_Release(&view);
                    Py_DECREF(results);
                    PyErr_NoMemory();
                    return NULL;
                }
                for (idx = 0; idx < plan->nvars; ++idx) {
                    q1_row1[idx] = (unsigned char) (row1_base[idx] & 7LL);
                }
                if (evaluate_q3_free_treewidth_plan_pair_level3(plan, q1_row0, q1_row1, &scalar0, &scalar1) < 0) {
                    PyMem_Free(q1_row0);
                    PyMem_Free(q1_row1);
                    PyBuffer_Release(&view);
                    Py_DECREF(results);
                    return NULL;
                }
                PyMem_Free(q1_row1);
            }
            PyMem_Free(q1_row0);
        } else {
            const long long *q1_row0 = (const long long *) view.buf;
            if (batch_size == 1) {
                if (evaluate_q3_free_treewidth_plan_row_generic(plan, q1_row0, &scalar0) < 0) {
                    PyBuffer_Release(&view);
                    Py_DECREF(results);
                    return NULL;
                }
            } else {
                const long long *q1_row1 = q1_row0 + row_stride_elems;
                if (evaluate_q3_free_treewidth_plan_pair_generic(plan, q1_row0, q1_row1, &scalar0, &scalar1) < 0) {
                    PyBuffer_Release(&view);
                    Py_DECREF(results);
                    return NULL;
                }
            }
        }

        row_result0 = make_q3_free_treewidth_row_result(scalar0, plan->max_scope);
        if (batch_size == 2) {
            row_result1 = make_q3_free_treewidth_row_result(scalar1, plan->max_scope);
        }
        if (row_result0 == NULL || (batch_size == 2 && row_result1 == NULL)) {
            Py_XDECREF(row_result0);
            Py_XDECREF(row_result1);
            PyBuffer_Release(&view);
            Py_DECREF(results);
            return NULL;
        }
        PyList_SET_ITEM(results, 0, row_result0);
        if (batch_size == 2) {
            PyList_SET_ITEM(results, 1, row_result1);
        }
        PyBuffer_Release(&view);
        return results;
    }

    for (row_idx = 0; row_idx < batch_size; ++row_idx) {
        const long long *row_ptr = ((const long long *) view.buf) + (row_idx * row_stride_elems);
        PyObject *row_result = NULL;
        NativeScaledComplex scalar;
        if (plan->level == 3) {
            unsigned char *q1_row = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(unsigned char));
            Py_ssize_t idx;
            if (q1_row == NULL) {
                PyBuffer_Release(&view);
                Py_DECREF(results);
                PyErr_NoMemory();
                return NULL;
            }
            for (idx = 0; idx < plan->nvars; ++idx) {
                q1_row[idx] = (unsigned char) (row_ptr[idx] & 7LL);
            }
            if (evaluate_q3_free_treewidth_plan_row_level3(plan, q1_row, &scalar) < 0) {
                PyMem_Free(q1_row);
                PyBuffer_Release(&view);
                Py_DECREF(results);
                return NULL;
            }
            PyMem_Free(q1_row);
        } else {
            if (evaluate_q3_free_treewidth_plan_row_generic(plan, row_ptr, &scalar) < 0) {
                PyBuffer_Release(&view);
                Py_DECREF(results);
                return NULL;
            }
        }
        row_result = make_q3_free_treewidth_row_result(scalar, plan->max_scope);
        if (row_result == NULL) {
            PyBuffer_Release(&view);
            Py_DECREF(results);
            return NULL;
        }
        PyList_SET_ITEM(results, row_idx, row_result);
    }

    PyBuffer_Release(&view);
    return results;
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

    if (batch_size == 1 || batch_size == 2) {
        PyObject *q1_seq0 = PySequence_Fast(PySequence_Fast_GET_ITEM(batch_seq, 0), "Each q1 row must be a sequence.");
        PyObject *q1_seq1 = NULL;
        PyObject *row_result0 = NULL;
        PyObject *row_result1 = NULL;
        if (q1_seq0 == NULL) {
            goto error;
        }
        if (PySequence_Fast_GET_SIZE(q1_seq0) != plan->nvars) {
            PyErr_SetString(PyExc_ValueError, "Each q1 row must have length nvars.");
            Py_DECREF(q1_seq0);
            goto error;
        }
        if (batch_size == 2) {
            q1_seq1 = PySequence_Fast(PySequence_Fast_GET_ITEM(batch_seq, 1), "Each q1 row must be a sequence.");
            if (q1_seq1 == NULL) {
                Py_DECREF(q1_seq0);
                goto error;
            }
            if (PySequence_Fast_GET_SIZE(q1_seq1) != plan->nvars) {
                PyErr_SetString(PyExc_ValueError, "Each q1 row must have length nvars.");
                Py_DECREF(q1_seq0);
                Py_DECREF(q1_seq1);
                goto error;
            }
        }

        if (plan->level == 3) {
            unsigned char *q1_row0 = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(unsigned char));
            unsigned char *q1_row1 = NULL;
            NativeScaledComplex scalar0;
            NativeScaledComplex scalar1;
            if (q1_row0 == NULL) {
                Py_DECREF(q1_seq0);
                Py_XDECREF(q1_seq1);
                PyErr_NoMemory();
                goto error;
            }
            if (fill_q1_row_level3(q1_seq0, plan->nvars, q1_row0) < 0) {
                PyMem_Free(q1_row0);
                Py_DECREF(q1_seq0);
                Py_XDECREF(q1_seq1);
                goto error;
            }
            if (batch_size == 1) {
                if (evaluate_q3_free_treewidth_plan_row_level3(plan, q1_row0, &scalar0) < 0) {
                    PyMem_Free(q1_row0);
                    Py_DECREF(q1_seq0);
                    Py_XDECREF(q1_seq1);
                    goto error;
                }
            } else {
                q1_row1 = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(unsigned char));
                if (q1_row1 == NULL) {
                    PyMem_Free(q1_row0);
                    Py_DECREF(q1_seq0);
                    Py_DECREF(q1_seq1);
                    PyErr_NoMemory();
                    goto error;
                }
                if (
                    fill_q1_row_level3(q1_seq1, plan->nvars, q1_row1) < 0
                    || evaluate_q3_free_treewidth_plan_pair_level3(plan, q1_row0, q1_row1, &scalar0, &scalar1) < 0
                ) {
                    PyMem_Free(q1_row0);
                    PyMem_Free(q1_row1);
                    Py_DECREF(q1_seq0);
                    Py_DECREF(q1_seq1);
                    goto error;
                }
                PyMem_Free(q1_row1);
            }
            PyMem_Free(q1_row0);
            row_result0 = make_q3_free_treewidth_row_result(scalar0, plan->max_scope);
            if (batch_size == 2) {
                row_result1 = make_q3_free_treewidth_row_result(scalar1, plan->max_scope);
            }
        } else {
            long long *q1_row0 = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(long long));
            long long *q1_row1 = NULL;
            NativeScaledComplex scalar0;
            NativeScaledComplex scalar1;
            if (q1_row0 == NULL) {
                Py_DECREF(q1_seq0);
                Py_XDECREF(q1_seq1);
                PyErr_NoMemory();
                goto error;
            }
            if (fill_q1_row_generic(q1_seq0, plan->nvars, plan->mod_q1, q1_row0) < 0) {
                PyMem_Free(q1_row0);
                Py_DECREF(q1_seq0);
                Py_XDECREF(q1_seq1);
                goto error;
            }
            if (batch_size == 1) {
                if (evaluate_q3_free_treewidth_plan_row_generic(plan, q1_row0, &scalar0) < 0) {
                    PyMem_Free(q1_row0);
                    Py_DECREF(q1_seq0);
                    Py_XDECREF(q1_seq1);
                    goto error;
                }
            } else {
                q1_row1 = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(long long));
                if (q1_row1 == NULL) {
                    PyMem_Free(q1_row0);
                    Py_DECREF(q1_seq0);
                    Py_DECREF(q1_seq1);
                    PyErr_NoMemory();
                    goto error;
                }
                if (
                    fill_q1_row_generic(q1_seq1, plan->nvars, plan->mod_q1, q1_row1) < 0
                    || evaluate_q3_free_treewidth_plan_pair_generic(plan, q1_row0, q1_row1, &scalar0, &scalar1) < 0
                ) {
                    PyMem_Free(q1_row0);
                    PyMem_Free(q1_row1);
                    Py_DECREF(q1_seq0);
                    Py_DECREF(q1_seq1);
                    goto error;
                }
                PyMem_Free(q1_row1);
            }
            PyMem_Free(q1_row0);
            row_result0 = make_q3_free_treewidth_row_result(scalar0, plan->max_scope);
            if (batch_size == 2) {
                row_result1 = make_q3_free_treewidth_row_result(scalar1, plan->max_scope);
            }
        }
        Py_DECREF(q1_seq0);
        Py_XDECREF(q1_seq1);
        if (row_result0 == NULL || (batch_size == 2 && row_result1 == NULL)) {
            Py_XDECREF(row_result0);
            Py_XDECREF(row_result1);
            goto error;
        }
        PyList_SET_ITEM(results, 0, row_result0);
        if (batch_size == 2) {
            PyList_SET_ITEM(results, 1, row_result1);
        }
        Py_DECREF(batch_seq);
        return results;
    }

    for (row_idx = 0; row_idx < batch_size; ++row_idx) {
        PyObject *q1_seq = PySequence_Fast(PySequence_Fast_GET_ITEM(batch_seq, row_idx), "Each q1 row must be a sequence.");
        PyObject *row_result = NULL;
        if (q1_seq == NULL) {
            goto error;
        }
        if (PySequence_Fast_GET_SIZE(q1_seq) != plan->nvars) {
            PyErr_SetString(PyExc_ValueError, "Each q1 row must have length nvars.");
            Py_DECREF(q1_seq);
            goto error;
        }
        if (plan->level == 3) {
            unsigned char *q1_row = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(unsigned char));
            NativeScaledComplex scalar;
            if (q1_row == NULL) {
                Py_DECREF(q1_seq);
                PyErr_NoMemory();
                goto error;
            }
            if (fill_q1_row_level3(q1_seq, plan->nvars, q1_row) < 0 || evaluate_q3_free_treewidth_plan_row_level3(plan, q1_row, &scalar) < 0) {
                PyMem_Free(q1_row);
                Py_DECREF(q1_seq);
                goto error;
            }
            row_result = make_q3_free_treewidth_row_result(scalar, plan->max_scope);
            PyMem_Free(q1_row);
        } else {
            long long *q1_row = PyMem_Malloc((size_t) (plan->nvars > 0 ? plan->nvars : 1) * sizeof(long long));
            NativeScaledComplex scalar;
            if (q1_row == NULL) {
                Py_DECREF(q1_seq);
                PyErr_NoMemory();
                goto error;
            }
            if (fill_q1_row_generic(q1_seq, plan->nvars, plan->mod_q1, q1_row) < 0 || evaluate_q3_free_treewidth_plan_row_generic(plan, q1_row, &scalar) < 0) {
                PyMem_Free(q1_row);
                Py_DECREF(q1_seq);
                goto error;
            }
            row_result = make_q3_free_treewidth_row_result(scalar, plan->max_scope);
            PyMem_Free(q1_row);
        }
        Py_DECREF(q1_seq);
        if (row_result == NULL) {
            goto error;
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


static PyObject *build_scaled_factor_treewidth_plan_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    PyObject *factors_obj;
    PyObject *order_obj;
    PyObject *order_seq = NULL;
    NativeQ3FreeTreewidthPlan *plan = NULL;
    PyObject *capsule = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "nOO", &nvars, &factors_obj, &order_obj)) {
        return NULL;
    }
    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (order_seq == NULL) {
        return NULL;
    }
    if (build_scaled_factor_treewidth_plan_internal(nvars, factors_obj, order_seq, &plan) < 0) {
        Py_DECREF(order_seq);
        return NULL;
    }
    Py_DECREF(order_seq);

    capsule = PyCapsule_New(plan, q3_free_treewidth_plan_capsule_name, q3_free_treewidth_plan_capsule_destructor);
    if (capsule == NULL) {
        free_q3_free_treewidth_plan(plan);
        return NULL;
    }
    return capsule;
}


static PyObject *sum_scaled_factor_treewidth_preplanned_native(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    NativeQ3FreeTreewidthPlan *plan;
    NativeScaledComplex total;
    PyObject *total_obj = NULL;
    PyObject *exp_obj = NULL;
    PyObject *width_obj = NULL;
    PyObject *result = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    plan = (NativeQ3FreeTreewidthPlan *) PyCapsule_GetPointer(capsule, q3_free_treewidth_plan_capsule_name);
    if (plan == NULL) {
        return NULL;
    }
    if (evaluate_fixed_scaled_treewidth_plan(plan, &total) < 0) {
        return NULL;
    }

    total_obj = PyTuple_Pack(
        2,
        PyComplex_FromDoubles(total.value.real, total.value.imag),
        PyLong_FromLongLong(total.half_pow2_exp)
    );
    width_obj = PyLong_FromSsize_t(plan->max_scope);
    if (total_obj == NULL || width_obj == NULL) {
        Py_XDECREF(total_obj);
        Py_XDECREF(width_obj);
        return NULL;
    }
    result = PyTuple_Pack(2, total_obj, width_obj);
    Py_DECREF(total_obj);
    Py_DECREF(width_obj);
    return result;
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


static PyObject *sum_q3_free_treewidth_preplanned_batch_scaled_array_native(PyObject *self, PyObject *args)
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
    return evaluate_q3_free_treewidth_plan_batch_buffer(plan, q1_batch_obj);
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


