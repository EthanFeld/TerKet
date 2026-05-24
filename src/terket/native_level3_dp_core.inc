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


PyObject *sum_treewidth_dp_level3_native(PyObject *self, PyObject *args)
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
    plan->slot_source_kinds = PyMem_Calloc((size_t) factor_capacity, sizeof(unsigned char));
    plan->slot_source_indexes = PyMem_Calloc((size_t) factor_capacity, sizeof(Py_ssize_t));
    plan->steps = PyMem_Calloc((size_t) (order_len > 0 ? order_len : 1), sizeof(NativeLevel3PlanStep));
    if (
        plan->slot_arities == NULL
        || plan->slot_table_sizes == NULL
        || plan->slot_workspace_offsets == NULL
        || plan->initial_tables == NULL
        || plan->slot_source_kinds == NULL
        || plan->slot_source_indexes == NULL
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
        plan->slot_source_kinds[factor_count] = NATIVE_LEVEL3_SLOT_SOURCE_Q1;
        plan->slot_source_indexes[factor_count] = pos;
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
        plan->slot_source_kinds[factor_count] = NATIVE_LEVEL3_SLOT_SOURCE_Q2;
        plan->slot_source_indexes[factor_count] = plan->q2_support_count;
        plan->q2_support_count += 1;
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
        plan->slot_source_kinds[factor_count] = NATIVE_LEVEL3_SLOT_SOURCE_Q3;
        plan->slot_source_indexes[factor_count] = plan->q3_support_count;
        plan->q3_support_count += 1;
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


static void initialize_level3_plan_slot_views_for_batch_row(
    NativeLevel3TreewidthPlan *plan,
    const long long *q1_row,
    const long long *q2_row,
    const long long *q3_row
)
{
    Py_ssize_t slot_idx;
    for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
        plan->slot_views[slot_idx] = NULL;
        plan->slot_owned[slot_idx] = 0U;
    }

    for (slot_idx = 0; slot_idx < plan->initial_slot_count; ++slot_idx) {
        NativeComplex *table = plan->workspace + plan->slot_workspace_offsets[slot_idx];
        unsigned char kind = plan->slot_source_kinds[slot_idx];
        Py_ssize_t source_idx = plan->slot_source_indexes[slot_idx];
        size_t table_idx;
        size_t table_size = plan->slot_table_sizes[slot_idx];

        for (table_idx = 0; table_idx < table_size; ++table_idx) {
            table[table_idx] = complex_make(1.0, 0.0);
        }

        if (kind == NATIVE_LEVEL3_SLOT_SOURCE_Q1) {
            table[1] = omega_level3_value((int) positive_mod_ll(q1_row[source_idx], 8LL));
        } else if (kind == NATIVE_LEVEL3_SLOT_SOURCE_Q2) {
            long long residue = positive_mod_ll(q2_row[source_idx], 4LL);
            table[3] = omega_level3_value((int) ((2LL * residue) & 7LL));
        } else if (kind == NATIVE_LEVEL3_SLOT_SOURCE_Q3) {
            long long residue = positive_mod_ll(q3_row[source_idx], 2LL);
            table[7] = omega_level3_value((int) ((4LL * residue) & 7LL));
        }

        plan->slot_views[slot_idx] = table;
        plan->slot_owned[slot_idx] = 1U;
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


static int evaluate_preinitialized_level3_treewidth_plan(
    NativeLevel3TreewidthPlan *plan,
    const long long *q1_row,
    NativeComplex *out_total
)
{
    NativeComplex scalar = complex_make(1.0, 0.0);
    Py_ssize_t step_idx;

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        NativeLevel3PlanStep *step = &plan->steps[step_idx];
        NativeComplex q1_weights[2];
        q1_weights[0] = complex_make(1.0, 0.0);
        q1_weights[1] = q1_row == NULL
            ? complex_make(1.0, 0.0)
            : omega_level3_value((int) positive_mod_ll(q1_row[step->var], 8LL));
        if (step->bucket_count == 0) {
            scalar = complex_mul(scalar, complex_add(q1_weights[0], q1_weights[1]));
            continue;
        }

        if (step->output_mode == 0) {
            NativeComplex total = complex_make(0.0, 0.0);
            int fixed_value;
            for (fixed_value = 0; fixed_value <= 1; ++fixed_value) {
                NativeComplex weight = q1_weights[fixed_value];
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
                    NativeComplex weight = q1_weights[fixed_value];
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


static int evaluate_level3_treewidth_plan(
    NativeLevel3TreewidthPlan *plan,
    NativeComplex *out_total
)
{
    initialize_level3_plan_slot_views(plan);
    return evaluate_preinitialized_level3_treewidth_plan(plan, NULL, out_total);
}


static int validate_level3_factor_batch_buffer(
    PyObject *batch_obj,
    const char *name,
    Py_ssize_t expected_rows,
    Py_ssize_t expected_cols,
    Py_buffer *view
)
{
    if (PyObject_GetBuffer(batch_obj, view, PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
        return -1;
    }
    if (view->ndim != 2) {
        PyErr_Format(PyExc_ValueError, "%s buffer must be 2-dimensional.", name);
        PyBuffer_Release(view);
        return -1;
    }
    if (view->shape == NULL || view->strides == NULL) {
        PyErr_Format(PyExc_ValueError, "%s buffer is missing shape/stride metadata.", name);
        PyBuffer_Release(view);
        return -1;
    }
    if (view->itemsize != (Py_ssize_t) sizeof(long long)) {
        PyErr_Format(PyExc_ValueError, "%s buffer must use 64-bit signed integers.", name);
        PyBuffer_Release(view);
        return -1;
    }
    if ((expected_rows >= 0 && view->shape[0] != expected_rows) || view->shape[1] != expected_cols) {
        if (expected_rows >= 0) {
            PyErr_Format(
                PyExc_ValueError,
                "%s buffer must have shape (%zd, %zd).",
                name,
                expected_rows,
                expected_cols
            );
        } else {
            PyErr_Format(
                PyExc_ValueError,
                "%s buffer must have %zd columns.",
                name,
                expected_cols
            );
        }
        PyBuffer_Release(view);
        return -1;
    }
    if (expected_cols > 0 && view->strides[1] != (Py_ssize_t) sizeof(long long)) {
        PyErr_Format(PyExc_ValueError, "%s rows must be contiguous int64 arrays.", name);
        PyBuffer_Release(view);
        return -1;
    }
    if ((view->strides[0] % (Py_ssize_t) sizeof(long long)) != 0) {
        PyErr_Format(PyExc_ValueError, "%s row stride must be a whole number of int64 values.", name);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}


PyObject *build_level3_treewidth_plan_native(PyObject *self, PyObject *args)
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


PyObject *sum_level3_treewidth_preplanned_native(PyObject *self, PyObject *args)
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


PyObject *sum_level3_treewidth_preplanned_batch_array_native(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    PyObject *q1_batch_obj;
    PyObject *q2_batch_obj;
    PyObject *q3_batch_obj;
    NativeLevel3TreewidthPlan *plan;
    Py_buffer q1_view;
    Py_buffer q2_view;
    Py_buffer q3_view;
    Py_ssize_t batch_size;
    Py_ssize_t q1_row_stride_elems;
    Py_ssize_t q2_row_stride_elems;
    Py_ssize_t q3_row_stride_elems;
    Py_ssize_t row_idx;
    PyObject *results = NULL;
    PyObject *width_obj = NULL;
    PyObject *result = NULL;

    (void) self;

    memset(&q1_view, 0, sizeof(Py_buffer));
    memset(&q2_view, 0, sizeof(Py_buffer));
    memset(&q3_view, 0, sizeof(Py_buffer));

    if (!PyArg_ParseTuple(args, "OOOO", &capsule, &q1_batch_obj, &q2_batch_obj, &q3_batch_obj)) {
        return NULL;
    }
    plan = (NativeLevel3TreewidthPlan *) PyCapsule_GetPointer(capsule, level3_treewidth_plan_capsule_name);
    if (plan == NULL) {
        return NULL;
    }

    if (validate_level3_factor_batch_buffer(q1_batch_obj, "q1_batch", -1, plan->nvars, &q1_view) < 0) {
        return NULL;
    }
    batch_size = q1_view.shape[0];
    if (validate_level3_factor_batch_buffer(q2_batch_obj, "q2_batch", batch_size, plan->q2_support_count, &q2_view) < 0) {
        PyBuffer_Release(&q1_view);
        return NULL;
    }
    if (validate_level3_factor_batch_buffer(q3_batch_obj, "q3_batch", batch_size, plan->q3_support_count, &q3_view) < 0) {
        PyBuffer_Release(&q1_view);
        PyBuffer_Release(&q2_view);
        return NULL;
    }

    q1_row_stride_elems = q1_view.strides[0] / (Py_ssize_t) sizeof(long long);
    q2_row_stride_elems = q2_view.strides[0] / (Py_ssize_t) sizeof(long long);
    q3_row_stride_elems = q3_view.strides[0] / (Py_ssize_t) sizeof(long long);
    results = PyList_New(batch_size);
    if (results == NULL) {
        PyBuffer_Release(&q1_view);
        PyBuffer_Release(&q2_view);
        PyBuffer_Release(&q3_view);
        return NULL;
    }

    for (row_idx = 0; row_idx < batch_size; ++row_idx) {
        const long long *q1_row = ((const long long *) q1_view.buf) + (row_idx * q1_row_stride_elems);
        const long long *q2_row = q2_view.buf == NULL
            ? NULL
            : ((const long long *) q2_view.buf) + (row_idx * q2_row_stride_elems);
        const long long *q3_row = q3_view.buf == NULL
            ? NULL
            : ((const long long *) q3_view.buf) + (row_idx * q3_row_stride_elems);
        NativeComplex total;
        PyObject *total_obj = NULL;

        initialize_level3_plan_slot_views_for_batch_row(plan, q1_row, q2_row, q3_row);
        if (evaluate_preinitialized_level3_treewidth_plan(plan, q1_row, &total) < 0) {
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }

        total_obj = PyComplex_FromDoubles(total.real, total.imag);
        if (total_obj == NULL) {
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }
        PyList_SET_ITEM(results, row_idx, total_obj);
    }

    width_obj = PyLong_FromSsize_t(plan->max_scope);
    if (width_obj == NULL) {
        PyBuffer_Release(&q1_view);
        PyBuffer_Release(&q2_view);
        PyBuffer_Release(&q3_view);
        Py_DECREF(results);
        return NULL;
    }
    result = PyTuple_Pack(2, results, width_obj);
    PyBuffer_Release(&q1_view);
    PyBuffer_Release(&q2_view);
    PyBuffer_Release(&q3_view);
    Py_DECREF(results);
    Py_DECREF(width_obj);
    return result;
}


