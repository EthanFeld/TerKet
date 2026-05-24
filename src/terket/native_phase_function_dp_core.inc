PyObject *sum_factor_tables_scaled_native(PyObject *self, PyObject *args)
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


static int build_phase_function_treewidth_support_plan_internal(
    Py_ssize_t nvars,
    long level,
    PyObject *q2_obj,
    PyObject *q3_obj,
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
    if (!PyDict_Check(q2_obj) || !PyDict_Check(q3_obj)) {
        PyErr_SetString(PyExc_TypeError, "q2 and q3 support must be dicts.");
        return -1;
    }
    if (level < 1 || level > 60) {
        PyErr_SetString(PyExc_ValueError, "level must lie in [1, 60] for native phase-function treewidth plans.");
        return -1;
    }

    order_len = PySequence_Fast_GET_SIZE(order_seq);
    factor_capacity = PyDict_Size(q2_obj) + PyDict_Size(q3_obj) + order_len + 1;
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
    plan->level = level;
    plan->mod_q1 = 1LL << level;
    plan->slot_arities = PyMem_Calloc((size_t) factor_capacity, sizeof(Py_ssize_t));
    plan->slot_table_sizes = PyMem_Calloc((size_t) factor_capacity, sizeof(size_t));
    plan->slot_workspace_offsets = PyMem_Calloc((size_t) factor_capacity, sizeof(size_t));
    plan->steps = PyMem_Calloc((size_t) (order_len > 0 ? order_len : 1), sizeof(NativeQ3FreePlanStep));
    plan->initial_tables = PyMem_Calloc((size_t) factor_capacity, sizeof(NativeScaledComplex *));
    plan->slot_source_kinds = PyMem_Calloc((size_t) factor_capacity, sizeof(unsigned char));
    plan->slot_source_indexes = PyMem_Calloc((size_t) factor_capacity, sizeof(Py_ssize_t));
    if (
        plan->slot_arities == NULL
        || plan->slot_table_sizes == NULL
        || plan->slot_workspace_offsets == NULL
        || plan->steps == NULL
        || plan->initial_tables == NULL
        || plan->slot_source_kinds == NULL
        || plan->slot_source_indexes == NULL
    ) {
        PyErr_NoMemory();
        goto error;
    }

    while (PyDict_Next(q2_obj, &pos, &key, &value)) {
        Py_ssize_t left;
        Py_ssize_t right;
        NativePlanScopeFactor *factor;
        if (parse_pair_key(key, &left, &right) < 0) {
            goto error;
        }
        if (factor_count >= factor_capacity) {
            PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
            goto error;
        }
        factor = &states[factor_count];
        factor->arity = 2;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(2 * sizeof(Py_ssize_t));
        if (factor->scope == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = left;
        factor->scope[1] = right;
        plan->slot_arities[factor_count] = 2;
        plan->slot_table_sizes[factor_count] = 4;
        plan->slot_source_kinds[factor_count] = NATIVE_LEVEL3_SLOT_SOURCE_Q2;
        plan->slot_source_indexes[factor_count] = plan->q2_support_count;
        plan->q2_support_count += 1;
        factor_count += 1;
    }

    pos = 0;
    while (PyDict_Next(q3_obj, &pos, &key, &value)) {
        Py_ssize_t a;
        Py_ssize_t b;
        Py_ssize_t c;
        NativePlanScopeFactor *factor;
        if (parse_triple_key(key, &a, &b, &c) < 0) {
            goto error;
        }
        if (factor_count >= factor_capacity) {
            PyErr_SetString(PyExc_RuntimeError, "Factor capacity exceeded.");
            goto error;
        }
        factor = &states[factor_count];
        factor->arity = 3;
        factor->alive = 1;
        factor->scope = PyMem_Malloc(3 * sizeof(Py_ssize_t));
        if (factor->scope == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        factor->scope[0] = a;
        factor->scope[1] = b;
        factor->scope[2] = c;
        plan->slot_arities[factor_count] = 3;
        plan->slot_table_sizes[factor_count] = 8;
        plan->slot_source_kinds[factor_count] = NATIVE_LEVEL3_SLOT_SOURCE_Q3;
        plan->slot_source_indexes[factor_count] = plan->q3_support_count;
        plan->q3_support_count += 1;
        factor_count += 1;
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
                PyErr_SetString(PyExc_OverflowError, "Union scope too wide for native phase-function treewidth DP.");
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

    {
        size_t workspace_size = 0;
        for (pos = 0; pos < plan->initial_slot_count; ++pos) {
            plan->slot_workspace_offsets[pos] = workspace_size;
            workspace_size += plan->slot_table_sizes[pos];
        }
        plan->initial_workspace_size = workspace_size;
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


static int fill_phase_function_dynamic_slot_table(
    const NativeQ3FreeTreewidthPlan *plan,
    unsigned char kind,
    Py_ssize_t source_idx,
    const long long *q2_row,
    const long long *q3_row,
    NativeScaledComplex *table,
    size_t table_size
);

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


static int initialize_phase_function_treewidth_plan_slot_views(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables,
    unsigned char *slot_owned,
    const long long *q2_row,
    const long long *q3_row,
    NativeScaledComplex *row_workspace
)
{
    Py_ssize_t slot_idx;

    for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
        slot_tables[slot_idx] = NULL;
        slot_owned[slot_idx] = 0U;
    }

    for (slot_idx = 0; slot_idx < plan->initial_slot_count; ++slot_idx) {
        NativeScaledComplex *table = row_workspace + plan->slot_workspace_offsets[slot_idx];
        size_t table_size = plan->slot_table_sizes[slot_idx];
        unsigned char kind = plan->slot_source_kinds[slot_idx];
        Py_ssize_t source_idx = plan->slot_source_indexes[slot_idx];
        if (fill_phase_function_dynamic_slot_table(plan, kind, source_idx, q2_row, q3_row, table, table_size) < 0) {
            return -1;
        }

        slot_tables[slot_idx] = table;
    }
    return 0;
}


static int fill_phase_function_dynamic_slot_table(
    const NativeQ3FreeTreewidthPlan *plan,
    unsigned char kind,
    Py_ssize_t source_idx,
    const long long *q2_row,
    const long long *q3_row,
    NativeScaledComplex *table,
    size_t table_size
)
{
    long long mod_q2 = (plan->level > 1) ? (plan->mod_q1 >> 1) : 1LL;
    long long mod_q3 = (plan->level > 2) ? (plan->mod_q1 >> 2) : 1LL;
    size_t table_idx;

    for (table_idx = 0; table_idx < table_size; ++table_idx) {
        table[table_idx] = scaled_complex_make(1.0, 0.0, 0);
    }

    if (kind == NATIVE_LEVEL3_SLOT_SOURCE_Q2) {
        long long residue = positive_mod_ll(q2_row[source_idx], mod_q2);
        NativeComplex phase = omega_dyadic_value((2LL * residue) % plan->mod_q1, plan->mod_q1);
        table[3] = scaled_complex_make(phase.real, phase.imag, 0);
        return 0;
    }
    if (kind == NATIVE_LEVEL3_SLOT_SOURCE_Q3) {
        long long residue = positive_mod_ll(q3_row[source_idx], mod_q3);
        NativeComplex phase = omega_dyadic_value((4LL * residue) % plan->mod_q1, plan->mod_q1);
        table[7] = scaled_complex_make(phase.real, phase.imag, 0);
        return 0;
    }

    PyErr_SetString(PyExc_RuntimeError, "Unsupported dynamic source kind for phase-function treewidth plan.");
    return -1;
}


static int initialize_phase_function_treewidth_plan_slot_views_for_pair_rows(
    const NativeQ3FreeTreewidthPlan *plan,
    NativeScaledComplex **slot_tables0,
    NativeScaledComplex **slot_tables1,
    unsigned char *slot_owned0,
    unsigned char *slot_owned1,
    const long long *q2_row0,
    const long long *q3_row0,
    const long long *q2_row1,
    const long long *q3_row1,
    NativeScaledComplex *row_workspace0,
    NativeScaledComplex *row_workspace1
)
{
    Py_ssize_t slot_idx;
    long long mod_q2 = (plan->level > 1) ? (plan->mod_q1 >> 1) : 1LL;
    long long mod_q3 = (plan->level > 2) ? (plan->mod_q1 >> 2) : 1LL;

    for (slot_idx = 0; slot_idx < plan->slot_count; ++slot_idx) {
        slot_tables0[slot_idx] = NULL;
        slot_tables1[slot_idx] = NULL;
        slot_owned0[slot_idx] = 0U;
        slot_owned1[slot_idx] = 0U;
    }

    for (slot_idx = 0; slot_idx < plan->initial_slot_count; ++slot_idx) {
        NativeScaledComplex *table0 = row_workspace0 + plan->slot_workspace_offsets[slot_idx];
        NativeScaledComplex *table1 = row_workspace1 + plan->slot_workspace_offsets[slot_idx];
        size_t table_size = plan->slot_table_sizes[slot_idx];
        unsigned char kind = plan->slot_source_kinds[slot_idx];
        Py_ssize_t source_idx = plan->slot_source_indexes[slot_idx];
        int share_table = 0;

        if (kind == NATIVE_LEVEL3_SLOT_SOURCE_Q2) {
            share_table = (
                positive_mod_ll(q2_row0[source_idx], mod_q2)
                == positive_mod_ll(q2_row1[source_idx], mod_q2)
            );
        } else if (kind == NATIVE_LEVEL3_SLOT_SOURCE_Q3) {
            share_table = (
                positive_mod_ll(q3_row0[source_idx], mod_q3)
                == positive_mod_ll(q3_row1[source_idx], mod_q3)
            );
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Unsupported dynamic source kind for phase-function treewidth plan.");
            return -1;
        }

        if (fill_phase_function_dynamic_slot_table(plan, kind, source_idx, q2_row0, q3_row0, table0, table_size) < 0) {
            return -1;
        }
        slot_tables0[slot_idx] = table0;

        if (share_table) {
            slot_tables1[slot_idx] = table0;
        } else {
            if (fill_phase_function_dynamic_slot_table(plan, kind, source_idx, q2_row1, q3_row1, table1, table_size) < 0) {
                return -1;
            }
            slot_tables1[slot_idx] = table1;
        }
    }
    return 0;
}


static int evaluate_phase_function_treewidth_plan_row_generic(
    const NativeQ3FreeTreewidthPlan *plan,
    const long long *q1_row,
    const long long *q2_row,
    const long long *q3_row,
    NativeScaledComplex *out_scalar
)
{
    NativeScaledComplex **slot_tables = NULL;
    unsigned char *slot_owned = NULL;
    NativeScaledComplex *row_workspace = NULL;
    NativeScaledComplex scalar = scaled_complex_make(1.0, 0.0, 0);
    Py_ssize_t step_idx;
    int failed = 0;

    slot_tables = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_owned = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    row_workspace = PyMem_Malloc(
        (plan->initial_workspace_size > 0 ? plan->initial_workspace_size : 1) * sizeof(NativeScaledComplex)
    );
    if (slot_tables == NULL || slot_owned == NULL || row_workspace == NULL) {
        PyErr_NoMemory();
        failed = 1;
        goto cleanup;
    }
    if (initialize_phase_function_treewidth_plan_slot_views(plan, slot_tables, slot_owned, q2_row, q3_row, row_workspace) < 0) {
        failed = 1;
        goto cleanup;
    }

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        const NativeQ3FreePlanStep *step = &plan->steps[step_idx];
        long long unary_coeff = positive_mod_ll(q1_row[step->var], plan->mod_q1);

        if (step->bucket_count == 0) {
            if (unary_coeff == 0) {
                scalar = renormalize_scaled_complex_if_needed(scalar.value, scalar.half_pow2_exp + 2);
            } else {
                NativeComplex phase = omega_dyadic_value(unary_coeff, plan->mod_q1);
                scalar = mul_scaled_complex(
                    scalar,
                    add_scaled_complex(
                        scaled_complex_make(1.0, 0.0, 0),
                        scaled_complex_make(phase.real, phase.imag, 0)
                    )
                );
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
                            PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native phase-function treewidth plan.");
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
                    slot_tables[step->output_slot][merge_idx] = mul_scaled_complex(
                        slot_tables[step->output_slot][merge_idx],
                        new_table[merge_idx]
                    );
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
    PyMem_Free(row_workspace);
    PyMem_Free(slot_tables);
    PyMem_Free(slot_owned);
    if (failed) {
        return -1;
    }
    *out_scalar = scalar;
    return 0;
}


static int evaluate_phase_function_treewidth_plan_pair_generic(
    const NativeQ3FreeTreewidthPlan *plan,
    const long long *q1_row0,
    const long long *q2_row0,
    const long long *q3_row0,
    const long long *q1_row1,
    const long long *q2_row1,
    const long long *q3_row1,
    NativeScaledComplex *out_scalar0,
    NativeScaledComplex *out_scalar1
)
{
    NativeScaledComplex **slot_tables0 = NULL;
    NativeScaledComplex **slot_tables1 = NULL;
    unsigned char *slot_owned0 = NULL;
    unsigned char *slot_owned1 = NULL;
    NativeScaledComplex *row_workspace0 = NULL;
    NativeScaledComplex *row_workspace1 = NULL;
    NativeScaledComplex scalar0 = scaled_complex_make(1.0, 0.0, 0);
    NativeScaledComplex scalar1 = scaled_complex_make(1.0, 0.0, 0);
    Py_ssize_t step_idx;
    int failed = 0;

    slot_tables0 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_tables1 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(NativeScaledComplex *));
    slot_owned0 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    slot_owned1 = PyMem_Calloc((size_t) (plan->slot_count > 0 ? plan->slot_count : 1), sizeof(unsigned char));
    row_workspace0 = PyMem_Malloc(
        (plan->initial_workspace_size > 0 ? plan->initial_workspace_size : 1) * sizeof(NativeScaledComplex)
    );
    row_workspace1 = PyMem_Malloc(
        (plan->initial_workspace_size > 0 ? plan->initial_workspace_size : 1) * sizeof(NativeScaledComplex)
    );
    if (
        slot_tables0 == NULL || slot_tables1 == NULL
        || slot_owned0 == NULL || slot_owned1 == NULL
        || row_workspace0 == NULL || row_workspace1 == NULL
    ) {
        PyErr_NoMemory();
        failed = 1;
        goto cleanup;
    }
    if (
        initialize_phase_function_treewidth_plan_slot_views_for_pair_rows(
            plan,
            slot_tables0,
            slot_tables1,
            slot_owned0,
            slot_owned1,
            q2_row0,
            q3_row0,
            q2_row1,
            q3_row1,
            row_workspace0,
            row_workspace1
        ) < 0
    ) {
        failed = 1;
        goto cleanup;
    }

    for (step_idx = 0; step_idx < plan->step_count; ++step_idx) {
        const NativeQ3FreePlanStep *step = &plan->steps[step_idx];
        long long unary_coeff0 = positive_mod_ll(q1_row0[step->var], plan->mod_q1);
        long long unary_coeff1 = positive_mod_ll(q1_row1[step->var], plan->mod_q1);

        if (step->bucket_count == 0) {
            if (unary_coeff0 == 0) {
                scalar0 = renormalize_scaled_complex_if_needed(scalar0.value, scalar0.half_pow2_exp + 2);
            } else {
                NativeComplex phase0 = omega_dyadic_value(unary_coeff0, plan->mod_q1);
                scalar0 = mul_scaled_complex(
                    scalar0,
                    add_scaled_complex(
                        scaled_complex_make(1.0, 0.0, 0),
                        scaled_complex_make(phase0.real, phase0.imag, 0)
                    )
                );
            }
            if (unary_coeff1 == 0) {
                scalar1 = renormalize_scaled_complex_if_needed(scalar1.value, scalar1.half_pow2_exp + 2);
            } else {
                NativeComplex phase1 = omega_dyadic_value(unary_coeff1, plan->mod_q1);
                scalar1 = mul_scaled_complex(
                    scalar1,
                    add_scaled_complex(
                        scaled_complex_make(1.0, 0.0, 0),
                        scaled_complex_make(phase1.real, phase1.imag, 0)
                    )
                );
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
                                PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native phase-function treewidth plan.");
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
                                PyErr_SetString(PyExc_RuntimeError, "Missing factor table for native phase-function treewidth plan.");
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
                        slot_tables0[step->output_slot][merge_idx] = mul_scaled_complex(
                            slot_tables0[step->output_slot][merge_idx],
                            new_table0[merge_idx]
                        );
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
                        slot_tables0[step->output_slot][merge_idx] = mul_scaled_complex(
                            slot_tables0[step->output_slot][merge_idx],
                            new_table0[merge_idx]
                        );
                        slot_tables1[step->output_slot][merge_idx] = mul_scaled_complex(
                            slot_tables1[step->output_slot][merge_idx],
                            identical_step ? new_table0[merge_idx] : new_table1[merge_idx]
                        );
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
    PyMem_Free(row_workspace0);
    PyMem_Free(row_workspace1);
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


static const size_t q3_free_treewidth_precomputed_bucket_index_limit = ((size_t) 1) << 20;

PyObject *build_scaled_factor_treewidth_plan_native(PyObject *self, PyObject *args)
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


PyObject *build_phase_function_treewidth_support_plan_native(PyObject *self, PyObject *args)
{
    Py_ssize_t nvars;
    long level;
    PyObject *q2_obj;
    PyObject *q3_obj;
    PyObject *order_obj;
    PyObject *order_seq = NULL;
    NativeQ3FreeTreewidthPlan *plan = NULL;
    PyObject *capsule = NULL;

    (void) self;

    if (!PyArg_ParseTuple(args, "nlOOO", &nvars, &level, &q2_obj, &q3_obj, &order_obj)) {
        return NULL;
    }
    order_seq = PySequence_Fast(order_obj, "order must be a sequence.");
    if (order_seq == NULL) {
        return NULL;
    }
    if (build_phase_function_treewidth_support_plan_internal(nvars, level, q2_obj, q3_obj, order_seq, &plan) < 0) {
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


PyObject *sum_scaled_factor_treewidth_preplanned_native(PyObject *self, PyObject *args)
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


PyObject *sum_phase_function_treewidth_preplanned_batch_scaled_array_native(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    PyObject *q1_batch_obj;
    PyObject *q2_batch_obj;
    PyObject *q3_batch_obj;
    NativeQ3FreeTreewidthPlan *plan;
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
    plan = (NativeQ3FreeTreewidthPlan *) PyCapsule_GetPointer(capsule, q3_free_treewidth_plan_capsule_name);
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

    for (row_idx = 0; row_idx + 1 < batch_size; row_idx += 2) {
        const long long *q1_row0 = ((const long long *) q1_view.buf) + (row_idx * q1_row_stride_elems);
        const long long *q2_row0 = q2_view.buf == NULL
            ? NULL
            : ((const long long *) q2_view.buf) + (row_idx * q2_row_stride_elems);
        const long long *q3_row0 = q3_view.buf == NULL
            ? NULL
            : ((const long long *) q3_view.buf) + (row_idx * q3_row_stride_elems);
        const long long *q1_row1 = q1_row0 + q1_row_stride_elems;
        const long long *q2_row1 = q2_view.buf == NULL ? NULL : q2_row0 + q2_row_stride_elems;
        const long long *q3_row1 = q3_view.buf == NULL ? NULL : q3_row0 + q3_row_stride_elems;
        NativeScaledComplex scalar0;
        NativeScaledComplex scalar1;
        PyObject *row_result0 = NULL;
        PyObject *row_result1 = NULL;
        PyObject *value_obj0 = NULL;
        PyObject *exp_obj0 = NULL;
        PyObject *value_obj1 = NULL;
        PyObject *exp_obj1 = NULL;

        if (
            evaluate_phase_function_treewidth_plan_pair_generic(
                plan,
                q1_row0,
                q2_row0,
                q3_row0,
                q1_row1,
                q2_row1,
                q3_row1,
                &scalar0,
                &scalar1
            ) < 0
        ) {
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }

        value_obj0 = PyComplex_FromDoubles(scalar0.value.real, scalar0.value.imag);
        exp_obj0 = PyLong_FromLongLong(scalar0.half_pow2_exp);
        value_obj1 = PyComplex_FromDoubles(scalar1.value.real, scalar1.value.imag);
        exp_obj1 = PyLong_FromLongLong(scalar1.half_pow2_exp);
        if (value_obj0 == NULL || exp_obj0 == NULL || value_obj1 == NULL || exp_obj1 == NULL) {
            Py_XDECREF(value_obj0);
            Py_XDECREF(exp_obj0);
            Py_XDECREF(value_obj1);
            Py_XDECREF(exp_obj1);
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }
        row_result0 = PyTuple_Pack(2, value_obj0, exp_obj0);
        row_result1 = PyTuple_Pack(2, value_obj1, exp_obj1);
        Py_DECREF(value_obj0);
        Py_DECREF(exp_obj0);
        Py_DECREF(value_obj1);
        Py_DECREF(exp_obj1);
        if (row_result0 == NULL || row_result1 == NULL) {
            Py_XDECREF(row_result0);
            Py_XDECREF(row_result1);
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }
        PyList_SET_ITEM(results, row_idx, row_result0);
        PyList_SET_ITEM(results, row_idx + 1, row_result1);
    }

    if (row_idx < batch_size) {
        const long long *q1_row = ((const long long *) q1_view.buf) + (row_idx * q1_row_stride_elems);
        const long long *q2_row = q2_view.buf == NULL
            ? NULL
            : ((const long long *) q2_view.buf) + (row_idx * q2_row_stride_elems);
        const long long *q3_row = q3_view.buf == NULL
            ? NULL
            : ((const long long *) q3_view.buf) + (row_idx * q3_row_stride_elems);
        NativeScaledComplex scalar;
        PyObject *row_result = NULL;
        PyObject *value_obj = NULL;
        PyObject *exp_obj = NULL;

        if (evaluate_phase_function_treewidth_plan_row_generic(plan, q1_row, q2_row, q3_row, &scalar) < 0) {
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }
        value_obj = PyComplex_FromDoubles(scalar.value.real, scalar.value.imag);
        exp_obj = PyLong_FromLongLong(scalar.half_pow2_exp);
        if (value_obj == NULL || exp_obj == NULL) {
            Py_XDECREF(value_obj);
            Py_XDECREF(exp_obj);
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }
        row_result = PyTuple_Pack(2, value_obj, exp_obj);
        Py_DECREF(value_obj);
        Py_DECREF(exp_obj);
        if (row_result == NULL) {
            PyBuffer_Release(&q1_view);
            PyBuffer_Release(&q2_view);
            PyBuffer_Release(&q3_view);
            Py_DECREF(results);
            return NULL;
        }
        PyList_SET_ITEM(results, row_idx, row_result);
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


