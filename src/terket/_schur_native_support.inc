static PyObject *ensure_little_endian_string(void)
{
    if (little_endian_string == NULL) {
        little_endian_string = PyUnicode_FromString("little");
    }
    if (little_endian_string == NULL) {
        return NULL;
    }
    Py_INCREF(little_endian_string);
    return little_endian_string;
}


static PyObject *ensure_global_support_cache(void)
{
    if (global_support_cache == NULL) {
        global_support_cache = PyDict_New();
    }
    return global_support_cache;
}


static inline size_t word_index(Py_ssize_t bit)
{
    return ((size_t) bit) >> 6;
}


static inline uint64_t bit_mask(Py_ssize_t bit)
{
    return 1ULL << (((size_t) bit) & 63U);
}


static inline void bitset_set(uint64_t *bits, Py_ssize_t bit)
{
    bits[word_index(bit)] |= bit_mask(bit);
}


static inline void bitset_clear(uint64_t *bits, Py_ssize_t bit)
{
    bits[word_index(bit)] &= ~bit_mask(bit);
}


static inline Py_ssize_t popcount_u64(uint64_t word)
{
    Py_ssize_t count = 0;
    while (word) {
        word &= word - 1;
        ++count;
    }
    return count;
}


static inline int lowest_bit_index(uint64_t word)
{
    int idx = 0;
    while ((word & 1ULL) == 0) {
        word >>= 1;
        ++idx;
    }
    return idx;
}


static inline NativeComplex complex_make(double real, double imag)
{
    NativeComplex value;
    value.real = real;
    value.imag = imag;
    return value;
}


static inline NativeComplex complex_add(NativeComplex left, NativeComplex right)
{
    return complex_make(left.real + right.real, left.imag + right.imag);
}


static inline NativeComplex complex_mul(NativeComplex left, NativeComplex right)
{
    return complex_make(
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real
    );
}


static inline int scaled_complex_is_zero(NativeScaledComplex value)
{
    return value.value.real == 0.0 && value.value.imag == 0.0;
}


static inline NativeScaledComplex scaled_complex_make(double real, double imag, long long half_pow2_exp)
{
    NativeScaledComplex value;
    value.value = complex_make(real, imag);
    value.half_pow2_exp = half_pow2_exp;
    return value;
}


static NativeScaledComplex normalize_scaled_complex(NativeComplex value, long long half_pow2_exp)
{
    double magnitude;
    int shift = 0;
    if (value.real == 0.0 && value.imag == 0.0) {
        return scaled_complex_make(0.0, 0.0, 0);
    }

    magnitude = fmax(fabs(value.real), fabs(value.imag));
    frexp(magnitude, &shift);
    if (shift != 0) {
        value.real = ldexp(value.real, -shift);
        value.imag = ldexp(value.imag, -shift);
        half_pow2_exp += 2LL * (long long) shift;
    }
    return scaled_complex_make(value.real, value.imag, half_pow2_exp);
}


static NativeScaledComplex renormalize_scaled_complex_if_needed(NativeComplex value, long long half_pow2_exp)
{
    double magnitude;
    if (value.real == 0.0 && value.imag == 0.0) {
        return scaled_complex_make(0.0, 0.0, 0);
    }
    magnitude = fmax(fabs(value.real), fabs(value.imag));
    if (magnitude >= SCALED_RENORMALIZE_MIN && magnitude < SCALED_RENORMALIZE_MAX) {
        return scaled_complex_make(value.real, value.imag, half_pow2_exp);
    }
    return normalize_scaled_complex(value, half_pow2_exp);
}


static NativeComplex scale_complex_by_half_pow2(NativeComplex value, long long half_pow2_exp)
{
    if ((value.real == 0.0 && value.imag == 0.0) || half_pow2_exp == 0) {
        return value;
    }

    if (half_pow2_exp > 0 && (half_pow2_exp % 2LL) != 0) {
        value.real *= SQRT2;
        value.imag *= SQRT2;
        half_pow2_exp -= 1;
    } else if (half_pow2_exp < 0 && (half_pow2_exp % 2LL) != 0) {
        value.real *= SQRT1_2;
        value.imag *= SQRT1_2;
        half_pow2_exp += 1;
    }

    if (half_pow2_exp != 0) {
        int shift = (int) (half_pow2_exp / 2LL);
        value.real = ldexp(value.real, shift);
        value.imag = ldexp(value.imag, shift);
    }
    return value;
}


static NativeScaledComplex add_scaled_complex(NativeScaledComplex left, NativeScaledComplex right)
{
    NativeComplex aligned_right;
    if (scaled_complex_is_zero(left)) {
        return right;
    }
    if (scaled_complex_is_zero(right)) {
        return left;
    }
    if (left.half_pow2_exp < right.half_pow2_exp) {
        NativeScaledComplex tmp = left;
        left = right;
        right = tmp;
    }
    aligned_right = scale_complex_by_half_pow2(
        right.value,
        right.half_pow2_exp - left.half_pow2_exp
    );
    return renormalize_scaled_complex_if_needed(
        complex_add(left.value, aligned_right),
        left.half_pow2_exp
    );
}


static NativeScaledComplex mul_scaled_complex(NativeScaledComplex left, NativeScaledComplex right)
{
    return renormalize_scaled_complex_if_needed(
        complex_mul(left.value, right.value),
        left.half_pow2_exp + right.half_pow2_exp
    );
}


static inline int positive_mod(long value, int modulus)
{
    int residue = (int) (value % (long) modulus);
    if (residue < 0) {
        residue += modulus;
    }
    return residue;
}


static inline long long positive_mod_ll(long long value, long long modulus)
{
    long long residue = value % modulus;
    if (residue < 0) {
        residue += modulus;
    }
    return residue;
}


static void write_u64_le(unsigned char *dest, uint64_t value)
{
    size_t idx;
    for (idx = 0; idx < 8; ++idx) {
        dest[idx] = (unsigned char) ((value >> (8U * idx)) & 0xffU);
    }
}


static int compare_classification_pair_records(const void *left_ptr, const void *right_ptr)
{
    const ClassificationPairRecord *left = (const ClassificationPairRecord *) left_ptr;
    const ClassificationPairRecord *right = (const ClassificationPairRecord *) right_ptr;

    if (left->left < right->left) {
        return -1;
    }
    if (left->left > right->left) {
        return 1;
    }
    if (left->right < right->right) {
        return -1;
    }
    if (left->right > right->right) {
        return 1;
    }
    if (left->flags < right->flags) {
        return -1;
    }
    if (left->flags > right->flags) {
        return 1;
    }
    return 0;
}


static int compare_classification_triple_records(const void *left_ptr, const void *right_ptr)
{
    const ClassificationTripleRecord *left = (const ClassificationTripleRecord *) left_ptr;
    const ClassificationTripleRecord *right = (const ClassificationTripleRecord *) right_ptr;

    if (left->a < right->a) {
        return -1;
    }
    if (left->a > right->a) {
        return 1;
    }
    if (left->b < right->b) {
        return -1;
    }
    if (left->b > right->b) {
        return 1;
    }
    if (left->c < right->c) {
        return -1;
    }
    if (left->c > right->c) {
        return 1;
    }
    return 0;
}


static inline int factor_contains_var(const NativeFactor *factor, Py_ssize_t var)
{
    Py_ssize_t idx;
    for (idx = 0; idx < factor->arity; ++idx) {
        if (factor->scope[idx] == var) {
            return 1;
        }
    }
    return 0;
}


static inline Py_ssize_t project_assignment_bits_u64(uint64_t assignment, const Py_ssize_t *positions, Py_ssize_t count)
{
    Py_ssize_t idx = 0;
    Py_ssize_t out_pos;
    for (out_pos = 0; out_pos < count; ++out_pos) {
        idx |= (Py_ssize_t) (((assignment >> (size_t) positions[out_pos]) & 1ULL) << (size_t) out_pos);
    }
    return idx;
}


static inline uint64_t insert_assignment_bit_u64(uint64_t assignment, Py_ssize_t position, int bit)
{
    uint64_t low_mask = (
        position <= 0
        ? 0ULL
        : ((((uint64_t) 1) << (size_t) position) - 1ULL)
    );
    return (
        (assignment & low_mask)
        | (((uint64_t) (bit & 1)) << (size_t) position)
        | ((assignment & ~low_mask) << 1)
    );
}


static void free_native_factor(NativeFactor *factor);


static NativeComplex native_factor_value_at_index(const NativeFactor *factor, Py_ssize_t assignment_idx)
{
    if (factor->kind == NATIVE_FACTOR_KIND_DEFERRED_DOMLOCAL) {
        const NativeDeferredDominantLocal *deferred = (const NativeDeferredDominantLocal *) factor->aux;
        Py_ssize_t local_idx = 0;
        Py_ssize_t dominant_idx0 = 0;
        Py_ssize_t dominant_idx1 = 0;
        Py_ssize_t pos;
        NativeComplex dominant0;
        NativeComplex dominant1;
        if (deferred == NULL || deferred->dominant_factor == NULL) {
            return complex_make(0.0, 0.0);
        }
        for (pos = 0; pos < deferred->local_support_arity; ++pos) {
            if (((size_t) assignment_idx >> deferred->local_result_positions[pos]) & 1U) {
                local_idx |= ((Py_ssize_t) 1) << pos;
            }
        }
        for (pos = 0; pos < deferred->dominant_arity; ++pos) {
            int bit;
            if (pos == deferred->dominant_var_pos) {
                continue;
            }
            bit = (int) ((((size_t) assignment_idx) >> deferred->dominant_result_positions[pos]) & 1U);
            dominant_idx0 |= ((Py_ssize_t) bit) << pos;
            dominant_idx1 |= ((Py_ssize_t) bit) << pos;
        }
        dominant_idx1 |= ((Py_ssize_t) 1) << deferred->dominant_var_pos;
        dominant0 = native_factor_value_at_index(deferred->dominant_factor, dominant_idx0);
        dominant1 = native_factor_value_at_index(deferred->dominant_factor, dominant_idx1);
        return complex_add(
            complex_mul(dominant0, deferred->local_products[0][local_idx]),
            complex_mul(dominant1, deferred->local_products[1][local_idx])
        );
    }
    return factor->table[assignment_idx];
}


static int materialize_native_factor(NativeFactor *factor)
{
    NativeComplex *table;
    size_t table_size;
    Py_ssize_t idx;
    if (factor == NULL || factor->kind == NATIVE_FACTOR_KIND_CONCRETE) {
        return 0;
    }
    if (factor->arity < 0 || factor->arity >= (Py_ssize_t) (8 * sizeof(size_t))) {
        PyErr_SetString(PyExc_OverflowError, "Deferred factor scope too wide to materialize.");
        return -1;
    }
    table_size = ((size_t) 1) << (size_t) factor->arity;
    table = PyMem_Malloc((table_size > 0 ? table_size : 1) * sizeof(NativeComplex));
    if (table == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (idx = 0; idx < (Py_ssize_t) table_size; ++idx) {
        table[idx] = native_factor_value_at_index(factor, idx);
    }
    if (factor->aux != NULL) {
        NativeDeferredDominantLocal *deferred = (NativeDeferredDominantLocal *) factor->aux;
        if (deferred->dominant_factor != NULL) {
            free_native_factor(deferred->dominant_factor);
            PyMem_Free(deferred->dominant_factor);
            deferred->dominant_factor = NULL;
        }
        PyMem_Free(deferred->dominant_result_positions);
        PyMem_Free(deferred);
        factor->aux = NULL;
    }
    factor->table = table;
    factor->kind = NATIVE_FACTOR_KIND_CONCRETE;
    return 0;
}


static inline int scopes_equal(const NativeFactor *factor, const Py_ssize_t *scope, Py_ssize_t arity)
{
    Py_ssize_t idx;
    if (!factor->alive || factor->arity != arity) {
        return 0;
    }
    for (idx = 0; idx < arity; ++idx) {
        if (factor->scope[idx] != scope[idx]) {
            return 0;
        }
    }
    return 1;
}


static inline int scaled_scopes_equal(const NativeScaledFactor *factor, const Py_ssize_t *scope, Py_ssize_t arity)
{
    Py_ssize_t idx;
    if (!factor->alive || factor->arity != arity) {
        return 0;
    }
    for (idx = 0; idx < arity; ++idx) {
        if (factor->scope[idx] != scope[idx]) {
            return 0;
        }
    }
    return 1;
}


static void free_native_factor(NativeFactor *factor)
{
    if (factor == NULL) {
        return;
    }
    if (factor->aux != NULL) {
        NativeDeferredDominantLocal *deferred = (NativeDeferredDominantLocal *) factor->aux;
        if (deferred->dominant_factor != NULL) {
            free_native_factor(deferred->dominant_factor);
            PyMem_Free(deferred->dominant_factor);
            deferred->dominant_factor = NULL;
        }
        PyMem_Free(deferred->dominant_result_positions);
        PyMem_Free(deferred);
        factor->aux = NULL;
    }
    PyMem_Free(factor->scope);
    PyMem_Free(factor->table);
    factor->scope = NULL;
    factor->table = NULL;
    factor->arity = 0;
    factor->alive = 0;
    factor->kind = NATIVE_FACTOR_KIND_CONCRETE;
}


static void free_native_scaled_factor(NativeScaledFactor *factor)
{
    if (factor == NULL) {
        return;
    }
    PyMem_Free(factor->scope);
    PyMem_Free(factor->table);
    factor->scope = NULL;
    factor->table = NULL;
    factor->arity = 0;
    factor->alive = 0;
}


static inline int plan_scope_contains_var(const NativePlanScopeFactor *factor, Py_ssize_t var)
{
    Py_ssize_t idx;
    for (idx = 0; idx < factor->arity; ++idx) {
        if (factor->scope[idx] == var) {
            return 1;
        }
    }
    return 0;
}


static inline int plan_scopes_equal(const NativePlanScopeFactor *factor, const Py_ssize_t *scope, Py_ssize_t arity)
{
    Py_ssize_t idx;
    if (!factor->alive || factor->arity != arity) {
        return 0;
    }
    for (idx = 0; idx < arity; ++idx) {
        if (factor->scope[idx] != scope[idx]) {
            return 0;
        }
    }
    return 1;
}


static void free_q3_free_treewidth_plan_step(NativeQ3FreePlanStep *step)
{
    if (step == NULL) {
        return;
    }
    PyMem_Free(step->bucket_slot_ids);
    PyMem_Free(step->bucket_arities);
    PyMem_Free(step->bucket_pos_offsets);
    PyMem_Free(step->bucket_positions);
    PyMem_Free(step->bucket_table_indexes);
    step->bucket_slot_ids = NULL;
    step->bucket_arities = NULL;
    step->bucket_pos_offsets = NULL;
    step->bucket_positions = NULL;
    step->bucket_table_indexes = NULL;
    step->bucket_count = 0;
    step->union_arity = 0;
    step->var_pos = -1;
    step->output_slot = -1;
    step->output_mode = 0;
}


static void free_q3_free_treewidth_plan(NativeQ3FreeTreewidthPlan *plan)
{
    Py_ssize_t idx;
    if (plan == NULL) {
        return;
    }
    if (plan->initial_tables != NULL) {
        for (idx = 0; idx < plan->initial_slot_count; ++idx) {
            PyMem_Free(plan->initial_tables[idx]);
        }
    }
    if (plan->steps != NULL) {
        for (idx = 0; idx < plan->step_count; ++idx) {
            free_q3_free_treewidth_plan_step(&plan->steps[idx]);
        }
    }
    PyMem_Free(plan->slot_arities);
    PyMem_Free(plan->slot_table_sizes);
    PyMem_Free(plan->initial_tables);
    PyMem_Free(plan->steps);
    plan->slot_arities = NULL;
    plan->slot_table_sizes = NULL;
    plan->initial_tables = NULL;
    plan->steps = NULL;
    PyMem_Free(plan);
}


static void q3_free_treewidth_plan_capsule_destructor(PyObject *capsule)
{
    NativeQ3FreeTreewidthPlan *plan = (NativeQ3FreeTreewidthPlan *) PyCapsule_GetPointer(
        capsule,
        q3_free_treewidth_plan_capsule_name
    );
    if (plan != NULL) {
        free_q3_free_treewidth_plan(plan);
    } else {
        PyErr_Clear();
    }
}


static void free_level3_treewidth_plan_step(NativeLevel3PlanStep *step)
{
    if (step == NULL) {
        return;
    }
    PyMem_Free(step->bucket_slot_ids);
    PyMem_Free(step->bucket_arities);
    PyMem_Free(step->bucket_pos_offsets);
    PyMem_Free(step->bucket_positions);
    PyMem_Free(step->bucket_table_indexes);
    step->bucket_slot_ids = NULL;
    step->bucket_arities = NULL;
    step->bucket_pos_offsets = NULL;
    step->bucket_positions = NULL;
    step->bucket_table_indexes = NULL;
    step->bucket_count = 0;
    step->union_arity = 0;
    step->var_pos = -1;
    step->output_slot = -1;
    step->output_mode = 0;
}


static void free_level3_treewidth_plan(NativeLevel3TreewidthPlan *plan)
{
    Py_ssize_t idx;
    if (plan == NULL) {
        return;
    }
    if (plan->initial_tables != NULL) {
        for (idx = 0; idx < plan->initial_slot_count; ++idx) {
            PyMem_Free(plan->initial_tables[idx]);
        }
    }
    if (plan->steps != NULL) {
        for (idx = 0; idx < plan->step_count; ++idx) {
            free_level3_treewidth_plan_step(&plan->steps[idx]);
        }
    }
    PyMem_Free(plan->slot_arities);
    PyMem_Free(plan->slot_table_sizes);
    PyMem_Free(plan->slot_workspace_offsets);
    PyMem_Free(plan->initial_tables);
    PyMem_Free(plan->steps);
    PyMem_Free(plan->workspace);
    PyMem_Free(plan->merge_scratch);
    PyMem_Free(plan->slot_views);
    PyMem_Free(plan->slot_owned);
    plan->slot_arities = NULL;
    plan->slot_table_sizes = NULL;
    plan->slot_workspace_offsets = NULL;
    plan->initial_tables = NULL;
    plan->steps = NULL;
    plan->workspace = NULL;
    plan->merge_scratch = NULL;
    plan->slot_views = NULL;
    plan->slot_owned = NULL;
    PyMem_Free(plan);
}


static void level3_treewidth_plan_capsule_destructor(PyObject *capsule)
{
    NativeLevel3TreewidthPlan *plan = (NativeLevel3TreewidthPlan *) PyCapsule_GetPointer(
        capsule,
        level3_treewidth_plan_capsule_name
    );
    if (plan != NULL) {
        free_level3_treewidth_plan(plan);
    } else {
        PyErr_Clear();
    }
}


static int parse_scaled_complex_entry(PyObject *obj, NativeScaledComplex *out)
{
    PyObject *entry_seq = NULL;
    PyObject *value_obj;
    PyObject *exp_obj;
    Py_complex complex_value;
    long long half_pow2_exp;

    entry_seq = PySequence_Fast(obj, "Scaled-complex entries must be length-2 sequences.");
    if (entry_seq == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(entry_seq) != 2) {
        PyErr_SetString(PyExc_ValueError, "Scaled-complex entries must have length 2.");
        Py_DECREF(entry_seq);
        return -1;
    }

    value_obj = PySequence_Fast_GET_ITEM(entry_seq, 0);
    exp_obj = PySequence_Fast_GET_ITEM(entry_seq, 1);
    complex_value = PyComplex_AsCComplex(value_obj);
    if (complex_value.real == -1.0 && complex_value.imag == 0.0 && PyErr_Occurred()) {
        Py_DECREF(entry_seq);
        return -1;
    }
    half_pow2_exp = PyLong_AsLongLong(exp_obj);
    if (half_pow2_exp == -1 && PyErr_Occurred()) {
        Py_DECREF(entry_seq);
        return -1;
    }

    *out = normalize_scaled_complex(
        complex_make(complex_value.real, complex_value.imag),
        half_pow2_exp
    );
    Py_DECREF(entry_seq);
    return 0;
}


static NativeComplex omega_level3_value(int residue)
{
    switch (residue & 7) {
        case 0:
            return complex_make(1.0, 0.0);
        case 1:
            return complex_make(SQRT1_2, SQRT1_2);
        case 2:
            return complex_make(0.0, 1.0);
        case 3:
            return complex_make(-SQRT1_2, SQRT1_2);
        case 4:
            return complex_make(-1.0, 0.0);
        case 5:
            return complex_make(-SQRT1_2, -SQRT1_2);
        case 6:
            return complex_make(0.0, -1.0);
        default:
            return complex_make(SQRT1_2, -SQRT1_2);
    }
}


static NativeComplex omega_dyadic_value(long long residue, long long modulus)
{
    static const double two_pi = 6.28318530717958647692528676655900576839;
    double angle;
    residue = positive_mod_ll(residue, modulus);
    if (residue == 0) {
        return complex_make(1.0, 0.0);
    }
    angle = two_pi * ((double) residue / (double) modulus);
    return complex_make(cos(angle), sin(angle));
}


static Py_ssize_t bitset_count(const uint64_t *bits, Py_ssize_t nwords)
{
    Py_ssize_t count = 0;
    Py_ssize_t idx;
    for (idx = 0; idx < nwords; ++idx) {
        count += popcount_u64(bits[idx]);
    }
    return count;
}


static Py_ssize_t bitset_pop_first(uint64_t *bits, Py_ssize_t nwords)
{
    Py_ssize_t word_idx;
    for (word_idx = 0; word_idx < nwords; ++word_idx) {
        uint64_t word = bits[word_idx];
        if (word != 0) {
            int bit_idx = lowest_bit_index(word);
            bits[word_idx] = word & (word - 1);
            return (word_idx << 6) + bit_idx;
        }
    }
    return -1;
}


static int mark_true(PyObject *values, Py_ssize_t index)
{
    PyObject *current = PyList_GET_ITEM(values, index);
    if (current == Py_True) {
        return 0;
    }
    Py_INCREF(Py_True);
    PyList_SET_ITEM(values, index, Py_True);
    Py_DECREF(current);
    return 0;
}


static int parse_pair_key(PyObject *key, Py_ssize_t *left, Py_ssize_t *right)
{
    if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) != 2) {
        PyErr_SetString(PyExc_TypeError, "q2 keys must be 2-tuples.");
        return -1;
    }
    *left = PyLong_AsSsize_t(PyTuple_GET_ITEM(key, 0));
    if (*left == -1 && PyErr_Occurred()) {
        return -1;
    }
    *right = PyLong_AsSsize_t(PyTuple_GET_ITEM(key, 1));
    if (*right == -1 && PyErr_Occurred()) {
        return -1;
    }
    return 0;
}


static int parse_triple_key(PyObject *key, Py_ssize_t *a, Py_ssize_t *b, Py_ssize_t *c)
{
    if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) != 3) {
        PyErr_SetString(PyExc_TypeError, "q3 keys must be 3-tuples.");
        return -1;
    }
    *a = PyLong_AsSsize_t(PyTuple_GET_ITEM(key, 0));
    if (*a == -1 && PyErr_Occurred()) {
        return -1;
    }
    *b = PyLong_AsSsize_t(PyTuple_GET_ITEM(key, 1));
    if (*b == -1 && PyErr_Occurred()) {
        return -1;
    }
    *c = PyLong_AsSsize_t(PyTuple_GET_ITEM(key, 2));
    if (*c == -1 && PyErr_Occurred()) {
        return -1;
    }
    return 0;
}


static PyObject *support_from_mask_native(PyObject *self, PyObject *arg)
{
    PyObject *mask = NULL;
    PyObject *tuple = NULL;

    (void) self;

    mask = PyNumber_Index(arg);
    if (mask == NULL) {
        return NULL;
    }
    tuple = get_support_from_cache(NULL, mask);
    Py_DECREF(mask);
    return tuple;
}


static PyObject *support_tuple_from_mask(PyObject *mask)
{
    PyObject *bit_length_obj = NULL;
    PyObject *little = NULL;
    PyObject *bytes_obj = NULL;
    PyObject *tuple = NULL;
    const unsigned char *data;
    int truthy;
    Py_ssize_t bit_length;
    Py_ssize_t byte_len;
    Py_ssize_t count = 0;
    Py_ssize_t byte_idx;
    Py_ssize_t out_idx = 0;
    truthy = PyObject_IsTrue(mask);
    if (truthy < 0) {
        goto error;
    }
    if (!truthy) {
        return PyTuple_New(0);
    }

    bit_length_obj = PyObject_CallMethod(mask, "bit_length", NULL);
    if (bit_length_obj == NULL) {
        goto error;
    }
    bit_length = PyLong_AsSsize_t(bit_length_obj);
    if (bit_length < 0 && PyErr_Occurred()) {
        goto error;
    }
    byte_len = (bit_length + 7) >> 3;

    little = ensure_little_endian_string();
    if (little == NULL) {
        goto error;
    }
    bytes_obj = PyObject_CallMethod(mask, "to_bytes", "nO", byte_len, little);
    if (bytes_obj == NULL) {
        goto error;
    }

    data = (const unsigned char *) PyBytes_AS_STRING(bytes_obj);
    for (byte_idx = 0; byte_idx < byte_len; ++byte_idx) {
        count += popcount_u64((uint64_t) data[byte_idx]);
    }

    tuple = PyTuple_New(count);
    if (tuple == NULL) {
        goto error;
    }

    for (byte_idx = 0; byte_idx < byte_len; ++byte_idx) {
        unsigned char byte = data[byte_idx];
        int bit;
        if (byte == 0) {
            continue;
        }
        for (bit = 0; bit < 8; ++bit) {
            if (byte & (1U << bit)) {
                PyObject *value = PyLong_FromSsize_t((byte_idx << 3) + bit);
                if (value == NULL) {
                    goto error;
                }
                PyTuple_SET_ITEM(tuple, out_idx, value);
                ++out_idx;
            }
        }
    }

    Py_DECREF(bit_length_obj);
    Py_DECREF(little);
    Py_DECREF(bytes_obj);
    return tuple;

error:
    Py_XDECREF(bit_length_obj);
    Py_XDECREF(little);
    Py_XDECREF(bytes_obj);
    Py_XDECREF(tuple);
    return NULL;
}


static PyObject *get_support_from_cache(PyObject *cache, PyObject *mask)
{
    PyObject *support = NULL;
    PyObject *global_cache = NULL;

    if (cache != NULL) {
        support = PyDict_GetItemWithError(cache, mask);
        if (support != NULL) {
            Py_INCREF(support);
            return support;
        }
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    global_cache = ensure_global_support_cache();
    if (global_cache == NULL) {
        return NULL;
    }
    if (cache != global_cache) {
        support = PyDict_GetItemWithError(global_cache, mask);
        if (support != NULL) {
            if (cache != NULL && PyDict_SetItem(cache, mask, support) < 0) {
                return NULL;
            }
            Py_INCREF(support);
            return support;
        }
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    support = support_tuple_from_mask(mask);
    if (support == NULL) {
        return NULL;
    }
    if (cache != NULL && PyDict_SetItem(cache, mask, support) < 0) {
        Py_DECREF(support);
        return NULL;
    }
    if (cache != global_cache) {
        if (PyDict_GET_SIZE(global_cache) >= SUPPORT_CACHE_MAX_SIZE) {
            PyDict_Clear(global_cache);
        }
        if (PyDict_SetItem(global_cache, mask, support) < 0) {
            Py_DECREF(support);
            return NULL;
        }
    }
    return support;
}


static PyObject *clear_support_cache_native(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *cache;

    (void) self;

    cache = ensure_global_support_cache();
    if (cache == NULL) {
        return NULL;
    }
    PyDict_Clear(cache);
    Py_RETURN_NONE;
}


