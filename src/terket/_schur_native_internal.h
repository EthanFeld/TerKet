#ifndef TERKET_SCHUR_NATIVE_INTERNAL_H
#define TERKET_SCHUR_NATIVE_INTERNAL_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SUPPORT_CACHE_MAX_SIZE 65536
#define SQRT1_2 0.70710678118654752440
#define SQRT2 1.41421356237309504880
#define SCALED_RENORMALIZE_MIN 8.63616855509444462539e-78
#define SCALED_RENORMALIZE_MAX 1.15792089237316195424e+77

typedef struct {
    double real;
    double imag;
} NativeComplex;

typedef struct {
    Py_ssize_t arity;
    Py_ssize_t *scope;
    NativeComplex *table;
    int alive;
    unsigned char kind;
    void *aux;
} NativeFactor;

enum {
    NATIVE_FACTOR_KIND_CONCRETE = 0,
    NATIVE_FACTOR_KIND_DEFERRED_DOMLOCAL = 1,
};

typedef struct {
    unsigned char missing_pos;
    unsigned char dominant_var_pos;
    unsigned char dominant_arity;
    unsigned char local_support_arity;
    signed char *dominant_result_positions;
    unsigned char local_result_positions[4];
    NativeComplex local_products[2][16];
    NativeFactor *dominant_factor;
} NativeDeferredDominantLocal;

typedef struct {
    NativeComplex value;
    long long half_pow2_exp;
} NativeScaledComplex;

typedef struct {
    Py_ssize_t arity;
    Py_ssize_t *scope;
    NativeScaledComplex *table;
    int alive;
} NativeScaledFactor;

typedef struct {
    Py_ssize_t left;
    Py_ssize_t right;
    unsigned char flags;
} ClassificationPairRecord;

typedef struct {
    Py_ssize_t a;
    Py_ssize_t b;
    Py_ssize_t c;
} ClassificationTripleRecord;

typedef struct {
    Py_ssize_t left;
    Py_ssize_t right;
    long long shift;
} Q2PhaseRecord;

typedef struct {
    Py_ssize_t arity;
    Py_ssize_t *scope;
    int alive;
} NativePlanScopeFactor;

typedef struct {
    Py_ssize_t var;
    Py_ssize_t bucket_count;
    Py_ssize_t union_arity;
    Py_ssize_t var_pos;
    uint64_t low_mask;
    size_t reduced_table_size;
    Py_ssize_t *bucket_slot_ids;
    Py_ssize_t *bucket_arities;
    Py_ssize_t *bucket_pos_offsets;
    Py_ssize_t *bucket_positions;
    Py_ssize_t *bucket_table_indexes;
    int output_mode;
    Py_ssize_t output_slot;
} NativeQ3FreePlanStep;

typedef struct {
    Py_ssize_t nvars;
    long level;
    long long mod_q1;
    Py_ssize_t initial_slot_count;
    Py_ssize_t slot_count;
    Py_ssize_t step_count;
    Py_ssize_t max_scope;
    Py_ssize_t *slot_arities;
    size_t *slot_table_sizes;
    NativeScaledComplex **initial_tables;
    NativeQ3FreePlanStep *steps;
} NativeQ3FreeTreewidthPlan;

typedef struct {
    Py_ssize_t var;
    Py_ssize_t bucket_count;
    Py_ssize_t union_arity;
    Py_ssize_t var_pos;
    uint64_t low_mask;
    size_t reduced_table_size;
    Py_ssize_t *bucket_slot_ids;
    Py_ssize_t *bucket_arities;
    Py_ssize_t *bucket_pos_offsets;
    Py_ssize_t *bucket_positions;
    Py_ssize_t *bucket_table_indexes;
    int output_mode;
    Py_ssize_t output_slot;
} NativeLevel3PlanStep;

typedef struct {
    Py_ssize_t nvars;
    Py_ssize_t initial_slot_count;
    Py_ssize_t slot_count;
    Py_ssize_t step_count;
    Py_ssize_t max_scope;
    Py_ssize_t *slot_arities;
    size_t *slot_table_sizes;
    size_t *slot_workspace_offsets;
    NativeComplex **initial_tables;
    NativeLevel3PlanStep *steps;
    NativeComplex *workspace;
    size_t workspace_size;
    NativeComplex *merge_scratch;
    size_t merge_scratch_size;
    NativeComplex **slot_views;
    unsigned char *slot_owned;
} NativeLevel3TreewidthPlan;

extern PyObject *global_support_cache;
extern PyObject *little_endian_string;
extern const char *q3_free_treewidth_plan_capsule_name;
extern const char *level3_treewidth_plan_capsule_name;

size_t word_index(Py_ssize_t bit);
void bitset_set(uint64_t *bits, Py_ssize_t bit);
void bitset_clear(uint64_t *bits, Py_ssize_t bit);
Py_ssize_t popcount_u64(uint64_t word);
int lowest_bit_index(uint64_t word);
NativeComplex complex_make(double real, double imag);
NativeComplex complex_add(NativeComplex left, NativeComplex right);
NativeComplex complex_mul(NativeComplex left, NativeComplex right);
NativeScaledComplex scaled_complex_make(double real, double imag, long long half_pow2_exp);
NativeScaledComplex renormalize_scaled_complex_if_needed(NativeComplex value, long long half_pow2_exp);
NativeScaledComplex add_scaled_complex(NativeScaledComplex left, NativeScaledComplex right);
NativeScaledComplex mul_scaled_complex(NativeScaledComplex left, NativeScaledComplex right);
int positive_mod(long value, int modulus);
long long positive_mod_ll(long long value, long long modulus);
void write_u64_le(unsigned char *dest, uint64_t value);
int factor_contains_var(const NativeFactor *factor, Py_ssize_t var);
Py_ssize_t project_assignment_bits_u64(uint64_t assignment, const Py_ssize_t *positions, Py_ssize_t count);
NativeComplex native_factor_value_at_index(const NativeFactor *factor, Py_ssize_t assignment_idx);
int materialize_native_factor(NativeFactor *factor);
int scopes_equal(const NativeFactor *factor, const Py_ssize_t *scope, Py_ssize_t arity);
int scaled_scopes_equal(const NativeScaledFactor *factor, const Py_ssize_t *scope, Py_ssize_t arity);
void free_native_factor(NativeFactor *factor);
void free_native_scaled_factor(NativeScaledFactor *factor);
int plan_scope_contains_var(const NativePlanScopeFactor *factor, Py_ssize_t var);
int plan_scopes_equal(const NativePlanScopeFactor *factor, const Py_ssize_t *scope, Py_ssize_t arity);
void free_q3_free_treewidth_plan(NativeQ3FreeTreewidthPlan *plan);
void free_level3_treewidth_plan(NativeLevel3TreewidthPlan *plan);
void q3_free_treewidth_plan_capsule_destructor(PyObject *capsule);
void level3_treewidth_plan_capsule_destructor(PyObject *capsule);
int parse_scaled_complex_entry(PyObject *obj, NativeScaledComplex *out);
NativeComplex omega_level3_value(int residue);
NativeComplex omega_dyadic_value(long long residue, long long modulus);
Py_ssize_t bitset_count(const uint64_t *bits, Py_ssize_t nwords);
Py_ssize_t bitset_pop_first(uint64_t *bits, Py_ssize_t nwords);
int compare_classification_pair_records(const void *left_ptr, const void *right_ptr);
int compare_classification_triple_records(const void *left_ptr, const void *right_ptr);
int mark_true(PyObject *values, Py_ssize_t index);
int parse_pair_key(PyObject *key, Py_ssize_t *left, Py_ssize_t *right);
int parse_triple_key(PyObject *key, Py_ssize_t *a, Py_ssize_t *b, Py_ssize_t *c);
PyObject *support_tuple_from_mask(PyObject *mask);
PyObject *get_support_from_cache(PyObject *cache, PyObject *mask);

PyObject *support_from_mask_native(PyObject *self, PyObject *arg);
PyObject *clear_support_cache_native(PyObject *self, PyObject *ignored);
PyObject *evaluate_q_mask_terms_native(PyObject *self, PyObject *args);
PyObject *elim_single_partner_constraint_terms_native(PyObject *self, PyObject *args);
PyObject *aff_compose_terms_native(PyObject *self, PyObject *args);
PyObject *build_classification_data_native(PyObject *self, PyObject *args);
PyObject *classification_structure_key_native(PyObject *self, PyObject *args);
PyObject *build_classification_lookup_native(PyObject *self, PyObject *args);
PyObject *solve_output_shift_mask_u64_native(PyObject *self, PyObject *args);
PyObject *solve_output_shift_masks_u64_native(PyObject *self, PyObject *args);
PyObject *min_fill_cubic_order_native(PyObject *self, PyObject *args);
PyObject *min_degree_cubic_order_native(PyObject *self, PyObject *args);
PyObject *cubic_order_width_native(PyObject *self, PyObject *args);
PyObject *sum_treewidth_dp_level3_native(PyObject *self, PyObject *args);
PyObject *build_level3_treewidth_plan_native(PyObject *self, PyObject *args);
PyObject *sum_level3_treewidth_preplanned_native(PyObject *self, PyObject *args);
PyObject *sum_factor_tables_scaled_native(PyObject *self, PyObject *args);
PyObject *build_q3_free_treewidth_plan_native(PyObject *self, PyObject *args);
PyObject *build_scaled_factor_treewidth_plan_native(PyObject *self, PyObject *args);
PyObject *sum_scaled_factor_treewidth_preplanned_native(PyObject *self, PyObject *args);
PyObject *sum_q3_free_treewidth_preplanned_batch_scaled_native(PyObject *self, PyObject *args);
PyObject *sum_q3_free_treewidth_preplanned_batch_scaled_array_native(PyObject *self, PyObject *args);
PyObject *q3_free_treewidth_dp_work_native(PyObject *self, PyObject *args);
PyObject *sum_q3_free_treewidth_batch_scaled_native(PyObject *self, PyObject *args);

#endif
