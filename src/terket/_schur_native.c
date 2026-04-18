#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define SUPPORT_CACHE_MAX_SIZE 65536
#define SQRT1_2 0.70710678118654752440
#define SQRT2 1.41421356237309504880
#define SCALED_RENORMALIZE_MIN 8.63616855509444462539e-78
#define SCALED_RENORMALIZE_MAX 1.15792089237316195424e+77

static PyObject *support_tuple_from_mask(PyObject *mask);
static PyObject *get_support_from_cache(PyObject *cache, PyObject *mask);
static PyObject *global_support_cache = NULL;
static PyObject *little_endian_string = NULL;
static int load_binary_flags(PyObject *seq_obj, const char *message, unsigned char **out_bits, Py_ssize_t *out_len);


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


static const char *q3_free_treewidth_plan_capsule_name = "terket._schur_native.q3_free_treewidth_plan";

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


static const char *level3_treewidth_plan_capsule_name = "terket._schur_native.level3_treewidth_plan";



/* Low-level support helpers, caches, and scalar utilities. */
#include "_schur_native_support.inc"

/* Affine composition, classification, and output-shift routines. */
#include "_schur_native_algebra.inc"

/* Graph order heuristics and width evaluation helpers. */
#include "_schur_native_graph.inc"

/* Exact treewidth-based DP kernels and reusable q3-free plans. */
#include "_schur_native_dp.inc"

static PyMethodDef module_methods[] = {
    {
        "support_from_mask",
        (PyCFunction) support_from_mask_native,
        METH_O,
        PyDoc_STR("Return the support tuple of a Python integer bitmask."),
    },
    {
        "build_classification_data",
        (PyCFunction) build_classification_data_native,
        METH_VARARGS,
        PyDoc_STR("Build cubic incidence, odd bilinear flags, and parity partners."),
    },
    {
        "classification_structure_key",
        (PyCFunction) classification_structure_key_native,
        METH_VARARGS,
        PyDoc_STR("Serialize the parity-relevant q2/q3 structure used by classification caches."),
    },
    {
        "build_classification_lookup",
        (PyCFunction) build_classification_lookup_native,
        METH_VARARGS,
        PyDoc_STR("Build the full variable-classification lookup table from classification data."),
    },
    {
        "evaluate_q_mask_terms",
        (PyCFunction) evaluate_q_mask_terms_native,
        METH_VARARGS,
        PyDoc_STR("Return the non-constant q1/q2/q3 contribution on a bitmask in eighths."),
    },
    {
        "aff_compose_terms",
        (PyCFunction) aff_compose_terms_native,
        METH_VARARGS,
        PyDoc_STR("Compose affine q1/q2/q3 terms into a new cubic phase kernel."),
    },
    {
        "solve_output_shift_mask_u64",
        (PyCFunction) solve_output_shift_mask_u64_native,
        METH_VARARGS,
        PyDoc_STR("Solve one output constraint system with 64-bit masks, returning the pivot shift mask or None."),
    },
    {
        "solve_output_shift_masks_u64",
        (PyCFunction) solve_output_shift_masks_u64_native,
        METH_VARARGS,
        PyDoc_STR("Solve a batch of output constraint systems with 64-bit masks."),
    },
    {
        "elim_single_partner_constraint_terms",
        (PyCFunction) elim_single_partner_constraint_terms_native,
        METH_VARARGS,
        PyDoc_STR("Eliminate one parity-constraint partner in a level-3 cubic kernel."),
    },
    {
        "clear_support_cache",
        (PyCFunction) clear_support_cache_native,
        METH_NOARGS,
        PyDoc_STR("Clear the native support-mask cache."),
    },
    {
        "min_fill_cubic_order",
        (PyCFunction) min_fill_cubic_order_native,
        METH_VARARGS,
        PyDoc_STR("Return a min-fill style elimination order and its maximum scope."),
    },
    {
        "min_degree_cubic_order",
        (PyCFunction) min_degree_cubic_order_native,
        METH_VARARGS,
        PyDoc_STR("Return a min-degree style elimination order and its maximum scope."),
    },
    {
        "cubic_order_width",
        (PyCFunction) cubic_order_width_native,
        METH_VARARGS,
        PyDoc_STR("Return the maximum induced scope size for a fixed elimination order."),
    },
    {
        "sum_treewidth_dp_level3",
        (PyCFunction) sum_treewidth_dp_level3_native,
        METH_VARARGS,
        PyDoc_STR("Sum a level-3 cubic kernel by low-width variable elimination."),
    },
    {
        "build_level3_treewidth_plan",
        (PyCFunction) build_level3_treewidth_plan_native,
        METH_VARARGS,
        PyDoc_STR("Build a reusable native treewidth plan for a fixed level-3 cubic kernel."),
    },
    {
        "sum_level3_treewidth_preplanned",
        (PyCFunction) sum_level3_treewidth_preplanned_native,
        METH_VARARGS,
        PyDoc_STR("Evaluate a fixed level-3 cubic kernel with a reusable native treewidth plan."),
    },
    {
        "sum_factor_tables_scaled",
        (PyCFunction) sum_factor_tables_scaled_native,
        METH_VARARGS,
        PyDoc_STR("Sum generic scaled-complex factor tables by exact variable elimination."),
    },
    {
        "build_scaled_factor_treewidth_plan",
        (PyCFunction) build_scaled_factor_treewidth_plan_native,
        METH_VARARGS,
        PyDoc_STR("Build a reusable native treewidth plan for fixed scaled factor tables."),
    },
    {
        "sum_scaled_factor_treewidth_preplanned",
        (PyCFunction) sum_scaled_factor_treewidth_preplanned_native,
        METH_VARARGS,
        PyDoc_STR("Evaluate fixed scaled factor tables with a reusable native treewidth plan."),
    },
    {
        "build_q3_free_treewidth_plan",
        (PyCFunction) build_q3_free_treewidth_plan_native,
        METH_VARARGS,
        PyDoc_STR("Build a reusable native plan for q3-free exact treewidth DP."),
    },
    {
        "sum_q3_free_treewidth_preplanned_batch_scaled",
        (PyCFunction) sum_q3_free_treewidth_preplanned_batch_scaled_native,
        METH_VARARGS,
        PyDoc_STR("Evaluate a batch of q3-free kernels with a reusable native treewidth plan."),
    },
    {
        "sum_q3_free_treewidth_preplanned_batch_scaled_array",
        (PyCFunction) sum_q3_free_treewidth_preplanned_batch_scaled_array_native,
        METH_VARARGS,
        PyDoc_STR("Evaluate a contiguous int64 q1 batch with a reusable native q3-free treewidth plan."),
    },
    {
        "q3_free_treewidth_dp_work",
        (PyCFunction) q3_free_treewidth_dp_work_native,
        METH_VARARGS,
        PyDoc_STR("Estimate q3-free treewidth-DP work along a fixed order."),
    },
    {
        "sum_q3_free_treewidth_batch_scaled",
        (PyCFunction) sum_q3_free_treewidth_batch_scaled_native,
        METH_VARARGS,
        PyDoc_STR("Sum a batch of q3-free kernels with shared q2/order by native exact treewidth DP."),
    },
    {NULL, NULL, 0, NULL},
};


static void module_free(void *module)
{
    (void) module;
    Py_CLEAR(global_support_cache);
    Py_CLEAR(little_endian_string);
}


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_schur_native",
    "Native reducer helpers for TerKet.",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    module_free,
};


PyMODINIT_FUNC PyInit__schur_native(void)
{
    return PyModule_Create(&module_def);
}
