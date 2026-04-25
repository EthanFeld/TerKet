#include "_schur_native_internal.h"

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
        "elim_two_partner_constraint_terms",
        (PyCFunction) elim_two_partner_constraint_terms_native,
        METH_VARARGS,
        PyDoc_STR("Eliminate a two-partner parity constraint in a level-3 cubic kernel."),
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
