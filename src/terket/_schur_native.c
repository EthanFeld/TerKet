#include "_schur_native_internal.h"

PyObject *sum_residue_forest_batch_scaled_array_native(PyObject *self, PyObject *args);

#define NATIVE_METHOD(name, func, flags, doc) \
    {name, (PyCFunction) func, flags, PyDoc_STR(doc)}

static PyMethodDef module_methods[] = {
    NATIVE_METHOD("support_from_mask", support_from_mask_native, METH_O, "Support tuple for an int mask."),
    NATIVE_METHOD("build_classification_data", build_classification_data_native, METH_VARARGS, "Build cubic, odd, and parity data."),
    NATIVE_METHOD("classification_structure_key", classification_structure_key_native, METH_VARARGS, "Serialize q2/q3 structure for caches."),
    NATIVE_METHOD("build_classification_lookup", build_classification_lookup_native, METH_VARARGS, "Build the classification lookup table."),
    NATIVE_METHOD("evaluate_q_mask_terms", evaluate_q_mask_terms_native, METH_VARARGS, "Evaluate q1/q2/q3 mask terms in eighths."),
    NATIVE_METHOD("aff_compose_terms", aff_compose_terms_native, METH_VARARGS, "Compose affine q1/q2/q3 terms."),
    NATIVE_METHOD("solve_output_shift_mask_u64", solve_output_shift_mask_u64_native, METH_VARARGS, "Solve one 64-bit output shift mask."),
    NATIVE_METHOD("solve_output_shift_masks_u64", solve_output_shift_masks_u64_native, METH_VARARGS, "Solve a batch of 64-bit output shift masks."),
    NATIVE_METHOD("elim_single_partner_constraint_terms", elim_single_partner_constraint_terms_native, METH_VARARGS, "Eliminate one parity partner."),
    NATIVE_METHOD("elim_two_partner_constraint_terms", elim_two_partner_constraint_terms_native, METH_VARARGS, "Eliminate a two-partner parity constraint."),
    NATIVE_METHOD("elim_sparse_quadratics_batch_terms", elim_sparse_quadratics_batch_terms_native, METH_VARARGS, "Batch-eliminate sparse level-3 quadratic pivots."),
    NATIVE_METHOD("clear_support_cache", clear_support_cache_native, METH_NOARGS, "Clear the native support cache."),
    NATIVE_METHOD("min_fill_cubic_order", min_fill_cubic_order_native, METH_VARARGS, "Return a min-fill order and width."),
    NATIVE_METHOD("min_degree_cubic_order", min_degree_cubic_order_native, METH_VARARGS, "Return a min-degree order and width."),
    NATIVE_METHOD("cubic_order_width", cubic_order_width_native, METH_VARARGS, "Return width for a fixed order."),
    NATIVE_METHOD("rank_q3_free_cutset_extensions", rank_q3_free_cutset_extensions_native, METH_VARARGS, "Rank q3-free cutset extensions."),
    NATIVE_METHOD("sum_treewidth_dp_level3", sum_treewidth_dp_level3_native, METH_VARARGS, "Sum a level-3 kernel by elimination."),
    NATIVE_METHOD("build_level3_treewidth_plan", build_level3_treewidth_plan_native, METH_VARARGS, "Build a reusable level-3 plan."),
    NATIVE_METHOD("sum_level3_treewidth_preplanned", sum_level3_treewidth_preplanned_native, METH_VARARGS, "Evaluate a level-3 kernel with a plan."),
    NATIVE_METHOD("sum_level3_treewidth_preplanned_batch_array", sum_level3_treewidth_preplanned_batch_array_native, METH_VARARGS, "Evaluate a batch of level-3 kernels with a plan."),
    NATIVE_METHOD("sum_factor_tables_scaled", sum_factor_tables_scaled_native, METH_VARARGS, "Sum scaled factor tables by elimination."),
    NATIVE_METHOD("build_phase_function_treewidth_support_plan", build_phase_function_treewidth_support_plan_native, METH_VARARGS, "Build a reusable q2/q3 support plan."),
    NATIVE_METHOD("build_scaled_factor_treewidth_plan", build_scaled_factor_treewidth_plan_native, METH_VARARGS, "Build a reusable scaled-factor plan."),
    NATIVE_METHOD("sum_scaled_factor_treewidth_preplanned", sum_scaled_factor_treewidth_preplanned_native, METH_VARARGS, "Evaluate scaled factors with a plan."),
    NATIVE_METHOD("sum_phase_function_treewidth_preplanned_batch_scaled_array", sum_phase_function_treewidth_preplanned_batch_scaled_array_native, METH_VARARGS, "Evaluate a batch of phase-function rows with a plan."),
    NATIVE_METHOD("build_q3_free_treewidth_plan", build_q3_free_treewidth_plan_native, METH_VARARGS, "Build a reusable q3-free plan."),
    NATIVE_METHOD("sum_q3_free_treewidth_preplanned_batch_scaled", sum_q3_free_treewidth_preplanned_batch_scaled_native, METH_VARARGS, "Evaluate a batch of q3-free kernels with a plan."),
    NATIVE_METHOD("sum_q3_free_treewidth_preplanned_batch_scaled_array", sum_q3_free_treewidth_preplanned_batch_scaled_array_native, METH_VARARGS, "Evaluate a contiguous q1 batch with a q3-free plan."),
    NATIVE_METHOD("sum_residue_forest_batch_scaled_array", sum_residue_forest_batch_scaled_array_native, METH_VARARGS, "Evaluate arbitrary-residue forest rows."),
    NATIVE_METHOD("q3_free_treewidth_dp_work", q3_free_treewidth_dp_work_native, METH_VARARGS, "Estimate q3-free work for an order."),
    NATIVE_METHOD("sum_q3_free_treewidth_batch_scaled", sum_q3_free_treewidth_batch_scaled_native, METH_VARARGS, "Sum a q3-free batch by treewidth DP."),
    {NULL, NULL, 0, NULL},
};

#undef NATIVE_METHOD

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
