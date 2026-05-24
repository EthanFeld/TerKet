#include "_schur_native_internal.h"
#include "native_binary_loaders.inc"

static void free_output_solver_inputs(unsigned char *eps0, Py_ssize_t *pivot_col, uint64_t *row_ops)
{
    PyMem_Free(eps0);
    PyMem_Free(pivot_col);
    PyMem_Free(row_ops);
}

static int load_output_solver_inputs(PyObject *eps0_obj, PyObject *pivot_col_obj, PyObject *row_ops_obj, Py_ssize_t mvars, unsigned char **eps0, Py_ssize_t **pivot_col, uint64_t **row_ops, Py_ssize_t *nbits)
{
    if (load_binary_flags(eps0_obj, "eps0 must be a sequence.", eps0, nbits) < 0) {
        return -1;
    }
    if (*nbits > 64 || mvars > 64) {
        PyErr_SetString(PyExc_ValueError, "Native u64 output solving requires n <= 64 and m <= 64.");
        return -1;
    }
    if (load_ssize_array(pivot_col_obj, "pivot_col must be a sequence.", pivot_col, *nbits) < 0) {
        return -1;
    }
    if (load_u64_array(row_ops_obj, "row_ops must be a sequence.", row_ops, *nbits) < 0) {
        return -1;
    }
    return 0;
}

static int solve_shift_mask_from_target(uint64_t target_mask, const Py_ssize_t *pivot_col, const uint64_t *row_ops, Py_ssize_t nbits, uint64_t *shift_mask_out)
{
    uint64_t shift_mask = 0ULL;
    Py_ssize_t row_idx;

    for (row_idx = 0; row_idx < nbits; ++row_idx) {
        if ((popcount_u64(target_mask & row_ops[row_idx]) & 1) == 0) {
            continue;
        }
        if (pivot_col[row_idx] < 0) {
            return 0;
        }
        shift_mask |= 1ULL << (size_t) pivot_col[row_idx];
    }

    *shift_mask_out = shift_mask;
    return 1;
}

PyObject *solve_output_shift_mask_u64_native(PyObject *self, PyObject *args)
{
    PyObject *eps0_obj;
    PyObject *pivot_col_obj;
    PyObject *row_ops_obj;
    PyObject *output_bits_obj;
    Py_ssize_t mvars;
    unsigned char *eps0 = NULL;
    Py_ssize_t *pivot_col = NULL;
    uint64_t *row_ops = NULL;
    unsigned char *output_bits = NULL;
    Py_ssize_t nbits;
    Py_ssize_t output_len;
    PyObject *result = NULL;
    uint64_t target_mask = 0ULL;
    uint64_t shift_mask = 0ULL;
    Py_ssize_t row_idx;

    (void) self;

    if (!PyArg_ParseTuple(args, "OOOOn", &eps0_obj, &pivot_col_obj, &row_ops_obj, &output_bits_obj, &mvars)) {
        return NULL;
    }

    if (load_output_solver_inputs(eps0_obj, pivot_col_obj, row_ops_obj, mvars, &eps0, &pivot_col, &row_ops, &nbits) < 0) {
        goto error;
    }
    if (load_binary_flags(output_bits_obj, "output_bits must be a sequence.", &output_bits, &output_len) < 0) {
        goto error;
    }
    if (output_len != nbits) {
        PyErr_SetString(PyExc_ValueError, "output_bits length mismatch.");
        goto error;
    }

    for (row_idx = 0; row_idx < nbits; ++row_idx) {
        if (eps0[row_idx] != output_bits[row_idx]) {
            target_mask |= 1ULL << (size_t) row_idx;
        }
    }

    if (solve_shift_mask_from_target(target_mask, pivot_col, row_ops, nbits, &shift_mask)) {
        result = PyLong_FromUnsignedLongLong((unsigned long long) shift_mask);
    } else {
        Py_INCREF(Py_None);
        result = Py_None;
    }

    PyMem_Free(output_bits);
    free_output_solver_inputs(eps0, pivot_col, row_ops);
    return result;

error:
    PyMem_Free(output_bits);
    free_output_solver_inputs(eps0, pivot_col, row_ops);
    return NULL;
}

PyObject *solve_output_shift_masks_u64_native(PyObject *self, PyObject *args)
{
    PyObject *eps0_obj;
    PyObject *pivot_col_obj;
    PyObject *row_ops_obj;
    PyObject *outputs_obj;
    Py_ssize_t mvars;
    unsigned char *eps0 = NULL;
    Py_ssize_t *pivot_col = NULL;
    uint64_t *row_ops = NULL;
    Py_ssize_t nbits;
    PyObject *outputs_seq = NULL;
    PyObject *result = NULL;
    Py_ssize_t output_count;
    Py_ssize_t output_idx;

    (void) self;

    if (!PyArg_ParseTuple(args, "OOOOn", &eps0_obj, &pivot_col_obj, &row_ops_obj, &outputs_obj, &mvars)) {
        return NULL;
    }

    if (load_output_solver_inputs(eps0_obj, pivot_col_obj, row_ops_obj, mvars, &eps0, &pivot_col, &row_ops, &nbits) < 0) {
        goto error;
    }

    outputs_seq = PySequence_Fast(outputs_obj, "outputs must be a sequence.");
    if (outputs_seq == NULL) {
        goto error;
    }
    output_count = PySequence_Fast_GET_SIZE(outputs_seq);
    result = PyTuple_New(output_count);
    if (result == NULL) {
        goto error;
    }

    for (output_idx = 0; output_idx < output_count; ++output_idx) {
        PyObject *output_seq = NULL;
        Py_ssize_t output_len;
        Py_ssize_t row_idx;
        uint64_t target_mask = 0ULL;
        uint64_t shift_mask = 0ULL;
        PyObject *item = NULL;

        output_seq = PySequence_Fast(PySequence_Fast_GET_ITEM(outputs_seq, output_idx), "Each output must be a sequence.");
        if (output_seq == NULL) {
            goto error;
        }
        output_len = PySequence_Fast_GET_SIZE(output_seq);
        if (output_len != nbits) {
            Py_DECREF(output_seq);
            PyErr_SetString(PyExc_ValueError, "output_bits length mismatch.");
            goto error;
        }
        for (row_idx = 0; row_idx < nbits; ++row_idx) {
            int truthy = PyObject_IsTrue(PySequence_Fast_GET_ITEM(output_seq, row_idx));
            if (truthy < 0) {
                Py_DECREF(output_seq);
                goto error;
            }
            if (eps0[row_idx] != (truthy ? 1U : 0U)) {
                target_mask |= 1ULL << (size_t) row_idx;
            }
        }

        if (solve_shift_mask_from_target(target_mask, pivot_col, row_ops, nbits, &shift_mask)) {
            item = PyLong_FromUnsignedLongLong((unsigned long long) shift_mask);
        } else {
            Py_INCREF(Py_None);
            item = Py_None;
        }
        Py_DECREF(output_seq);
        if (item == NULL) {
            goto error;
        }
        PyTuple_SET_ITEM(result, output_idx, item);
    }

    free_output_solver_inputs(eps0, pivot_col, row_ops);
    Py_DECREF(outputs_seq);
    return result;

error:
    free_output_solver_inputs(eps0, pivot_col, row_ops);
    Py_XDECREF(outputs_seq);
    Py_XDECREF(result);
    return NULL;
}
