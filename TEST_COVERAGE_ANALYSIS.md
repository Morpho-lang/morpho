# Test Coverage Analysis for morpho-newlinalg

## Overview
This document analyzes the completeness of the test suite for the linear algebra library, with a focus on ComplexMatrix functionality.

## Current Test Coverage

### Constructors
**XMatrix:**
- ✅ `matrix_constructor.morpho` - Basic constructor
- ✅ `matrix_list_constructor.morpho` - Constructor from list of lists
- ✅ `vector_constructor.morpho` - Vector constructor

**ComplexMatrix:**
- ❌ Missing: `complexmatrix_constructor.morpho` - Basic constructor test
- ❌ Missing: `complexmatrix_list_constructor.morpho` - Constructor from list (if supported)

### Indexing Operations
**XMatrix:**
- ✅ `matrix_getindex.morpho` - Get element by index
- ✅ `matrix_setindex.morpho` - Set element by index
- ✅ `matrix_getcolumn.morpho` - Get column
- ✅ `matrix_setcolumn.morpho` - Set column

**ComplexMatrix:**
- ✅ `complexmatrix_getindex.morpho` - Get element by index
- ❌ Missing: `complexmatrix_setindex.morpho` - Set element by index
- ✅ `complexmatrix_getcolumn.morpho` - Get column
- ✅ `complexmatrix_setcolumn.morpho` - Set column

### Arithmetic Operations
**XMatrix:**
- ✅ `matrix_add_scalar.morpho` - Add scalar
- ✅ `matrix_add_nil.morpho` - Add nil
- ✅ `matrix_addr_nil.morpho` - Addr nil
- ✅ `matrix_addr_scalar.morpho` - Addr scalar
- ✅ `matrix_sub_scalar.morpho` - Subtract scalar
- ✅ `matrix_subr_scalar.morpho` - Subr scalar
- ✅ `matrix_acc.morpho` - Accumulate
- ❌ Missing: `matrix_mul_scalar.morpho` - Multiply by scalar
- ❌ Missing: `matrix_mul_matrix.morpho` - Matrix multiplication
- ❌ Missing: `matrix_div_scalar.morpho` - Divide by scalar
- ❌ Missing: `matrix_div_matrix.morpho` - Matrix division (solve)

**ComplexMatrix:**
- ✅ `complexmatrix_add_scalar.morpho` - Add scalar
- ✅ `complexmatrix_add_nil.morpho` - Add nil
- ✅ `complexmatrix_addr_nil.morpho` - Addr nil
- ❌ Missing: `complexmatrix_sub_scalar.morpho` - Subtract scalar (exists but may need verification)
- ✅ `complexmatrix_subr_scalar.morpho` - Subr scalar
- ✅ `complexmatrix_acc.morpho` - Accumulate
- ❌ Missing: `complexmatrix_mul_scalar.morpho` - Multiply by scalar
- ❌ Missing: `complexmatrix_mul_matrix.morpho` - Matrix multiplication
- ❌ Missing: `complexmatrix_div_scalar.morpho` - Divide by scalar
- ❌ Missing: `complexmatrix_div_matrix.morpho` - Matrix division (solve)

### Assignment and Cloning
**XMatrix:**
- ✅ `matrix_assign.morpho` - Assignment

**ComplexMatrix:**
- ✅ `complexmatrix_assign.morpho` - Assignment
- ❌ Missing: `complexmatrix_clone.morpho` - Clone test

### Methods
**XMatrix:**
- ✅ `matrix_trace.morpho` - Trace
- ✅ `matrix_inverse.morpho` - Inverse
- ✅ `matrix_inverse_singular.morpho` - Inverse error case
- ✅ `matrix_transpose.morpho` - Transpose
- ✅ `matrix_eigenvalues.morpho` - Eigenvalues
- ✅ `matrix_eigensystem.morpho` - Eigensystem
- ✅ `matrix_enumerate.morpho` - Enumerate
- ❌ Missing: `matrix_inner.morpho` - Inner product
- ❌ Missing: `matrix_norm.morpho` - Norm
- ❌ Missing: `matrix_count.morpho` - Count
- ❌ Missing: `matrix_dimensions.morpho` - Dimensions
- ❌ Missing: `matrix_reshape.morpho` - Reshape

**ComplexMatrix:**
- ✅ `complexmatrix_trace.morpho` - Trace
- ✅ `complexmatrix_inverse.morpho` - Inverse
- ❌ Missing: `complexmatrix_inverse_singular.morpho` - Inverse error case
- ✅ `complexmatrix_transpose.morpho` - Transpose
- ✅ `complexmatrix_eigenvalues.morpho` - Eigenvalues
- ✅ `complexmatrix_eigensystem.morpho` - Eigensystem
- ✅ `complexmatrix_enumerate.morpho` - Enumerate
- ❌ Missing: `complexmatrix_inner.morpho` - Inner product
- ❌ Missing: `complexmatrix_count.morpho` - Count
- ❌ Missing: `complexmatrix_dimensions.morpho` - Dimensions
- ❌ Missing: `complexmatrix_reshape.morpho` - Reshape

## Missing Test Categories

### 1. Error Handling Tests
- **Out of bounds indexing** - Test negative indices, indices >= matrix dimensions
- **Incompatible dimensions** - Test arithmetic operations with mismatched dimensions
- **Non-square matrix errors** - Test operations requiring square matrices (trace, inverse, eigenvalues) on non-square matrices
- **Singular matrix errors** - Test inverse on singular complex matrices
- **Division by zero** - Test scalar division by zero

### 2. Edge Cases
- **Empty matrices** - 0x0, 0x1, 1x0 matrices
- **Single element matrices** - 1x1 matrices
- **Large matrices** - Stress tests for large dimensions
- **Zero matrices** - Operations with all-zero matrices
- **Identity matrices** - Operations with identity matrices (if constructor exists)

### 3. Complex-Specific Tests
- **Pure real matrices** - ComplexMatrix with all real values
- **Pure imaginary matrices** - ComplexMatrix with all imaginary values
- **Complex conjugation** - Test if inner product uses conjugation correctly
- **Complex arithmetic edge cases** - Operations with very small/large complex numbers

### 4. Integration Tests
- **Chained operations** - Multiple operations in sequence
- **Mixed operations** - Combining different operations
- **Memory management** - Large number of operations to check for leaks

## Recommended New Tests

### ✅ Created Tests (High Priority - Core Functionality)

1. ✅ **complexmatrix_constructor.morpho** - Basic constructor test
2. ✅ **complexmatrix_mul_matrix.morpho** - Matrix multiplication
3. ✅ **complexmatrix_mul_scalar.morpho** - Scalar multiplication
4. ✅ **complexmatrix_div_scalar.morpho** - Scalar division
5. ✅ **complexmatrix_div_matrix.morpho** - Matrix division (linear solve)
6. ✅ **complexmatrix_inner.morpho** - Inner product
7. ✅ **complexmatrix_inverse_singular.morpho** - Error case for singular matrix
8. ✅ **complexmatrix_count.morpho** - Count elements
9. ✅ **complexmatrix_dimensions.morpho** - Get dimensions
10. ✅ **complexmatrix_reshape.morpho** - Reshape matrix
11. ✅ **complexmatrix_setindex.morpho** - Set element by index
12. ✅ **complexmatrix_clone.morpho** - Clone matrix

### ✅ Created Tests (Error Cases)

13. ✅ **complexmatrix_index_out_of_bounds.morpho** - Out of bounds indexing
14. ✅ **complexmatrix_incompatible_dimensions.morpho** - Dimension mismatch errors
15. ✅ **complexmatrix_non_square_error.morpho** - Non-square matrix errors

### ✅ Created Tests (XMatrix Missing Tests)

16. ✅ **matrix_mul_matrix.morpho** - XMatrix matrix multiplication
17. ✅ **matrix_mul_scalar.morpho** - XMatrix scalar multiplication
18. ✅ **matrix_div_scalar.morpho** - XMatrix scalar division
19. ✅ **matrix_div_matrix.morpho** - XMatrix matrix division
20. ✅ **matrix_inner.morpho** - XMatrix inner product
21. ✅ **matrix_norm.morpho** - XMatrix norm
22. ✅ **matrix_count.morpho** - XMatrix count
23. ✅ **matrix_dimensions.morpho** - XMatrix dimensions
24. ✅ **matrix_reshape.morpho** - XMatrix reshape

### Lower Priority (Edge Cases)

23. **complexmatrix_empty.morpho** - Empty matrix operations
24. **complexmatrix_single_element.morpho** - 1x1 matrix
25. **complexmatrix_zero.morpho** - Zero matrix operations
26. **complexmatrix_pure_real.morpho** - Real-valued complex matrix
27. **complexmatrix_pure_imaginary.morpho** - Imaginary-valued complex matrix

## Test Organization Suggestions

Consider organizing tests into additional subdirectories:
- `errors/` - Error handling tests
- `edge_cases/` - Edge case tests
- `integration/` - Integration tests

## Notes

- ComplexMatrix does NOT have a `norm()` method (unlike XMatrix)
- Some operations may need tests for both real and complex scalars
- The `inner()` method for ComplexMatrix uses complex conjugation (Frobenius inner product)
- Matrix division (`/`) is implemented as solving a linear system

