# Code Review: real_model_stealing_averaging.py

## Issues Found

### 1. **Critical Bugs**
- **Line 97**: `input_shape = (3, 32, 32)` overrides all previous input_shape calculations, making lines 86-95 useless
- **Global variable usage**: `RESULTS_PATH` and `LOSS_Acc_FOLDER` are used in `model_extraction_attack()` but defined in `main()`, causing NameError if function is called independently

### 2. **Code Quality Issues**
- **Duplicate imports**: `import os` appears twice (lines 1 and 3)
- **Unused variables**: `logging_path`, `results_path`, `model_to_attack_name`, `x_test0`, `y_test0`, `dropout` parameter
- **Commented code**: Large blocks of commented code (lines 51-73, 82-84, 136-142, 208) should be removed
- **File handling**: Using `open()` without context manager - file may not close on exceptions
- **No error handling**: Missing try-except blocks for file operations, model loading, array operations

### 3. **Design Issues**
- **Magic numbers**: Hardcoded values throughout (10 classes, 64 batch_size, 0.001 lr, 500/5000/20000 thresholds)
- **Type hints missing**: No type annotations for function parameters or return values
- **Inconsistent naming**: Mix of naming conventions
- **Hardcoded paths**: Relative paths like `"../data"` that could break
- **Function parameters**: `model_extraction_attack()` uses global variables instead of parameters

### 4. **Logic Issues**
- **Plotting bug**: Lines 231-232 plot the same data for each group, overwriting previous plots
- **Inefficient data loading**: Loading entire test set into memory at once
- **Unused split**: `x_test0` and `y_test0` are created but never used
- **Watermark selection logic**: Could be simplified with a helper function

### 5. **Maintenance Issues**
- **Code duplication**: Model dictionary duplicated from other files
- **No validation**: No checks for file existence, model architecture validity, array shapes
- **Poor separation**: Business logic mixed with I/O operations
- **No logging**: Using print statements instead of proper logging

## Recommendations

1. Fix the input_shape bug on line 97
2. Pass paths as function parameters instead of using globals
3. Add type hints throughout
4. Use context managers for file operations
5. Add error handling for all I/O operations
6. Extract magic numbers as constants
7. Remove all commented code
8. Fix the plotting logic
9. Add input validation
10. Use proper logging instead of print statements
11. Add docstrings to all functions
12. Refactor watermark selection into a helper function

