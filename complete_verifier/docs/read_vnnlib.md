# vnnlib Simple Utilities Documentation

This is a documentation for the vnnlib parser `../read_vnnlib.py`

**Code Author**: Stanley Bak <br>
**Date**: June 2021<br>

**Documentation Author**: Zhuoxuan Zhang<br>
**Date**: Aug 2023<br>

## Overview

This utility provides functions to process `.vnnlib` files, a format used to specify verification problems for neural networks. The core functionalities include reading statements from a `.vnnlib` file and parsing them into a machine-friendly format. The parser is not designed to be fully general; instead, it assumes files are given in a nice format. Below, we document the return value format of core functions and give several vnnlib file examples to help the reader understand the parser better.

## Table of Contents

- [vnnlib Simple Utilities Documentation](#vnnlib-simple-utilities-documentation)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Functions](#functions)
    - [read\_statements](#read_statements)
    - [update\_rv\_tuple](#update_rv_tuple)
    - [make\_input\_box\_dict](#make_input_box_dict)
    - [read\_vnnlib](#read_vnnlib)

## Functions

### read_statements

**Signature**: `read_statements(vnnlib_filename) -> List[str]` 

**Description**:  
Processes a `.vnnlib` file and returns a list of string statements. Useful for removing comments, blank lines, and combining multi-line statements.

**Parameters**:   
`vnnlib_filename`: Path to the `.vnnlib` file to process.

**Returns**:  
A list of statements from the file, processed for comments, blank lines, and combined multi-line statements.

### update_rv_tuple

**Signature**: `update_rv_tuple(rv_tuple, op, first, second, num_inputs, num_outputs) -> None`

**Description**:  
Updates a tuple with constraints from the `.vnnlib` file.

**Parameters**:   
`rv_tuple`: The tuple to update.  
`op`: Operation ("<=" or ">=").  
`first`: First operand.  
`second`: Second operand.  
`num_inputs`: Total number of input variables.  
`num_outputs`: Total number of output variables.

### make_input_box_dict

**Signature**: `make_input_box_dict(num_inputs) -> Dict[int, List[float]]`

**Description**:  
Creates a dictionary for input box constraints.

**Parameters**:   
`num_inputs`: Total number of input variables.

**Returns**:  
A dictionary with input indices as keys and their constraints as values.

### read_vnnlib

**Signature**: `read_vnnlib(vnnlib_filename, regression=False) -> Union[List[Tuple]], List[Tuple]]`

**Description**:  
Reads and processes a `.vnnlib` file. It is designed for 'nice' format files. For optimization, it caches the parsed output using a hash of the original file, so subsequent loads are faster.

**Parameters**:   
`vnnlib_filename`: Path to the `.vnnlib` file to process.  
`regression`: Boolean indicating if the specification is for a regression problem. Default is `False`.

**Returns**:  
A processed list containing the constraints or regression cases parsed from the `.vnnlib` file.
The list a list containing 2-tuples:
        1. input ranges (box), list of pairs for each input variable
        2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
                          Each element in the list is a term in a disjunction for the specification.

**Examples**:
1. For `./test_vnnlib/little.vnnlib`:
  
-**Constraint Input**:  
```python
(assert (>= Y_0 3.991125645861615))
```
-**Output**:
```python
[(
    [[0.6, 0.679857769], [-0.5, 0.5], [-0.5, 0.5], [0.45, 0.5], [-0.5, -0.45]], 
    [(
        array([[-1.,  0.,  0.,  0.,  0.]]), 
        array([-3.99112565])
    )]
)]
```
-**Output Explanation**: 
>X_range:<br>
>(List[X_lower_bound, X_higher_bound]): [[0.6, 0.679857769], [-0.5, 0.5], [-0.5, 0.5], [0.45, 0.5], [-0.5, -0.45]], which indicates 0.6 <= X_0 <= >0.679857769, -0.5 <= X_1 <= 0.5, ... and so on<br>

>mat: array([[-1.,  0.,  0.,  0.,  0.]])<br>
>rhs: array([-3.99112565])<br>
>--> mat * y <= rhs<br>
>So, -Y_0 <= -3.991125645861615<br>
>Or, Y_0 >= 3.991125645861615<br>


2. For `./test_vnnlib/little2.vnnlib`:

-**Constraint Input (AND clause)**:
```python
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
(assert (<= Y_3 Y_0))
(assert (<= Y_4 Y_0))
```
-**Output**:
```python
[(
    [[0.6, 0.679857769], [-0.5, 0.5], [-0.5, 0.5], [0.45, 0.5], [-0.5, -0.45]], 
    [(
        array([[-1.,  1.,  0.,  0.,  0.], [-1.,  0.,  1.,  0.,  0.], [-1.,  0.,  0.,  1.,  0.],[-1.,  0.,  0.,  0.,  1.]]), 
       array([0., 0., 0., 0.])
    )]
)]
```
-**Output Explanation**:
>X_range:<br>
>(List[X_lower_bound, X_higher_bound]): [[0.6, 0.679857769], [-0.5, 0.5], [-0.5, 0.5], [0.45, 0.5], [-0.5, -0.45]], which indicates 0.6 <= X_0 <= >0.679857769, -0.5 <= X_1 <= 0.5, ... and so on<br>

>mat: array([[-1.,  1.,  0.,  0.,  0.], [-1.,  0.,  1.,  0.,  0.], [-1.,  0.,  0.,  1.,  0.],[-1.,  0.,  0.,  0.,  1.]])<br>
>rhs: array([0., 0., 0., 0.])<br>
>--> mat * y <= rhs<br>
>So, -Y_0 + Y_1 <= 0, -Y_0 + Y_2 <= 0, -Y_0 + Y_3 <= 0, -Y_0 + Y_4 <= 0<br>
>Or, Y_1 <= Y_0, Y_2 <= Y_0, Y_3 <= Y_0, Y_4 <= Y_0<br>


3. For `./test_vnnlib/little3.vnnlib`:
   
-**Constraint Input (OR clause)**:
```python
(assert (or
    (and (<= Y_0 Y_3))
    (and (<= Y_1 Y_3))
    (and (<= Y_2 Y_3))
    (and (<= Y_4 Y_3))
))
```
-**Output**:
```python
[(
    [[-0.295233916, -0.212261512], [-0.063661977, -0.022281692], [-0.499999896, -0.498408347], [-0.5, -0.454545455], [-0.5, -0.375]], 
    [(
        array([[ 1.,  0.,  0., -1.,  0.]]), 
        array([0.])
        ), 
        (
        array([[ 0.,  1.,  0., -1.,  0.]]), 
        array([0.])
        ), 
        (
        array([[ 0.,  0.,  1., -1.,  0.]]), 
        array([0.])
        ), 
        (
        array([[ 0.,  0.,  0., -1.,  1.]]), 
        array([0.])
        )]
)]
```
-**Output Explanation**:
>X_range:<br>
>(List[X_lower_bound, X_higher_bound]): [[-0.295233916, -0.212261512], [-0.063661977, -0.022281692], [-0.499999896, -0.498408347], [-0.5, >-0.454545455], [-0.5, -0.375]], which indicates -0.295233916 <= X_0 <= -0.212261512, -0.063661977 <= X_1 <= -0.022281692, ... and so on<br>

>(Note: If one of the following is True, then assertion returns True)<br>
>mat: array([[ 1.,  0.,  0., -1.,  0.]])<br>
>rhs:  array([0.])<br>
>--> mat * y <= rhs<br>
>So, Y_0 + (-Y_3) <= 0<br>
>Or, Y_0 <= Y_3<br>

>   OR<br>

>mat: array([[ 0.,  1.,  0., -1.,  0.]])<br>
>rhs:  array([0.])<br>
>--> mat * y <= rhs<br>
>So, Y_1 + (-Y_3) <= 0<br>
>Or, Y_1 <= Y_3<br>

>    OR<br>

>mat: array([[ 0.,  0.,  1., -1.,  0.]])<br>
>rhs:  array([0.])<br>
>--> mat * y <= rhs<br>
>So, Y_2 + (-Y_3) <= 0<br>
>Or, Y_2 <= Y_3<br>

>    OR<br>

>mat: array([[ 0.,  0.,  0., -1.,  1.]])<br>
>rhs:  array([0.])<br>
>--> mat * y <= rhs<br>
>So, (-Y_3) + Y_4 <= 0<br>
>Or, Y_4 <= Y_3<br>

4. For `./test_vnnlib/littleGeneral.vnnlib`:
   
-**Constraint Input (mixed AND and OR clause):**:
```python
(assert (or (and (<= Y_1 0.5) (>= Y_2 Y_3) (<= Y_4 0.1))
            (and (<= Y_2 Y_0) (<= Y_3 1.0))
            (and (<= Y_1 1) (>= Y_2 2))
        )
)
```
-**Output**:
```python
[(
    [[-0.295233916, -0.212261512], [-0.063661977, -0.022281692], [-0.499999896, -0.498408347], [-0.5, -0.454545455], [-0.5, -0.375]], 
    [(
        array([[ 0.,  1.,  0.,  0.,  0.], [ 0.,  0., -1.,  1.,  0.],[ 0.,  0.,  0.,  0.,  1.]]),
        array([0.5, 0. , 0.1])
    ), 
    (
        array([[-1.,  0.,  1.,  0.,  0.],[ 0.,  0.,  0.,  1.,  0.]]), 
        array([0., 1.])
    ), 
    (   
        array([[ 0.,  1.,  0.,  0.,  0.], [ 0.,  0., -1.,  0.,  0.]]), 
        array([ 1., -2.])
    )
    ]
)]
```
-**Output Explanation**:
>X_range:<br>
>(List[X_lower_bound, X_higher_bound]): [[-0.295233916, -0.212261512], [-0.063661977, -0.022281692], [-0.499999896, -0.498408347], [-0.5, >-0.454545455], [-0.5, -0.375]], which indicates -0.295233916 <= X_0 <= -0.212261512, -0.063661977 <= X_1 <= -0.022281692, ... and so on<br>

>(Note: If one of the following is True, then assertion returns True)<br>
>mat: array([[ 0.,  1.,  0.,  0.,  0.], [ 0.,  0., -1.,  1.,  0.],[ 0.,  0.,  0.,  0.,  1.]])<br>
>rhs: array([0.5, 0. , 0.1])<br>
>--> mat * y <= rhs<br>
>So, Y_1 <= 0.5, -Y_2 + Y_3 <= 0, Y_4 <= 0.1<br>
>Or, Y_1 <= 0.5, Y_2 >= Y_3, Y_4 <= 0.1<br>

>    OR<br>

>mat: array([[-1.,  0.,  1.,  0.,  0.],[ 0.,  0.,  0.,  1.,  0.]]), <br>
>rhs: array([0., 1.])<br>
>--> mat * y <= rhs<br>
>So, -Y_0 + Y_2 <= 0, Y_3 <= 1.0<br>
>Or, Y_2 <= Y_0, Y_3 <= 1.0<br>

>    OR<br>

>mat: array([[ 0.,  1.,  0.,  0.,  0.], [ 0.,  0., -1.,  0.,  0.]]), <br>
>rhs: array([ 1., -2.])<br>
>--> mat * y <= rhs<br>
>So, Y_1 <= 1, -Y_2 <= -2<br>
>Or, Y_1 <= 1, Y_2 >= 2<br>

5. For `./test_vnnlib/littleXY.vnnlib`:
   
-**Constraint Input (both X and Y appear in mixed AND and OR clause):**:
```python
(assert (or (and (<= X_0 0.66) (>= Y_2 Y_3) (<= Y_4 0.1))
            (and (<= X_1 3) (>= X_2 0.2) (<= Y_3 1.0))
        )
)
```
-**Output**:
```python
[(
    [[0.6, 0.66], [-0.5, 0.5], [-0.5, 0.5], [0.45, 0.5], [-0.5, -0.45]], 
    [(
        array([[ 0.,  0., -1.,  1.,  0.],[ 0.,  0.,  0.,  0.,  1.]]), 
        array([0. , 0.1])
    )]
), 
(
    [[0.6, 0.679857769], [-0.5, 0.5], [0.2, 0.5], [0.45, 0.5], [-0.5, -0.45]], 
    [(
        array([[0., 0., 0., 1., 0.]]), 
        array([1.])
    )]
)]
```
-**Output Explanation**:
>Note that since we defined X_? values in assertion, we reflect this constriction on X ranges. If this definition can constrain X's original range, >we shorten the X range. If not, skip.<br>

>In our case, X_range should be:<br>
>[[0.6, 0.679857769], [-0.5, 0.5], [-0.5, 0.5], [0.45, 0.5], [-0.5, -0.45]]<br>

>However, in the first clause, X_0 is asserted to be <= 0.66. We, thus, modify X_0_range from [0.6, 0.679857769] to [0.6, 0.66]. Similarly, in the >second clause, X_2 is asserted to be >= 0.2. We, thus, modify X_2_range from [-0.5, 0.5] to [-0.2, 0.5]. Note that, though X_1 is asserted to be <= >3, we do not modify its range since the original range better constrains X_1 value.<br>

>Y value assertion runs the same as in littleGeneral.vnnlib.<br>
