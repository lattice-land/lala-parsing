# TCN format

A `.tcn` file describes a ternary constraint network: integer variables with interval domains, and constraints of the form `x = y <op> z`.
It is read by `parse_tcn` (`lala/tcn_parser.hpp`) and written by `write_preprocessed_tcn` (`lala/tcn_writer.hpp`).

```
[minimize K]
N
lb_0 ub_0
...
lb_{N-1} ub_{N-1}
M
x_0 y_0 op_0 z_0
...
x_{M-1} y_{M-1} op_{M-1} z_{M-1}
[search var_order val_order K v_1 ... v_K]*
```

* Variables are identified by their index in `0..N-1`.
* The objective line is optional; `K` is the index of the variable to minimize. To maximize `x`, minimize a variable constrained to `-x`.
* A bound is an integer, `-inf` or `inf`.
* Each constraint line `x y op z` means `x = y op z`, where `x`, `y`, `z` are variable indices.
* The file ends with zero or more search strategies, applied in order. Each one branches on the `K` listed variables;
  `var_order` is one of `input_order`, `first_fail`, `anti_first_fail`, `smallest`, `largest`, `random`,
  and `val_order` is one of `min`, `max`, `split`, `reverse_split`.

| Symbol | Operator |
|--------|----------|
| `+` | addition |
| `*` | multiplication |
| `tdiv`, `fdiv`, `cdiv`, `ediv` | truncated, floor, ceil and Euclidean division |
| `tmod`, `fmod`, `cmod`, `emod` | the corresponding moduli |
| `A` | maximum |
| `I` | minimum |
| `L` | `x = (y <= z)` |
| `=` | `x = (y == z)` |

Example (minimize `x2 = x0 + x1`, branching on `x0` and `x1`):

```
minimize 2
3
0 10
1 5
-inf inf
1
2 0 + 1
search first_fail min 2 0 1
```
