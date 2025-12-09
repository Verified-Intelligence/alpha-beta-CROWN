# alpha-beta-CROWN New API Overview
Here's an introduction to the high-level entry point for alpha-beta-CROWN. Provide a
model (PyTorch `nn.Module` or ONNX path), describe the property via
`VerificationSpec`, optionally tweak a config, then call `ABCrownSolver.solve()`. The
notebook demos under `complete_verifier/examples_abcrown` reuse exactly the same
API pieces described here.

---

## 1. Quick start
To install `alpha-beta-CROWN`, follow these steps:

1. Clone the repository:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
```

2. (Optional) Enable CPLEX Cuts: If you require CPLEX cuts for verification (e.g. GCP-CROWN), you must manually install CPLEX and compile the `get_cuts` executable now. See the instructions in [`complete_verifier/cuts/CPLEX_cuts/README.md`](../cuts/CPLEX_cuts/README.md).

3. Install the package:
```bash
pip install .
```

Then the API can be imported and called via

```python
from abcrown import (
    ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars
)

# Construct spec with L-infinity box and logit ordering
x = input_vars((1, 28, 28))
y = output_vars(3)
input_constraint = (x >= base - eps) & (x <= base + eps)
output_constraint = (y[0] > y[1]) & (y[0] > y[2])

spec = VerificationSpec.build_spec(
    input_vars=x,
    output_vars=y,
    input_constraint=input_constraint,
    output_constraint=output_constraint,
)
# Config (defaults)
config = ConfigBuilder.from_defaults()

# Solve
solver = ABCrownSolver(spec, model, config=config)
result = solver.solve()
print(result.status, result.success)
```

---

## 2. Function reference

### `input_vars(shape)` / `output_vars(dim)`
Creates symbolic `VariableVector` objects used by the expression Domain-Specific Language (DSL).
- `shape`: int / tuple / `torch.Size` describing the input tensor.
- `dim`: integer length of the output tensor.

### `VerificationSpec.build_spec(...)`
Single entry point that dispatches automatically:
- Bounds + clauses: pass `lower`, `upper`, `clauses`.
- Expression DSL: pass `input_vars`, `output_vars`, and boolean combinations of
  comparisons (see cheat sheet below).
- `vnnlib_path="path/to/spec.vnnlib"`: load an existing VNNLIB property.
- Notice that the API only accept **strict output specification**, i.e., **<, >** are allowed while **>=, <=** are not.

Internally the spec is converted and stored in an OR-of-AND list of `(C, rhs)` inequalities with
`C @ output < rhs` (Disjunction Normal Form), but you don't need to manually build the DNF.

### DSL cheat sheet

| Goal               | DSL example                                             | Notes                                         |
|-------------------|----------------------------------------------------------|-----------------------------------------------|
| 1-D bound         | `(x[0] > -0.1) & (x[0] < 0.1)`                           | combine predicates with `&`                   |
| Tensor L∞ box     | `(x > lower_tensor) & (x < upper_tensor)`                | tensors broadcast to the symbolic shape       |
| Logit ordering    | `(y[0] > y[1]) & (y[0] > y[2])`                          | indexing syntax mirrors PyTorch               |
| Linear inequality | `y[0] - y[1] > 0`                                        | auto-converted to `(C, rhs)`                  |
| OR between specs  | `(y[0] > 0) \| (y[1] > 0)`                               | produces two OR clauses internally            |
| Pin to a point    | `point = torch.tensor([1.0]); (x > point) & (x < point)` | handy for single-sample demos                 |

Comparisons must involve `input_vars` or `output_vars`. The opposite operand can be a
scalar, list, NumPy array, or torch tensor.

### `default_config()` / `ConfigBuilder`
- `default_config()` returns a deep copy of the built-in dict. See `complete_verifier/arguments.py` for the default values of all configuration options. **NOTE:** `"complete_verifier"` defaults to `"auto"` when the verifier is invoked through this API. 
- `ConfigBuilder.from_defaults()` provides a chainable helper:
  ```python
  cfg = (
      ConfigBuilder.from_defaults()
      .set(general__device="cpu")
      .set(attack__pgd_order="skip")
      ()
  )
  ```
- `.update()` merges nested dicts, `.from_yaml(path)` loads overrides from YAML,
  `.from_config(cfg)` clones an existing configuration.
- `.set()` only accepts keyword overrides; use `.update()` when you need to
  inject nested dicts or callables:
  ```python
  builder = ConfigBuilder.from_defaults()
  builder.set(attack__pgd_order="skip")              # simple path override
  builder.update({"attack": {"pgd_order": "skip"}})  # deep-merge mapping
  ```

### `ABCrownSolver`
`ABCrownSolver(spec, computing_graph, config=None, name=None)`
- `spec`: `VerificationSpec` object (or dict with `lower/upper/clauses`).
- `computing_graph`: `torch.nn.Module` instance or ONNX path (auto-imported).
- `config`: dict / builder result; omitted → clone defaults.
- `name`: optional identifier used in logs.

`solve()` returns `SolveResult` with:
- `status`: `verified`, `unsafe-pgd`, `unsafe-bab`, `safe-incomplete`, `unknown`, …
- `success`: boolean (`True` if the property is satisfied, or a counterexample is
  confirmed when unsafety is expected).
- `reference`: optional dict of intermediate data (bounds, attack traces, etc.).
- `stats`: metadata such as elapsed time, PGD iterations, BaB splits.

`SolveResult` exposes `status`, `success`, `reference`, and `stats` as attributes.

---

## 3. Typical workflow recap

1. Declare symbols: `x = input_vars(shape)`, `y = output_vars(dim)`.
2. Write input/output constraints using the DSL.
3. Build the spec via `VerificationSpec.build_spec(...)`.
4. (Optional) tweak a config with `ConfigBuilder` or `default_config()`.
5. Instantiate `ABCrownSolver(spec, model, config)` and call `.solve()`.

`result.status` indicates which stage produced the answer:
- `unsafe-pgd`: PGD attack already found a counterexample.
- `unsafe-bab`: branching-and-bound found a counterexample.
- `verified` / `safe`: property proven.
- `safe-incomplete`: incomplete verification sufficed.
- `unknown`: search stopped early (timeout / resource limit).

---

## 4. Reference snippets

### Image classification (logit ordering)
```python
def run_image_demo():
    base = torch.rand(1, 1, 28, 28); eps = 0.02
    x = input_vars((1, 28, 28)); y = output_vars(3)
    input_constraint = (x >= base - eps) & (x <= base + eps)
    output_constraint = (y[0] > y[1]) & (y[0] > y[2])
    spec = VerificationSpec.build_spec(x, y, input_constraint, output_constraint)

    model = SimpleConvClassifier()
    solver = ABCrownSolver(spec, model, config=demo_config())
    print(solver.solve().status)
```

### Lyapunov controller check
```python
def run_lyapunov_demo():
    x = input_vars(2); y = output_vars(2)      # y[0] = V(x), y[1] = V_dot
    input_constraint = (x >= -1.0) & (x <= 1.0)
    output_constraint = (y[0] > 0.0) & (y[1] < 0.0)
    spec = VerificationSpec.build_spec(x, y, input_constraint, output_constraint)

    model = LyapunovComputationGraph(...)
    res = ABCrownSolver(spec, model, config=demo_config()).solve()
    print(res.status, res.success)
```

### Scalar “safe vs unsafe” toy
```python
class SimpleResidual(torch.nn.Module):
    def forward(self, x):
        return 1.5 - x

def run_scalar_demo(x_value):
    x = input_vars(1); y = output_vars(1)
    point = torch.tensor([x_value])
    spec = VerificationSpec.build_spec(
        x,
        y,
        input_constraint=(x >= point) & (x <= point),
        output_constraint=(y[0] > 0.0),
    )
    result = ABCrownSolver(spec, SimpleResidual()).solve()
    print(f"x={x_value} -> {result.status}")
```

Use these templates as drop-in references: swap in your model, adjust the two
constraints, and the solver plumbing stays identical.

## 5. Examples 
Examples of simple image classification verification and neural Lyapunov stability verification are provided in `complete_verifier/examples_abcrown`