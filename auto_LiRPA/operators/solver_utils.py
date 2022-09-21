class DummyGurobipyClass:
    """A dummy class with error message when gurobi is not installed."""
    def __getattr__(self, attr):
        def _f(*args, **kwargs):
            raise RuntimeError(f"method {attr} not available because gurobipy module was not built.")
        return _f

try:
    import gurobipy as grb
except ModuleNotFoundError:
    grb = DummyGurobipyClass()