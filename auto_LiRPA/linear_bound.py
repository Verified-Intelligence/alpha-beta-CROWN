class LinearBound:
    def __init__(
            self, lw=None, lb=None, uw=None, ub=None, lower=None, upper=None,
            from_input=None, x_L=None, x_U=None, offset=0, tot_dim=None):
        self.lw = lw
        self.lb = lb
        self.uw = uw
        self.ub = ub
        self.lower = lower
        self.upper = upper
        self.from_input = from_input
        self.x_L = x_L
        self.x_U = x_U
        # Offset for input variables. Used for batched forward bound
        # propagation.
        self.offset = offset
        if tot_dim is not None:
            self.tot_dim = tot_dim
        elif lw is not None:
            self.tot_dim = lw.shape[1]
        else:
            self.tot_dim = 0

    def is_single_bound(self):
        """Check whether the linear lower bound and the linear upper bound are
        the same."""
        if (self.lw is not None and self.uw is not None
                and self.lb is not None and self.ub is not None):
            return (self.lw.data_ptr() == self.uw.data_ptr()
                and self.lb.data_ptr() == self.ub.data_ptr()
                and self.x_L is not None and self.x_U is not None)
        else:
            return True
