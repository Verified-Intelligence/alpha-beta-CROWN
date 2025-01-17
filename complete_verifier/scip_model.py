#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import time
import os

try:
    from pyscipopt import Model, scip, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING
    import gurobipy as grb
    # import cvxpy as cp
except ModuleNotFoundError:
    pass


class SCIPVariable:
    def __init__(self,  var):
        self.var = var

    @property
    def lb(self):
        return self.var.getLbLocal()

    @property
    def ub(self):
        return self.var.getUbLocal()

    @property
    def LB(self):
        return self.var.getLbLocal()

    @property
    def UB(self):
        return self.var.getUbLocal()

    @property
    def VarName(self):
        return self.var.name


class SCIPModel(Model):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setParam('MIPGap', 1e-4)  # default gurobi setting
        self.setParam('MIPGapAbs', 1e-10)  # default gurobi setting
        _setParam = super(SCIPModel, self).setParam
        # TODO: for Shiqi:
        # 1. Enable RLT Cuts (make its priority very negative?)
        # 2. Disable all primal heuristics (we don't care about the "primal" feasible solution of the MIP; we only care the dual bound).
        # 3. Warm start the solver with adversarial candidates (a good primal solution is important).
        # 4. Change focus of the solver, focusing on finding cuts (no direct option, need to change priorities of several things).
        _setParam('separating/rlt/priority', -10000)
        _setParam('separating/rlt/maxunknownterms', -1)
        _setParam('separating/rlt/maxusedvars', -1)
        _setParam('separating/rlt/maxroundsroot', -1)
        _setParam('separating/rlt/onlyoriginal', False)
        _setParam('separating/rlt/goodscore', 0.5)
        _setParam('separating/rlt/hiddenrlt', True)

    @property
    def status(self, *args, **kwargs):
        # https://github.com/scipopt/PySCIPOpt/blob/769db477ca99260f1a5129c95dd08d9dd61088c1/src/pyscipopt/scip.pyx#L4370
        # https://www.gurobi.com/documentation/9.5/refman/optimization_status_codes.html#sec:StatusCodes
        stat = super().getStatus(*args, **kwargs)
        if stat == "optimal":
            return 2
        elif stat == "timelimit":
            return 9
        elif stat == "infeasible":
            return 3
        elif stat == "unbounded":
            return 5
        elif stat == "userinterrupt":
            return 11
        elif stat == "inforunbd":
            return 4
        elif stat == "nodelimit":
            # return 8
            return 9  # We should return 9 for any limit reached.
        elif stat == "totalnodelimit":
            # return "totalnodelimit"  # no exact match
            return 9
        elif stat == "stallnodelimit":
            # return "stallnodelimit"  # no exact match
            return 9
        elif stat == "gaplimit":
            # return "gaplimit"  # no exact match
            return 9
        elif stat == "memlimit":
            # return "memlimit"  # no exact match
            return 9
        elif stat == "sollimit":
            # return "sollimit"  # no exact match
            return 9
        elif stat == "bestsollimit":
            return 15
        elif stat == "restartlimit":
            # return "restartlimit"
            return 9
        else:
            return "unknown"

    def setObjective(self, expression, sense=None):
        if sense == grb.GRB.MAXIMIZE:
            sense = 'maximize'
        elif sense == grb.GRB.MINIMIZE:
            sense = 'minimize'
        else:
            raise NotImplementedError

        if isinstance(expression, SCIPVariable):
            expression = expression.var
        super().setObjective(expression, sense)

    def addVar(self, lb=0.0, ub=None, obj=0.0, vtype=grb.GRB.CONTINUOUS, name=''):
        vtype_dict = {
            grb.GRB.CONTINUOUS: 'C',
            grb.GRB.BINARY: 'B',
            grb.GRB.INTEGER: 'I',
        }

        # args orders are different by default
        # param vtype: type of the variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
        return super().addVar(lb=lb, ub=ub, obj=obj, vtype=vtype_dict[vtype], name=name)

    def getVarByName(self, name):
        vars = super().getVars()
        for v in vars:
            if name in v.name:
                return SCIPVariable(v)

    def reset(self):
        # TODO: not find exact match
        # super().resetParams()  # This is wrong.
        pass

    def setParam(self, name, value):
        # https://www.scipopt.org/doc-8.0.0/html/PARAMETERS.php
        # https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters
        def _cuts_handler(name, val):
            _setParam = super(SCIPModel, self).setParam
            if name == 'Cuts':
                # Global cutting plane switch.
                if val <= 0:
                    # Disable all cuts.
                    print('Disabling all SCIP cutting planes.')
                    _setParam('separating/cgmip/freq', -1)
                    _setParam('separating/clique/freq', -1)
                    _setParam('separating/closecuts/freq', -1)
                    _setParam('separating/flowcover/freq', -1)
                    _setParam('separating/cmir/freq', -1)
                    _setParam('separating/knapsackcover/freq', -1)
                    _setParam('separating/aggregation/freq', -1)
                    _setParam('separating/convexproj/freq', -1)
                    _setParam('separating/disjunctive/freq', -1)
                    _setParam('separating/eccuts/freq', -1)
                    _setParam('separating/gauge/freq', -1)
                    _setParam('separating/gomory/freq', -1)
                    _setParam('separating/strongcg/freq', -1)
                    _setParam('separating/gomorymi/freq', -1)
                    _setParam('separating/impliedbounds/freq', -1)
                    _setParam('separating/interminor/freq', -1)
                    _setParam('separating/intobj/freq', -1)
                    _setParam('separating/mcf/freq', -1)
                    _setParam('separating/minor/freq', -1)
                    _setParam('separating/mixing/freq', -1)
                    _setParam('separating/oddcycle/freq', -1)
                    _setParam('separating/rapidlearning/freq', -1)
                    _setParam('separating/rlt/freq', -1)
                    _setParam('separating/zerohalf/freq', -1)
                    # _setParam('limits/nodes', 1)  # Set this for just solving the root node.
                    _setParam('presolving/maxrestarts', 0)  # Disable restarts.
                else:
                    # Change all cuts parameter to default.
                    _setParam('separating/cgmip/freq', -1)  # Default is never.
                    _setParam('separating/clique/freq', 0)  # Default is root node only.
                    _setParam('separating/closecuts/freq', -1)
                    _setParam('separating/flowcover/freq', 10)
                    _setParam('separating/cmir/freq', 10)
                    _setParam('separating/knapsackcover/freq', 10)
                    _setParam('separating/aggregation/freq', 10)
                    _setParam('separating/convexproj/freq', -1)
                    _setParam('separating/disjunctive/freq', 0)
                    _setParam('separating/eccuts/freq', -1)
                    _setParam('separating/gauge/freq', -1)
                    _setParam('separating/gomory/freq', 10)
                    _setParam('separating/strongcg/freq', 10)
                    _setParam('separating/gomorymi/freq', 10)
                    _setParam('separating/impliedbounds/freq', 10)
                    _setParam('separating/interminor/freq', -1)
                    _setParam('separating/intobj/freq', -1)
                    _setParam('separating/mcf/freq', 0)
                    _setParam('separating/minor/freq', 10)
                    _setParam('separating/mixing/freq', 10)
                    _setParam('separating/oddcycle/freq', -1)
                    _setParam('separating/rapidlearning/freq', 5)
                    _setParam('separating/rlt/freq', 0)
                    _setParam('separating/zerohalf/freq', 10)
            else:
                cuts_dict = {
                        # GurobiOptionName: (SCIPOptionName, SCIPOptionDefault),
                        'GomoryPasses': [('separating/gomory/freq', 10), ('separating/gomorymi/freq', 10)],
                        'RLTCuts': [('separating/rlt/freq', 0)],
                        'FlowCoverCuts': [('separating/flowcover/freq', 10), ('separating/aggregation/freq', 10)],
                        'MIRCuts': [('separating/cmir/freq', 10), ('separating/aggregation/freq', 10)],
                        'ImpliedCuts': [('separating/impliedbounds/freq', 10), ('separating/aggregation/freq', 10)],
                }
                if name in cuts_dict:
                    if value > 0:
                        for setting_item in cuts_dict[name]:
                            scip_opt, default_value = setting_item
                            print(f'Setting {scip_opt} to {default_value}')
                            _setParam(scip_opt, default_value)  # Set to default.
                else:
                    raise NotImplementedError(f"SCIP option {name} not supported.")

        params_dict = {
            'OutputFlag': 'display/verblevel',
            'Threads': 'parallel/maxnthreads',
            'FeasibilityTol': 'numerics/feastol',
            'TimeLimit': 'limits/time',
            'MIPGapAbs': 'limits/absgap',
            'MIPGap': 'limits/gap',
            'Cuts': _cuts_handler,
            'GomoryPasses': _cuts_handler,
            'RLTCuts': _cuts_handler,
            'FlowCoverCuts': _cuts_handler,
            'MIRCuts': _cuts_handler,
            'ImpliedCuts': _cuts_handler,
        }

        if name not in params_dict.keys():
            raise NotImplementedError(f"SCIP option {name} not supported.")

        if name == 'OutputFlag':
            value = 1 if value is False else 4

        if isinstance(params_dict[name], str):
            super().setParam(params_dict[name], value)
        else:
            params_dict[name](name, value)

    def addConstr(self, cons, name=''):
        super().addCons(cons=cons, name=name)

    def getConstrs(self):
        return super().getConss()

    def update(self):
        # Not sure SCIP handles model modifications in a lazy fashion
        # but model.optimize() will update all modifications
        pass

    def getConstrByName(self):
        # TODO
        raise NotImplementedError

    def copy(self):
        new_model = SCIPModel(sourceModel=self)
        return new_model

    @property
    def solcount(self):
        sols = self.getSols()
        return len(sols)

    @property
    def objbound(self):
        return self.getDualbound()

    @property
    def objval(self):
        return self.getPrimalbound()


class EarlyStop(Exception):
    def __init__(self, message):
        print(message)


class GenerateCutsEvent(Eventhdlr):
    def eventinit(self):
        self._call_count = 0
        self.model.catchEvent(SCIP_EVENTTYPE.ROWADDEDSEPA, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.ROWADDEDSEPA, self)

    def eventexec(self, event):
        cuts = self.model.getCuts()
        if len(cuts) > 0:
            self._call_count += 1
            filename = f"cuts_{self._call_count}.txt"
            with open(filename, "w") as f:
                for i, c in enumerate(cuts):
                    f.write(f'row {i:5d}/{len(cuts):5d}: ')
                    f.flush()
                    self.model.printRow(c, f)
            print(f"{len(cuts)} cuts saved to {filename}")


class EarlyStopEvent(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexec(self, event):
        # assert event.getType() == SCIP_EVENTTYPE.NODESOLVED
        if self.model.getDualbound() > 1e-5:
            print('Early stop: lower bound > 0!')
            self.model.interruptSolve()
            time.sleep(1)

        elif self.model.getPrimalbound() < -1e-5:
            print('Early stop: upper bound < 0!')
            self.model.interruptSolve()
            time.sleep(1)

        # raise EarlyStop('early stop!')


def event():
    # create solver instance
    s = Model()
    s.hideOutput()
    s.setPresolve(SCIP_PARAMSETTING.OFF)
    eventhdlr = EarlyStopEvent()
    s.includeEventhdlr(eventhdlr, "TestFirstLPevent", "python event handler to catch FIRSTLPEVENT")

    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5)
    # solve problem
    s.optimize()

    # print solution
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0


if __name__ == "__main__":
    # event()
    # exit()

    # Test wrapper
    model = SCIPModel("Example")  # model name is optional

    # Test functions
    x = model.addVar(name="x")
    y = model.addVar(name="y")

    model.setParam('OutputFlag', bool(os.environ.get('ALPHA_BETA_CROWN_MIP_DEBUG', False)))
    model.setParam('Threads', 1)
    model.setParam("FeasibilityTol", 2e-5)
    model.setParam('TimeLimit', 10)
    model.setParam('BestBdStop', 1e-5)  # Terminate as long as we find a positive lower bound.
    model.setParam('BestObjStop', 1e-5)  # Terminate as long as we find a adversarial example.
    # model.setParam('Heuristics', 0.5)  # no exact match
    # model.setParam('MIPFocus', 1) # no exact match

    obj = (x-y)
    model.setObjective(obj, grb.GRB.MINIMIZE)
    model.addConstr(x + y == 1)
    model.addConstr(x - y >= 1)
    model.update()

    cons = model.getConstrs()

    model.optimize()  # pass
    print('model.status', model.status)
    model.reset()
    tmp_x = model.getVarByName('x')
    model.objval

    sol = model.getBestSol()
    print("x: {}".format(sol[x]))
    print("y: {}".format(sol[y]))


