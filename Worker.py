from types import SimpleNamespace

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

class WorkerClass:

    def __init__(self,par=None):

        # a. setup
        self.setup_worker()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.nu = 0.015 # weight on labor disutility
        par.epsilon = 1.0 # curvature of labor disutility
        
        # b. productivity and wages
        par.w = 1.0 # wage rate
        par.ps = np.linspace(0.5,3.0,100) # productivities
        par.ell_max = 16.0 # max labor supply
        
        # c. taxes
        par.tau = 0.50 # proportional tax rate
        par.zeta = 0.10 # lump-sum tax
        par.kappa = np.nan # income threshold for top tax
        par.omega = 0.20 # top rate rate
          
    def utility(self,c,ell):

        par = self.par

        # If consumption is non-positive, treat it as infeasible (very low utility)
        if c <= 0.0:
            return -1e12

        # U(c, ell) = log(c) - nu * ell^(1+epsilon) / (1+epsilon)
        u = np.log(c) - par.nu * ell**(1.0 + par.epsilon) / (1.0 + par.epsilon)
        
        return u
    
    def tax(self,pre_tax_income):

        par = self.par

        # Basic tax: proportional tax + lump-sum component
        tax = par.tau * pre_tax_income + par.zeta

        # Optional top tax term:
        # tax += omega * max(pre_tax_income - kappa, 0)
        if not np.isnan(par.kappa):
            tax += par.omega * np.maximum(pre_tax_income - par.kappa, 0.0)

        return tax
    
    def income(self,p,ell):

        par = self.par

        # Pre-tax income: y = w * p * ell
        return par.w * p * ell

    def post_tax_income(self,p,ell):

        pre_tax_income = self.income(p,ell)
        tax = self.tax(pre_tax_income)

        return pre_tax_income - tax
    
    def max_post_tax_income(self,p):

        par = self.par
        return self.post_tax_income(p,par.ell_max)

    def value_of_choice(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        U = self.utility(c,ell)

        return U
    
    def get_min_ell(self,p):
    
        par = self.par

        # Minimum labor to make post-tax income non-negative in the simple system
        min_ell = par.zeta/(par.w*p*(1-par.tau))

        # Ensure non-negative and add a tiny epsilon to stay inside the feasible set
        return np.fmax(min_ell,0.0) + 1e-8
    
    def optimal_choice(self,p):

        par = self.par
        opt = SimpleNamespace()

        # a. objective function: minimize -U to maximize U
        def obj(ell):
            c = self.post_tax_income(p,ell)
            u = self.utility(c,ell)
            return -u

        # b. bounds and minimization
        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max

        res = minimize_scalar(
            obj,
            bounds=(ell_min, ell_max),
            method='bounded'
        )

        # c. results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p,opt.ell)

        return opt
    
    def FOC(self,p,ell):

        par = self.par

        # First compute consumption at (p, ell)
        c = self.post_tax_income(p,ell)

        # If consumption is non-positive, FOC is not meaningful
        if c <= 0.0:
            return np.nan

        # FOC for simple tax system (no top-tax in derivative):
        # phi = (1 - tau) * w * p / c - nu * ell^epsilon
        FOC = (1.0 - par.tau) * par.w * p / c - par.nu * ell**par.epsilon

        return FOC
    
    def optimal_choice_FOC(self,p):

        par = self.par
        opt = SimpleNamespace()

        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max

        # Helper function for root finding
        def f(ell):
            return self.FOC(p,ell)

        f_min = f(ell_min)
        f_max = f(ell_max)

        # If FOC does not change sign or is invalid → corner solution ell_min
        if np.isnan(f_min) or np.isnan(f_max) or f_min * f_max > 0:
            ell_star = ell_min
        else:
            res = root_scalar(f, bracket=(ell_min, ell_max), method='brentq')
            ell_star = res.root

        c_star = self.post_tax_income(p,ell_star)
        U_star = self.utility(c_star,ell_star)

        opt.ell = ell_star
        opt.c = c_star
        opt.U = U_star

        return opt
