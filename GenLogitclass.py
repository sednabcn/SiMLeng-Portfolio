#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:17:43 2018
These classes are added to statsmodels package

@author: sedna
"""
import numpy as np
from statsmodels import base
from statsmodels.discrete.discrete_model import BinaryModel
    
class GenLogit(BinaryModel):
    __doc__ = """
    Binary choice genlogit model

%(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
    """ % {'params' : base._model_params_doc,
           'extra_params' : base._missing_param_doc}
    

    def get_c_value(self):        
        #self.c= float(input("Shape genlogistic: "))
        #self.c=0.35
        self.c=0.2
        return self.c  

    
    
    def cdf(self, X):
        """
        The genlogistic cumulative distribution function

        Parameters
        ----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.
        c :  'c' is as a shape parameter c>0 [c=1 becomes to logistic case]
        Returns
        -------
         1/(1 + exp(-x))^c

        Notes
        ------
        In the genlogit model,

        .. math:: \\Lambda\\left(x^{\\prime}\\beta,c\\right)=\\text{Prob}\\left(Y=1|x\\right)=\\frac{1}{\\left(1+e^{-x^{\\prime}\\beta\\right)^{c}}
        """
        
        c = self.get_c_value()
        X = np.asarray(X)
        return 1/(1+np.exp(-X))**c

    def pdf(self, X):
        """
        The logistic probability density function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.
        c : 'c' as a shape parameter
        Returns
        -------
        pdf : ndarray
            The value of the Logit probability mass function, PMF, for each
            point of X. ``np.exp(-x)/(1+np.exp(-X))**2``

        Notes
        -----
        In the genlogit model,

        .. math:: \\lambda\\left(x^{\\prime}\\beta,c\\right)=\\frac{c e^{-x^{\\prime}\\beta}}{\\left(1+e^{-x^{\\prime}\\beta}\\right)^{c+1}}
        """
        
        c = self.get_c_value()
        X = np.asarray(X)
        return c*np.exp(-X)/(1+np.exp(-X))**(c+1)

    def loglike(self, params):
        """
        Log-likelihood of genlogit model.

        Parameters
        -----------
        params : array-like
            The parameters of the logit model.
            c : 'c' as a shape parameter
        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i}\\ln\\Lambda\\left({\\prime}\\beta,c\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        return np.sum( (2*y-1)*np.log(Lcdf) +(1-y)*np.log(Lpdf))
    
    def loglikeobs(self, params):
        """
        Log-likelihood of logit model for each observation.
        c : 'c' as a shape parameter
        Parameters
        -----------
        params : array-like
            The parameters of the logit model. 

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        ------
        .. math:: \\ln L=\\sum_{i}\\ln\\Lambda\\left(q_{i}x_{i}^{\\prime}\\beta,c\\right)

        for observations :math:`i=1,...,n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        return  (2*y-1)*np.log(Lcdf) +(1-y)*np.log(Lpdf)
       

    def score(self, params):
        """
        Logit model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params: array-like
            The parameters of the model
         c : 'c' as a shape parameter
        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        Lgrad= (1-y)*(1-2.0*Lcdf) + (2.0*y-1.0)*(Lpdf/Lcdf)
        return np.dot(Lgrad,X)

    def jac(self, params):
        """ 
        Logit model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        jac : ndarray, (nobs, k_vars)
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\Lambda_{i}\\right)x_{i}

        for observations :math:`i=1,...,n`

        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        Lgrad= (1-y)*(1-2.0*Lcdf) + (2.0*y-1.0)*(Lpdf/Lcdf)
        return Lgrad[:,None] * X

    def hessian(self, params):
        """
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model
            c: 'c' as shape parameter

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}
        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        Lop=Lpdf/Lcdf
        Lgrad_grad=2.0*(1-y)*Lpdf
        Lgrad_grad+=(1-2.0*y)*(1-2.0*Lcdf)*Lop
        Lgrad_grad+=(2.0*y-1.0)*(Lop**2)
        return -np.dot(Lgrad_grad*X.T,X)


    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        bnryfit = super(GenLogit, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)
        discretefit = GenLogitResults(self, bnryfit)
        return BinaryResultsWrapper(discretefit)
    fit.__doc__ = DiscreteModel.fit.__doc__

class GenLogitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Logit Model",
                    "extra_attr" : ""}
    @cache_readonly
    def resid_generalized(self):
        """
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Logit model are defined

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`. This is the same as the `resid_response`
        for the Logit model.
        """
        # Generalized residuals
        return self.model.endog - self.predict()
