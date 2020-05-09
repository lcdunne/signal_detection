# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:37:29 2020

@author: L
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Some global helper function
def accumulate(*arrays):
    accumulated = []
    for a in arrays:
        accum = np.cumsum(a)
        accumulated.append(accum)

    # In case it's just a single array pop it out
    if len(accumulated) == 1:
        accumulated = accumulated.pop()

    return accumulated


def roc(*arrays, truncate=False):
    # Assumes inputs are observed counts at each decision point on scale
    rates = []
    for a in arrays:
        # accumulate it then calculate it
        accum = accumulate(a)
        frequency = np.array([(x + i / len(accum)) / (max(accum) + 1) for i, x in enumerate(accum, start=1)])
        if truncate:
            # trim off the last element
            frequency = frequency[:-1]
        rates.append(frequency)

    if len(rates) == 1:
        rates = rates.pop()

    return rates


def z_score(*arrays):
    # compute the z score of ROC data
    z_arrays = []
    for a in arrays:
        z_arrays.append(stats.norm.ppf(a))

    if len(z_arrays) == 1:
        z_arrays = z_arrays.pop()

    return z_arrays

def auc(roc_signal, roc_noise):
    return np.trapz(roc_signal, roc_noise)
    



class Model:   
    def __init__(self):
        self.name = 'MODEL'
    
    def add_data(self, signal, noise):
        '''
        Adds signal and noise data to model attributes. Computes the following transformations for both arrays:
            - Conversion from observed count data to ROC data, truncating the final element (which is always 1)
            - Conversion from ROC data to Z-ROC.
            - Conversion from observed count data to accumulated data (also used as part of the ROC calculation). This is required for assessing fit with G-squared.
            - Maximum values for both of accumulated signal and accumulated noise arrays. Also required for assessing fit with G-squared.
            
        Parameters
        ----------
        signal : array-like
            Array denoting response count for each rating level in cases where signal truly present. *Assumes order is from most positive to least positive.
        noise : array-like
            Array denoting response count for each rating level in cases where signal truly absent. *Assumes order is from most positive to least positive.

        '''
        self.signal = np.array(signal)
        self.noise = np.array(noise)
        # Do some computations
        self.signal_roc, self.noise_roc = roc(self.signal, self.noise, truncate=True)
        self.signal_z, self.noise_z = z_score(self.signal_roc, self.noise_roc)
        self.signal_acc, self.noise_acc = accumulate(self.signal, self.noise)
        self.signal_acc_max = self.signal_acc[-1]  # extract the maximum
        self.signal_acc = self.signal_acc[:-1]  # truncate signal_acc
        self.noise_acc_max = self.noise_acc[-1] # Extract maximum...
        self.noise_acc = self.noise_acc[:-1] # Truncate noise_acc.
        self.scale_range = len(self.signal)
        self.data_added = True

        return self
    
    def _set_optimization_inputs(self, signal, noise):
        if not self.data_added:
            # If the data haven't been added yet, add them
            self.add_data(signal, noise)
                
        # Create x0
        parameter_names = list(self._parameters.keys()) # 
        parameter_values = list(self._parameters.values()) # used to be self.parameters
        
        
        if self._use_c:
            initial_c_values = np.random.sample(self.scale_range-1)
            
            labels = np.concatenate([parameter_names, np.arange(1, self.scale_range)]) # needs to be same length
            c_bounds = [(None, None) for c in initial_c_values]
            bounds = self.boundaries + c_bounds
            inputs = {l:p for l, p in zip(parameter_names, parameter_values)}
            
            inputs['c'] = initial_c_values
        else:
            initial_c_values = np.array([])
            labels = parameter_names
            bounds = self.boundaries
            inputs = {l:p for l, p in zip(parameter_names, parameter_values)}
            
        x0 = np.concatenate([parameter_values, initial_c_values])
        inputs = {'target_input':{'x0':x0, 'labels':labels, 'bounds':bounds},
                  'fit_inputs':inputs}
        
        return inputs
    
    
    def optifit(self):
        self._optimization_results = Optimizer().optimize(self)
        self.optimal_parameters = self._optimization_results['parameters'] # Dictionary
        self.optimized = True
        if self._use_c:
            self.optimal_c = self._optimization_results['c']
        self.x, self.y = self.fit(**self.optimal_parameters)
    
    def get_strength(self, d=None, c=None):
        # Does this only make sense for the dual process model? I think in the SDT this is simply the Hit rate?
        # e.g. recognition familiarity
        
        # Check if it's optimized...
        if self.optimized:
            # Check if it has optimal c values
            if hasattr(self , 'optimal_c'):
                c = self.optimal_c
                # Check if it has a d value (it should, but just in case)
                if 'd' in self.optimal_parameters.keys():
                    d = self.optimal_parameters['d']
            else:
                # Doesn't have optimal c values
                raise ValueError("Can't compute strength estimate for threshold process")
        
        # Will error if d or c not provided
        cut_point = int(np.median(np.arange(0, len(c))))
        strength =  stats.norm.cdf(d / 2 - c[cut_point])
        
        return strength
    
    
    def plot(self, data=False, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        if data:
            if self.data_added:
                ax.plot(self.noise_roc, self.signal_roc, 
                        marker='o', markersize=10, c='k', label='data', zorder=3)
        
        ax.plot([0,1], [0,1], c='gray', ls='dashed')
        ax.plot(self.x, self.y, label=self.name)
        
        ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_title('ROC')
        ax.legend(loc='lower right')
        return ax


class HighThreshold(Model):
    # Must return the expected values, given inputs of R and noise
    # We get the noise from the add_data() method. Noise should be none
    def __init__(self):
        self.name = 'HT'
        self.boundaries = [(0, 1)] # boundary for the model parameters
        self._use_c = False
        self.data_added = False
        self.optimized = False # Initial state
        self._parameters = {'R':0.99}
        
    
    def fit(self, R=None):
        # If wanting to plot, just set noise roc to be [0, 1]
        
        if self.optimized:
            noise_roc = np.array([0,1])
        else:
            noise_roc = np.array(self.noise_roc)
        
        x = noise_roc
        y = (1 - R) * noise_roc + R
        
        return x, y
    

class SignalDetection(Model):
    def __init__(self, equal_variance=True):
        
        self.name = 'SDT'
        self.equal_variance = equal_variance
        
        if self.equal_variance:
            self.boundaries = [(None, None), (1, 1)] # boundary for the model parameters free
        else:
            self.name = 'UV' + self.name # prefix it
            self.boundaries = [(None, None), (0, None)] # second argument (i.e. signal_var) restricted to 1
        
        self._use_c = True
        self.data_added = False
        self.optimized = False
        self._parameters = {'d': 0, 'signal_var':1}
    
    
    def fit(self, d=None, signal_var=None, c=None):
        
        if signal_var is None:
            signal_var = self._parameters['signal_var']
        
        if c is None:
            c = np.linspace(-5, 5, 500)
        else:
            c = np.array(c)
        
        
        x = stats.norm.cdf(-d / 2 - c)
        y = stats.norm.cdf(d / 2 - c, scale=signal_var)
        
        return x, y


class DualProcess(Model):
    def __init__(self):
        self.name = 'DPSDT'
        self.boundaries = [(0, 1), (None, None)] # boundary for the model parameters
        self._use_c = True
        self.data_added = False
        self.optimized = False
        self._parameters = {'R': 0.99, 'd':0}
    
    
    def fit(self, R=None, d=None, c=None):
        
        if c is None:
            c = np.linspace(-5, 5, 500)
        else:
            c = np.array(c)
        
        x = stats.norm.cdf(-d / 2 - c) # Expected noise
        y = R + (1 - R) * stats.norm.cdf(d / 2 - c) # Expected signal
        return x, y
    
    def process_estimates():
        pass

class DoubleDualProcess(Model):
    def __init__(self):
        self.name = '2DPSDT'
        self.boundaries = [(0,1), (-1,1), (None, None)]
        self._use_c = True
        self.data_added = False
        self.optimized = False
        self._parameters = {'R':0.99, 'Rx':0, 'd':0}
    
    def fit(self, R=None, Rx=None, d=None, c=None):
        
        if c is None:
            c = np.linspace(-5, 5, 500)
        else:
            c = np.array(c)
        
        x = (1 - Rx) * stats.norm.cdf(-d / 2 - c) # Expected noise
        y = R + (1 - R) * stats.norm.cdf(d / 2 - c) # Expected signal
        return x, y
        
    
    

class Optimizer:
    def __init__(self):
        return
    
    def optimize(self, model):
        
        optimization_inputs = model._set_optimization_inputs(model.signal, model.noise)# Initialise the inputs
        x0 = optimization_inputs['target_input']['x0']# Extract the x0 input argument
        
        bounds = optimization_inputs['target_input']['bounds']# Extract the bounds
        
        #-Minimization-----------------------------------------------------------------------#
        result = minimize(fun=self.sum_gsquared, x0=x0, args=(model), bounds=bounds, tol=1e-6)
        #-------------------------------------------------------------------------------------#
        
        # Create a decent output
        starting_state = {str(k):v for k, v in zip(optimization_inputs['target_input']['labels'], x0)}
        optimized_state = {str(k):v for k, v in zip(optimization_inputs['target_input']['labels'], result.x)}
        
        results = {'objective': result.fun, 'success': result.success, 'n_iterations': result.nit}
        results['starting_state'] = starting_state
        results['final_state'] = optimized_state
        results['optimize_result'] = result
        
        non_c_labs = list(model._parameters.keys()) # get labels of x0 corresponding to non-c variables
        if model._use_c:
            c_start_idx = len(model._parameters) # get the index of x0 where c begins
            non_c_labs = list(model._parameters.keys()) # get labels of x0 corresponding to non-c variables
            non_c_vals = result.x[:c_start_idx] # get the values of them
            results['parameters'] = {str(l):p for l, p in zip(non_c_labs, non_c_vals)}
            results['c'] = result.x[c_start_idx:]
        else:
            results['parameters'] = {str(l):p for l, p in zip(non_c_labs, result.x)}

        return results
    
    def sum_gsquared(self, x0, model):
        # Objective function to minimize
        if model._use_c:
            c_start_idx = len(model._parameters) # get the index of x0 where c begins 
            non_c_labs = list(model._parameters.keys()) # get labels of x0 corresponding to non-c variables
            non_c_vals = x0[:c_start_idx] # get the values of them
            
            # c_labs = np.arange(1,len(x0[c_start_idx:]))#inputs['target_input']['labels'][c_start_idx:]
            c_vals = x0[c_start_idx:]
            to_fit = {l:p for l, p in zip(non_c_labs, non_c_vals)}
            to_fit['c'] = c_vals
        else:
            # Just get the parameters
            parameter_names = list(model._parameters.keys()) 
            to_fit = {l:p for l, p in zip(parameter_names, x0)}
            
        expected_noise, expected_signal = model.fit(**to_fit) # Unpack model.parameters dictionary
        
        signal_g_squared = self._gtest(model.signal_acc, model.signal_roc, expected_signal, model.signal_acc_max)
        noise_g_squared = self._gtest(model.noise_acc, model.noise_roc, expected_noise, model.noise_acc_max)
        sum_of_g_squared = np.sum([signal_g_squared, noise_g_squared])
        
        return sum_of_g_squared
    
    def _gtest(self, x_acc, x_roc, expected, x_max):
        '''
        Two-way log-likelihood G-test
        Implementation issues:
            Log expressions can throw errors from negative/inf values during optimization, anddiv/0 etc. Numpy just warns and continues. Needs fixing somehow.
    
        Parameters
        ----------
        x : array_like
            Accumulated data.
        x_freq : array_like
            `x` Transformed to frequency domain.
        expected : array_like
            Expected values of `x` after fitting a model with specific parameter(s). 
        x_max : scalar
            Maximum value of the accumulated data `x`.
    
        Returns
        -------
        g_squared
            G squared statistic for each element in expected.
    
        '''        
        a = 2 * x_acc * np.log(x_roc / expected)
        b = 2 * (x_max - x_acc) * np.log((1 - x_roc) / (1 - expected))
        g_squared = a + b
        
        return g_squared

if __name__ == '__main__': 
    # Input signal & noise data here
    signal = [431, 218, 211, 167, 119, 69]
    noise = [102, 161, 288, 472, 492, 308]
    
    # Initialise a model and add signal & noise data to it
    HT = HighThreshold().add_data(signal, noise)
    HT.optifit() # optimize and fit, which creates a dictionary as an instance attribute that we can call
    
    # Initialise another model
    SDT = SignalDetection().add_data(signal, noise)
    SDT.optifit()
    
    UVSDT = SignalDetection(equal_variance=False).add_data(signal, noise)
    UVSDT.optifit()
    
    # Another
    DP = DualProcess().add_data(signal, noise)
    DP.optifit()
    
    # And another
    dDP = DoubleDualProcess().add_data(signal, noise)
    dDP.optifit()
    
    # After this we can make a plot
    fig, ax = plt.subplots(figsize=(8,8), dpi=100)
    HT.plot(data=True, ax=ax) # specify whether to plot observed data as well
    SDT.plot(ax=ax)
    UVSDT.plot(ax=ax)
    DP.plot(ax=ax)
    dDP.plot(ax=ax)
    plt.show()
    