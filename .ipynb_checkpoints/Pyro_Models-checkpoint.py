# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


# -

# Suppose we are trying to figure out how much something weighs, but the scale we’re using is unreliable and gives slightly different answers every time we weigh the same object. We could try to compensate for this variability by integrating the noisy measurement information with a guess based on some prior knowledge about the object. We model not only the belief over weight, but also the result of taking a measurement.

def scale(guess):
    weight = pyro.sample('weight', dist.Normal(guess, 1.0))
    return pyro.sample('measurement', dist.Normal(weight, 0.75))


conditioned_scale = pyro.condition(scale, data={'measurement': 9.5}) #constrain the values of sample statements


# pyro.condition is a higher-order function that takes a model and a dictionary of observations and returns a new model that has the same input and output signatures but always uses the given values at observed sample statements

def deferred_conditioned_scale(measurement, guess): 
    return pyro.condition(scale, data={'measurement': measurement})(guess) 


def perfect_guide(guess):
    loc = (0.75**2 * guess + 9.5) / (1 + 0.75**2) # = 9.14
    scale = np.sqrt(0.75**2/(1 + 0.75**2)) # = 0.6
    return pyro.sample('weight', dist.Normal(loc, scale))


# Although we could write out the exact posterior distribution for scale, in general it is intractable to specify a guide that is a good approximation to the posterior distribution of an arbitrary conditioned stochastic function. In fact, stochastic functions for which we can determine the true posterior exactly are the exception rather than the rule.
# What we can do instead is use the top-level function pyro.param to specify a family of guides indexed by named parameters, and search for the member of that family that is the best approximation according to some loss function. This approach to approximate posterior inference is called variational inference.

# +
from torch.distributions import constraints

def scale_parametrized_guide(guess):
    a = pyro.param('a', torch.tensor(guess))
    b = pyro.param('b', torch.tensor(1.), constraint=constraints.positive)
    return pyro.sample('weight', dist.Normal(a, b))


# +
guess = 8.5
pyro.clear_param_store()
svi = pyro.infer.SVI(model=conditioned_scale,
                     guide=scale_parametrized_guide,
                     optim=pyro.optim.SGD({'lr': 0.001, 'momentum': 0.1}),
                     loss=pyro.infer.Trace_ELBO())
losses, a, b = [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(guess))
    a.append(pyro.param('a').item())
    b.append(pyro.param('b').item())
    
plt.plot(losses)
print('a = ', pyro.param('a').item())
print('b = ', pyro.param('b').item())

# +
plt.subplot(1,2,1)
plt.plot([0,num_steps],[9.14,9.14], 'k:')
plt.plot(a)
plt.ylabel('a')

plt.subplot(1,2,2)
plt.ylabel('b')
plt.plot([0,num_steps],[0.6,0.6], 'k:')
plt.plot(b)
plt.tight_layout()
# -


