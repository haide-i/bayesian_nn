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
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
import torch

import matplotlib.pyplot as plt
import numpy as np
# -

# Our model is a simple example: You’ve been given a two-sided coin. You want to determine whether the coin is fair or not, i.e. whether it falls heads or tails with the same frequency. You have a prior belief about the likely fairness of the coin based on two observations:
#
# 1) it’s a standard quarter issued by the US Mint
#
# 2) it’s a bit banged up from years of use
#
# So while you expect the coin to have been quite fair when it was first produced, you allow for its fairness to have since deviated from a perfect 1:1 ratio. So you wouldn’t be surprised if it turned out that the coin preferred heads over tails at a ratio of 11:10. By contrast you would be very surprised if it turned out that the coin preferred heads over tails at a ratio of 5:1—it’s not that banged up.

# +
import math
import os

smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

#assert pyro.__version__.startswith('1.3.0')
pyro.enable_validation(True)

pyro.clear_param_store()

data = [] #create some data 
for _ in range(10):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))


# -

# In Pyro, the model and guide are allowed to be arbitrary stochastic functions provided that
#
# 1) guide doesn’t contain pyro.sample statements with the obs argument
#
# 2) model and guide have the same call signature
#
# Indeed parameters may be created dynamically during the course of inference. In other words the space we’re doing optimization over, which is parameterized by $\theta$ and $\phi$, can grow and change dynamically.
#
# In order to support this behavior, Pyro needs to dynamically generate an optimizer for each parameter the first time it appears during learning. Luckily, PyTorch has a lightweight optimization library (see torch.optim) that can easily be repurposed for the dynamic case.
#
# All of this is controlled by the optim.PyroOptim class, which is basically a thin wrapper around PyTorch optimizers. PyroOptim takes two arguments: a constructor for PyTorch optimizers optim_constructor and a specification of the optimizer arguments optim_args

def model(data):
    alpha0 = torch.tensor(10.0) #define hyperparameters that control the prior distribution
    beta0 = torch.tensor(10.0)
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0)) #sample f from the beta prior
    for i in range(len(data)):
    #observe datapoint i using the bernoulli likelihood Bernoulli(f)
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])


# +
from torch.distributions import constraints

def guide(data):
    # the guide has to take the same arguments and has to have the same names of the random variables as the model
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0), constraint = constraints.positive) #torch tensors,
    beta_q = pyro.param("beta_q", torch.tensor(15.0), constraint=constraints.positive)#requires_grad = True
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q)) #same as in model


# +
adam_params = {'lr': 0.0005, 'betas': (0.90, 0.999)} 
optimizer = Adam(adam_params)

svi = SVI(model, guide, optimizer, loss=Trace_ELBO()) #variational inference algorithm setup

n_steps = 5000
alpha = []
beta = []
for step in range(n_steps):
    svi.step(data)
    alpha.append(pyro.param('alpha_q').item())
    beta.append(pyro.param('beta_q').item())
    if step % 100 == 0:
        print('.', end='')
        
alpha_q = pyro.param('alpha_q').item()
beta_q = pyro.param('beta_q').item()

inferred_mean = alpha_q / (alpha_q + beta_q)
inferred_std = inferred_mean * math.sqrt(beta_q / (alpha_q * (1.0 + alpha_q + beta_q)))

print('mean = ', inferred_mean, ' std = ', inferred_std, ' alpha = ', alpha_q, ' beta = ', beta_q)

# +
alpha = np.asarray(alpha)
beta = np.asarray(beta)

mean = alpha / (alpha + beta)
std = mean * np.sqrt(beta / (alpha * (1.0 + alpha + beta)))


plt.subplot(1,2,1)
#plt.plot([0,n_steps],[9.14,9.14], 'k:')
plt.plot(mean)
plt.ylabel('a')

plt.subplot(1,2,2)
plt.ylabel('b')
#plt.plot([0,num_steps],[0.6,0.6], 'k:')
plt.plot(std)
plt.tight_layout()
