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
import torch
import pyro

pyro.set_rng_seed(101)
# -

# using primitive stochastic functions in pyro by drawing samples from the distribution using pytorch's distribution library

loc = 0. #mean
scale = 1. #variance
normal = torch.distributions.Normal(loc, scale)
x = normal.rsample() #draw a sample
print("sample", x)
print("log prob", normal.log_prob(x)) #score the sample


# pyro's library pyro.distributions is a thin wrapper around torch.distributions

# ### Building a simple model
# Concrete model: bunch of data with daily mean temperatures and cloud cover - we want to reason about how temperature interacts with whether it was sunny or cloudy

def weather_torch():
    cloudy = torch.distributions.Bernoulli(0.3).sample() #random variable drawn from Bernoulli distribution
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny' 
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy] #set mean temp of weather cloudy and sunny
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy] #set variance of weather cloudy and sunny
    temp = torch.distributions.Normal(mean_temp, scale_temp).rsample() #draw sample of temp
    return cloudy, temp.item()


# to turn weather into a pyro program, we have to replace the torch calls with the pyro calls
#
# Drawing a sample from pyro works as following:

x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale)) #generating named sample
print(x)


# Converting weather into pyro:

# +
def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3)) #variable named cloudy, follows bernoulli
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny' 
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy] #set mean temp of weather cloudy and sunny
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy] #set variance of weather cloudy and sunny
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp)) #draw sample of temp
    return cloudy, temp.item()

for _ in range(3):
    print(weather(), "  ", weather_torch())


# -

# Building off a simple model:

def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream


# +
def geometric(p, t=None): #count how often the experiment fails before producing a 1
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(0.3))
    if x.item() == 1:
        return 0
    else:
        return 1+ geometric(p, t+1)
    
print(geometric(0.1))


# +
def normal_product(loc, scale):
    z1 = pyro.sample('z1', pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample('z2', pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y

def make_normal_normal():
    mu_latent = pyro.sample('mu_latent', pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn

print(make_normal_normal()(1.))
