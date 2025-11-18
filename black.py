from scipy.stats import norm
import numpy as np

def d1(F, K, T, sigma):
    numerator = np.log(F / K) + 1/2 * sigma**2 * T
    denominator = sigma * np.sqrt(T)
    return numerator / denominator

def d2(d1, T, sigma):
    return d1 - sigma * np.sqrt(T)

def call_price(F, K, d1, d2, discount_factor):
    price = discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return price

def put_price(F, K, d1, d2, discount_factor):
    price = discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return price

def call_delta(d1, discount_factor):
    return discount_factor * norm.cdf(d1)

def put_delta(d1, discount_factor):
    return -discount_factor * (norm.cdf(-d1))

def gamma(d1, F, T, sigma, discount_factor):
    coeff = 1 / (F * sigma * np.sqrt(T))
    return discount_factor * coeff * norm.pdf(d1)

def theta(d1, discount_factor, price, r, T, sigma, F):
    coeff = F * sigma / (2 * np.sqrt(T))
    return 1/256 * (r * price - discount_factor * coeff * norm.pdf(d1))

def vega(d1, F, T, discount_factor):
    return 1 / 100 * discount_factor * F * np.sqrt(T) * norm.pdf(d1) 

def vanna(discount_factor, sigma, d1, d2):
    return -1 / 100 * discount_factor * norm.pdf(d1) * d2 / sigma
    
def charm(r, delta, discount_factor, T, d2, d1):
    return 1/256 * (r * delta + discount_factor * 1 / (2 * T) * d2 * norm.pdf(d1))

def volga(discount_factor, F, T, sigma, d1, d2):
    return 1 / 10000 * discount_factor * F * np.sqrt(T) * 1/sigma * d1 * d2 * norm.pdf(d1)

def call_dual_delta(d2, discount_factor):
    return -discount_factor * norm.cdf(d2)

def put_dual_delta(d2, discount_factor):
    return discount_factor * norm.cdf(-d2)

def dual_gamma(discount_factor, K, sigma, T, d2):
    return discount_factor * norm.pdf(d2) / (K * sigma * np.sqrt(T))

def rho(price, T):
    return -T * price / 100
