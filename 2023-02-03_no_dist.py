import torch
import numpy as np
import matplotlib.pyplot as plt
from isochrones import get_ichrone
from scipy import stats
import os,sys,pickle

from sbi import utils
from sbi import analysis
from sbi.inference.base import infer, Distribution
#from chainconsumer import ChainConsumer

from sbi.inference import SNPE as method, prepare_for_sbi, simulate_for_sbi
from sbi.utils import process_prior

from scipy import interpolate

# Set priors.
from torch.distributions import (Uniform, Beta, Pareto, Independent)
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from torch import tensor as tt

#torch.set_default_tensor_type('torch.DoubleTensor')

from collections import OrderedDict

tracks = get_ichrone('mist', tracks=True)

NUM_SAMPLES = 10_000
NUM_SIMULATIONS = 100_000
BANDS = ("BP", "G", "RP", "J", "H", "K", "W1", "W2")

# TODO: These bounds
label_names = [r"$M_1$", r"$q$", r"$\tau$", "[M/H]", "distance"]

class StellarPrior:

    def __init__(
        self, 
        M1_bounds=(0.4, 5.0),   
        q_bounds=(0.0, 1.0),    
        tau_bounds=(1.0, 10.0), 
        m_h_bounds=(-2.0, 0.5), 
        M1_alpha=1.0,
        M1_beta=5.0,
        m_h_alpha=10.0,
        m_h_beta=2.0,
        m_h_scale=3.0,
        return_numpy=False
    ):
        self.bounds = dict(
            lower_bound=tt([M1_bounds[0], q_bounds[0], tau_bounds[0], m_h_bounds[0]]),
            upper_bound=tt([M1_bounds[1], q_bounds[1], tau_bounds[1], m_h_bounds[1]])
        )
        self.lower = tt([M1_alpha, 1.0, 1.0, m_h_alpha])
        self.upper = tt([M1_beta, 1.0, 1.0, m_h_beta])
        m_h_mode = (m_h_alpha - 1)/(m_h_alpha + m_h_beta - 2)
        loc = tt([M1_bounds[0], q_bounds[0], tau_bounds[0], -m_h_mode * m_h_scale])
        scale = tt([M1_bounds[1], q_bounds[1], tau_bounds[1], m_h_scale])
        self.return_numpy = return_numpy
        self.dist = Independent(
            TransformedDistribution(
                Beta(self.lower, self.upper, validate_args=False),
                AffineTransform(loc=loc, scale=scale)
            ),
            1
        )

    def sample(self, sample_shape=torch.Size([])):
        samples = self.dist.sample(sample_shape)
        return samples.numpy() if self.return_numpy else samples

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = self.dist.log_prob(values)
        return log_probs.numpy() if self.return_numpy else log_probs


custom_prior = StellarPrior()
prior, *_ = process_prior(
    custom_prior,
    custom_prior_wrapper_kwargs=custom_prior.bounds
)
def absolute_magnitudes(m1, q, tau, m_h, full_output=False):
    point = tracks.generate_binary(
        m1,
        q * m1,
        np.log10(tau) + 9,
        m_h,
        bands=BANDS
    )
    M = np.array([getattr(point, f"{band}_mag").values for band in BANDS])
    return (M, point) if full_output else M

def main_sequence_observables(theta):
    m1, q, tau, m_h = theta
    M, point = absolute_magnitudes(m1, q, tau, m_h, full_output=True)    
    # exclude things where the primary is not a main-sequence star.
    # Table 2 of https://waps.cfa.harvard.edu/MIST/README_tables.pdf
    primary_after_zams = (point.eep_0 >= 200)
    primary_before_tams = (point.eep_0 <= 454)
    primary_is_ms = primary_after_zams & primary_before_tams
    M[:, ~primary_is_ms] = np.nan
    return M


def main_sequence_simulator(theta):
    values = main_sequence_observables(theta)
    values += np.random.normal(0, 1e-5, size=values.shape)
    return values


def simulate_for_sbi_strict(simulator, proposal, num_simulations, max_trials=np.inf):
    num_trials, num_simulated, theta, x = (0, 0, [], [])
    while num_simulated < num_simulations:
        N = num_simulations - num_simulated
        print(f"Running {N} simulations")
        trial_theta = proposal.sample((N, ))
        trial_x = torch.tensor(absolute_magnitudes(np.array(trial_theta)[:,0],np.array(trial_theta)[:,1],np.array(trial_theta)[:,2],np.array(trial_theta)[:,3]).T)
        keep = np.all(np.isfinite(trial_x.numpy()),axis=1)
        theta.extend(np.array(trial_theta[keep]))
        x.extend(np.array(trial_x[keep]))
        assert len(theta) == len(x)
        num_trials += 1
        num_simulated += sum(keep)
        if num_trials > max_trials:
            print(f"Warning: exceeding max trials ({max_trials}) with {num_simulated} / {num_simulations} simulations")
            break
    theta = torch.tensor(np.array(theta))
    x = torch.tensor(np.array(x))
    return (theta, x)


def absolute_magnitude_simulator(theta):
    x = absolute_magnitudes(*theta)
    x += np.random.normal(0, 1e-5, size=x.shape)
    return x



simulator = main_sequence_observables

sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)

theta, x = simulate_for_sbi_strict(sbi_simulator, sbi_prior, NUM_SIMULATIONS)

pickle.dump((theta,x),open('sbi_theta_x_absolute.pkl','wb'))