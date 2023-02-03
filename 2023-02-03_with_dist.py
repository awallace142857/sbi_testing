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

(ids,real_x,real_coords) = pickle.load(open('../real_data_errs.pkl', 'rb'))
delete = np.where(real_x[:,1:9]>20)[0]
real_x = np.delete(real_x,list(set(delete)),axis=0)
delete = np.where(real_x[:,9:18]>4)[0]
real_x = np.delete(real_x,list(set(delete)),axis=0)
delete = np.where(real_x[:,2]<5)[0]
real_x = np.delete(real_x,list(set(delete)),axis=0)
delete = np.where(real_x[:,1]-real_x[:,3]<-0.5)[0]
real_x = np.delete(real_x,list(set(delete)),axis=0)
delete = np.where(real_x[:,1]-real_x[:,3]>2)[0]
real_x = np.delete(real_x,list(set(delete)),axis=0)
real_mags = real_x[:,1:9]
real_pars = real_x[:,0]
real_errs = real_x[:,10:18]
real_par_errs = real_x[:,9]
tracks = get_ichrone('mist', tracks=True)
def extinction(wavelength,distance):
	return 0.014*(wavelength/4.64e-7)**(-1.5)

def get_errors(all_app_mags):
	bins = 40
	bottom = np.min(real_mags[:,1])
	top = np.max(real_mags[:,1])
	left = np.min(real_mags[:,0]-real_mags[:,2])
	right = np.max(real_mags[:,0]-real_mags[:,2])
	x_bin_width = (right-left)/bins
	y_bin_width = (top-bottom)/bins
	errs = np.zeros((all_app_mags.shape[0],all_app_mags.shape[1]+1))
	for ii in range(bins):
		for jj in range(bins):
			bin_left = left+ii*x_bin_width
			bin_right = left+(ii+1)*x_bin_width
			bin_bottom = bottom+jj*y_bin_width
			bin_top = bottom+(jj+1)*y_bin_width
			els = np.where((real_mags[:,1]>=bin_bottom) & (real_mags[:,1]<=bin_top) & (real_mags[:,0]-real_mags[:,2]>=bin_left) & (real_mags[:,0]-real_mags[:,2]<=bin_right))[0]
			test_els = np.where((all_app_mags[:,1]>=bin_bottom) & (all_app_mags[:,1]<=bin_top) & (all_app_mags[:,0]-all_app_mags[:,2]>=bin_left) & (all_app_mags[:,0]-all_app_mags[:,2]<=bin_right))[0]
			if len(els)==0:
				continue
			rand_num = len(els)*np.random.random(len(test_els))
			par_errs = real_par_errs[els[np.array(np.floor(rand_num),dtype=np.int64)]]
			errs[test_els,0] = par_errs
			mag_errs = real_errs[els[np.array(np.floor(rand_num),dtype=np.int64)]]
			errs[test_els,1:9] = mag_errs
	return errs
	
NUM_SAMPLES = 10_000
NUM_SIMULATIONS = 1_000
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
        distance_bounds=(1.0, 1000.0), 
        M1_alpha=1.0,
        M1_beta=5.0,
        m_h_alpha=10.0,
        m_h_beta=2.0,
        m_h_scale=3.0,
        return_numpy=False
    ):
        self.bounds = dict(
            lower_bound=tt([M1_bounds[0], q_bounds[0], tau_bounds[0], m_h_bounds[0], distance_bounds[0]]),
            upper_bound=tt([M1_bounds[1], q_bounds[1], tau_bounds[1], m_h_bounds[1], distance_bounds[1]])
        )
        self.lower = tt([M1_alpha, 1.0, 1.0, m_h_alpha, 1.0])
        self.upper = tt([M1_beta, 1.0, 1.0, m_h_beta, 1.0])
        m_h_mode = (m_h_alpha - 1)/(m_h_alpha + m_h_beta - 2)
        loc = tt([M1_bounds[0], q_bounds[0], tau_bounds[0], -m_h_mode * m_h_scale, distance_bounds[0]])
        scale = tt([M1_bounds[1], q_bounds[1], tau_bounds[1], m_h_scale, distance_bounds[1]])
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

def apparent_magnitudes(M, distance):
    return M + 5 * (np.log10(np.array(distance)) - 1)

def main_sequence_observables(theta):
    m1, q, tau, m_h, distance = theta
    M, point = absolute_magnitudes(m1, q, tau, m_h, full_output=True)    
    # exclude things where the primary is not a main-sequence star.
    # Table 2 of https://waps.cfa.harvard.edu/MIST/README_tables.pdf
    primary_after_zams = (point.eep_0 >= 200)
    primary_before_tams = (point.eep_0 <= 454)
    primary_is_ms = primary_after_zams & primary_before_tams
    M[:, ~primary_is_ms] = np.nan
    m = apparent_magnitudes(M, distance)
    # Return parallax in  and apparent magnitudes.
    # 100 pc = 10 mas, so plx [mas] = 1000/distance [pc]
    plx = 1000.0/distance
    return np.vstack([plx, m])

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
        print(trial_theta)
        all_dists = np.random.uniform(1,1000,size=len(trial_theta)*100)
        theta_arr = np.zeros((len(all_dists),5))
        theta_arr[:,0:4] = np.repeat(np.array(trial_theta[:,0:4]),100,axis=0)
        theta_arr[:,4] = all_dists
        trial_theta = tt(theta_arr)
        print(trial_theta)
        sys.exit()
        trial_x = simulator(trial_theta)
        print(trial_x.shape)
        keep = np.all(np.isfinite(trial_x.numpy()), axis=1)
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

def add_distance(theta_old,x_abs,n_dist,dist_bounds):
	all_dists = np.random.uniform(dist_bounds[0],dist_bounds[1],size=len(theta_old)*n_dist)
	theta_arr = np.zeros((len(all_dists),5))
	x_arr = np.zeros((len(all_dists),18))
	theta_arr[:,0:4] = np.repeat(np.array(theta_old),n_dist,axis=0)
	theta_arr[:,4] = all_dists
	x_abs_arr = np.repeat(np.array(x_abs),n_dist,axis=0)
	for ii in range(8):
		x_arr[:,ii+1] = x_abs_arr[:,ii]+5*np.log10(all_dists/10)
	x_arr[:,0] = 1000./all_dists
	x_clean = x_arr[:,0:9].copy()
	errs = get_errors(x_arr[:,1:9])
	x_arr[:,9:18] = errs
	x_means = x_arr[:,0:9]
	for ii in range(9):
		x_arr[:,ii] = np.random.normal(x_means[:,ii],x_arr[:,ii+9])
	zeros = np.where(errs==0)[0]
	zeros = list(set(zeros))
	theta_arr = np.delete(theta_arr,zeros,axis=0)
	x_arr = np.delete(x_arr,zeros,axis=0)
	x_clean = np.delete(x_clean,zeros,axis=0)
	trial_theta = tt(theta_arr)
	trial_x = tt(x_arr)
	return (trial_theta,trial_x,tt(x_clean))
    
def absolute_magnitude_simulator(theta):
    x = absolute_magnitudes(*theta)
    x += np.random.normal(0, 1e-5, size=x.shape)
    return x



simulator = main_sequence_observables

sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)

(theta, x) = pickle.load(open('sbi_theta_x_absolute.pkl','rb'))
print(theta,x)
(theta, x, x_clean) = add_distance(theta,x,100,[1,1000])
print(theta,x,x_clean)
pickle.dump((theta,x,x_clean),open('sbi_theta_x_apparent.pkl','wb'))