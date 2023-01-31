import torch
import numpy as np
import matplotlib.pyplot as plt
from isochrones import get_ichrone
from scipy import stats
import os,sys,pickle

from sbi import utils
from sbi import analysis
from sbi.inference.base import infer

# First, some settings.
from sbi.inference import SNPE as method
from scipy import interpolate
num_simulations = 5_000
num_samples = 15_000

labels = ("M1", "q", "age", "[M/H]", "log10(distance)")
short_labels = ("m1","q","age","feh","dist")
bounds = np.array([
    [0.4,  2.5], # M1
    [0,      1], # q
    [0.2,   10], # (Gyr)
    [-2, 0.5],  # metallicity
    [50, 5000] # log(distance)

])
tracks = get_ichrone('mist', tracks=True)

def binary_color_mag_isochrones(m1, q, age, fe_h, dist):
	# isochrones.py needs log10(Age [yr]).
	# Our age is in Gyr, so we take log10(age * 10^9) = log10(age) + 9
	properties = tracks.generate_binary(
	m1,
	q * m1,
	np.log10(age) + 9,
	fe_h,
	bands=["G", "BP", "RP", "J", "H", "K", "W1", "W2"]
	)
	all_abs_mags = np.array([properties.BP_mag.values,properties.G_mag.values,properties.RP_mag.values,properties.J_mag.values,properties.H_mag.values,properties.K_mag.values,properties.W1_mag.values,properties.W2_mag.values])
	#all_abs_mags = all_abs_mags[0:3]
	dist = np.float64(dist)
	all_app_mags = all_abs_mags+5*(np.log10(dist)-1)
	par = 1000/dist #parallax in mas
	return_vals = [np.array([par])]
	for ii in range(len(all_app_mags)):
		return_vals.append(all_app_mags[ii])
	return_vals = np.stack(return_vals,axis=0)#np.array(return_vals)
	return return_vals.T
	
def simulator(theta):
    return torch.tensor(binary_color_mag_isochrones(*theta))

# Set priors.
from torch.distributions import (Uniform, Beta, Pareto)
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from torch import tensor as tt

prior = utils.BoxUniform(low=bounds.T[0], high=bounds.T[1])

from sbi.inference import prepare_for_sbi, simulate_for_sbi

sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)

inference = method(prior)

# Generate the simulations. 
# We do this ourselves (instead of using simulate_for_sbi) because if we don't then many will be NaNs
# and we end up with fewer simulations than we want.
def simulate_for_sbi_strict(simulator, proposal, num_simulations, max_trials=np.inf):
	#(theta_sim,x_sim) = pickle.load(open('theta_x_train.pkl','rb'))
	#theta_sim[:,4] = 10**theta_sim[:,4]
	num_trials, num_simulated, theta, x = (0, 0, [], [])
	while num_simulated < num_simulations:
		N = num_simulations - num_simulated
		print(f"Running {N} simulations")
		os.system('rm -rf current_*')
		textFile = open('current_'+str(N),'w')
		_theta = proposal.sample((N, ))
		_x = simulator(_theta)
		x_arr = np.array(_x)
		keep = np.where((x_arr[:,1]-x_arr[:,3]>0.9) & (x_arr[:,1]-x_arr[:,3]<1.1) & (x_arr[:,2]-5*np.log10(100/x_arr[:,0])>6) & (x_arr[:,2]-5*np.log10(100/x_arr[:,0])<6.5))[0]
		#if len(_theta[keep])>0:
		#print(_theta[keep][ii],x_arr[keep][ii],binary_color_mag_isochrones(_theta[keep][ii][0],_theta[keep][ii][1],_theta[keep][ii][2],_theta[keep][ii][3],_theta[keep][ii][4]))
		theta.extend(np.array(_theta[keep]))
		x.extend(x_arr[keep])
		num_trials += 1
		num_simulated += sum(keep)
		if num_trials > max_trials:
			print(f"Warning: exceeding max trials ({max_trials}) with {num_simulated} / {num_simulations} simulations")
			break
	#theta = theta_sim
	#x = x_sim
	theta = torch.tensor(np.vstack(theta))
	x = torch.tensor(np.vstack(x))
	return (theta, x)
	
"""theta, x = simulate_for_sbi_strict(sbi_simulator, sbi_prior, num_simulations)
pickle.dump((theta, x), open('sbi_theta_x_no_errors.pkl', 'wb'))
#sys.exit()"""
(theta, x, x_clean) = pickle.load(open('sbi_theta_x_with_errors.pkl', 'rb'))
#x = x[:,(0,1,2,3,9,10,11,12)]
x = torch.tensor(x_clean.astype('float32'))
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
pickle.dump((posterior, (theta, x)), open('sbi_posterior_no_errors.pkl', 'wb'))
#sys.exit()
(posterior, (theta, x)) = pickle.load(open('sbi_posterior_no_errors.pkl', 'rb'))#(true_theta,true_obs) = pickle.load(open('binary_data.pkl','rb'))
#x = true_obs#[:,(0,1,2,3,9,10,11,12)]
ii = 6
sample = posterior.sample(
		(num_samples,),
		x=x[ii],
		show_progress_bars=False
	)
pickle.dump((np.array(theta[ii]),np.array(x[ii]),np.array(sample)), open('sbi_ex_test_no_errors.pkl', 'wb'))