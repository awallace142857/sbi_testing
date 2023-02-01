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
num_simulations = 100_000
num_samples = 15_000

labels = ("M1", "q", "age", "[M/H]", "distance")
short_labels = ("m1","q","age","feh","dist")
bounds = np.array([
    [0.4,  2.5], # M1
    [0,      1], # q
    [0.2,   10], # (Gyr)
    [-2, 0.5]  # metallicity

])

(ids,real_x,real_coords) = pickle.load(open('real_data_errs.pkl', 'rb'))
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
for ii in range(8):
	real_mags[:,ii]-=5*np.log10(100./real_pars)
real_errs = real_x[:,10:18]
tracks = get_ichrone('mist', tracks=True)
def extinction(wavelength,distance):
	return 0.014*(wavelength/4.64e-7)**(-1.5)

def get_errors(all_abs_mags):
	bins = 40
	bottom = np.min(real_mags[:,1])
	top = np.max(real_mags[:,1])
	left = np.min(real_mags[:,0]-real_mags[:,2])
	right = np.max(real_mags[:,0]-real_mags[:,2])
	x_bin_width = (right-left)/bins
	y_bin_width = (top-bottom)/bins
	errs = np.zeros((all_abs_mags.shape[0],all_abs_mags.shape[1]))
	for ii in range(bins):
		for jj in range(bins):
			bin_left = left+ii*x_bin_width
			bin_right = left+(ii+1)*x_bin_width
			bin_bottom = bottom+jj*y_bin_width
			bin_top = bottom+(jj+1)*y_bin_width
			els = np.where((real_mags[:,1]>=bin_bottom) & (real_mags[:,1]<=bin_top) & (real_mags[:,0]-real_mags[:,2]>=bin_left) & (real_mags[:,0]-real_mags[:,2]<=bin_right))[0]
			test_els = np.where((all_abs_mags[:,1]>=bin_bottom) & (all_abs_mags[:,1]<=bin_top) & (all_abs_mags[:,0]-all_abs_mags[:,2]>=bin_left) & (all_abs_mags[:,0]-all_abs_mags[:,2]<=bin_right))[0]
			if len(els)==0:
				continue
			rand_num = len(els)*np.random.random(len(test_els))
			mag_errs = real_errs[els[np.array(np.floor(rand_num),dtype=np.int64)]]
			errs[test_els] = mag_errs
	return errs
	
def color_mag_no_errors(m1, q, age, fe_h, log_dist):
    properties = tracks.generate_binary(
        m1,
        q * m1,
        np.log10(age) + 9,
        fe_h,
        bands=["G", "BP", "RP", "J", "H", "K", "W1", "W2"]
    )
    all_abs_mags = np.array([properties.BP_mag.values,properties.G_mag.values,properties.RP_mag.values,properties.J_mag.values,properties.H_mag.values,properties.K_mag.values,properties.W1_mag.values,properties.W2_mag.values])
    #all_abs_mags = all_abs_mags[0:3]
    dist = np.float64(10**log_dist)
    all_app_mags = all_abs_mags+5*(np.float64(log_dist)-1)
    par = 1000/dist
    return_vals = [par]
    return_vals.extend(all_app_mags)
    return_vals = np.array(return_vals)
    return return_vals.T

def binary_color_mag_isochrones(m1, q, age, fe_h):
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
	all_errs = np.zeros((1,8))#get_errors(all_app_mags.T)
	#all_errs[np.where(np.isnan(all_errs))] = 0.1
	if type(np.float64(m1))==np.float64:
		n_par = 1
	else:
		n_par = len(m1)
	return_vals = []
	for ii in range(len(all_abs_mags)):
		all_abs_mags[ii] = np.random.normal(all_abs_mags[ii],all_errs[:,ii],n_par)
		return_vals.append(all_abs_mags[ii])
	return_vals.extend(all_errs.T)
	return_vals = np.stack(return_vals,axis=0)#np.array(return_vals)
	return return_vals.T

def binary_color_mag_data_sets(m1, q, age, fe_h):
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
	all_errs = np.zeros((len(m1),8))#get_errors(all_app_mags.T)
	#all_errs[np.where(np.isnan(all_errs))] = 0.1
	if type(np.float64(m1))==np.float64:
		n_par = 1
	else:
		n_par = len(m1)
		#par_er = par_er*np.ones(n_par)
	return_vals = []
	for ii in range(len(all_abs_mags)):
		all_abs_mags[ii] = np.random.normal(all_abs_mags[ii],all_errs[:,ii],n_par)
		return_vals.append(all_abs_mags[ii])
	return_vals.extend(all_errs.T)
	#print(return_vals)
	return_vals = np.stack(return_vals,axis=0)#np.array(return_vals)
	return return_vals.T
	
def simulator(theta):
    return torch.tensor(binary_color_mag_isochrones(*theta))

def simulate_data_sets(theta):
    return torch.tensor(binary_color_mag_data_sets(*theta))
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
		#_theta, _x = simulate_for_sbi(simulator, proposal=proposal, num_simulations=N)
		errs = get_errors(_x[:,0:8])
		x_arr = np.array(_x)
		x_arr[:,8:16] = errs
		x_means = x_arr[:,0:8]
		for ii in range(8):
			x_arr[:,ii] = np.random.normal(x_means[:,ii],x_arr[:,ii+8])
		zeros = np.where(errs==0)[0]
		zeros = list(set(zeros))
		keep = np.all(np.isfinite(_x).numpy(), axis=1)
		keep[zeros] = False
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
"""#sys.exit()          
theta, x = simulate_for_sbi_strict(sbi_simulator, sbi_prior, num_simulations)
pickle.dump((theta, x), open('sbi_theta_x_no_dist.pkl', 'wb'))
#sys.exit()
(theta, x) = pickle.load(open('sbi_theta_x_no_dist.pkl', 'rb'))
#theta = torch.tensor(theta.astype('float32'))
#x = torch.tensor(x.astype('float32'))
#x = x[:,(0,1,2,3,9,10,11,12)]
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
pickle.dump((posterior, (theta, x)), open('sbi_posterior_no_dist.pkl', 'wb'))
#sys.exit()"""
#(posterior, (theta, x)) = pickle.load(open(saveDir+'/sbi_posterior.pkl', 'rb'))"""
(posterior, (theta, x)) = pickle.load(open('sbi_posterior_no_dist.pkl', 'rb'))
# Now let's do a little injection and recovery test.
def simulate_singles(num_simulations):
	m1_lim = [0.4,2.5]
	feh_lim = [-2,0.4]
	age_lim = [0.2,10]
	num_simulated, theta, x = (0, [], [])
	while num_simulated < num_simulations:
		nTry = num_simulations-num_simulated
		m1s = (m1_lim[1]-m1_lim[0])*np.random.random(nTry)+m1_lim[0]
		qs = np.zeros(nTry)
		fehs = (feh_lim[1]-feh_lim[0])*np.random.random(nTry)+feh_lim[0]
		ages = (age_lim[1]-age_lim[0])*np.random.random(nTry)+age_lim[0]
		dists = 3000*np.random.beta(1.75,5,nTry)
		N = num_simulations - num_simulated
		print(f"Running {N} simulations")
		os.system('rm -rf current_*')
		textFile = open('current_'+str(N),'w')
		_theta = np.array([m1s,qs,ages,fehs,dists])
		_x = simulate_data_sets(_theta)
		errs = get_errors(_x[:,1:9])
		x_arr = np.array(_x)
		x_arr[:,9:18] = errs
		x_means = x_arr[:,0:9]
		for ii in range(9):
			x_arr[:,ii] = np.random.normal(x_means[:,ii],x_arr[:,ii+9])
		zeros = np.where(errs==0)[0]
		zeros = list(set(zeros))
		keep = np.all(np.isfinite(_x).numpy(), axis=1)
		keep[zeros] = False
		theta.extend(np.transpose(np.array(_theta[:,keep])))
		x.extend(x_arr[keep])
		num_simulated+=sum(keep)
	theta = torch.tensor(np.vstack(np.float32(theta)))
	x = torch.tensor(np.vstack(np.float32(x)))
	return (theta, x)
	
def simulate_binaries(num_simulations):
	m1_lim = [0.4,2.5]
	feh_lim = [-2,0.4]
	age_lim = [0.2,10]
	q_lim = [0.1,1]
	num_simulated, theta, x = (0, [], [])
	while num_simulated < num_simulations:
		nTry = num_simulations-num_simulated
		m1s = (m1_lim[1]-m1_lim[0])*np.random.random(nTry)+m1_lim[0]
		qs = (q_lim[1]-q_lim[0])*np.random.random(nTry)+q_lim[0]
		fehs = (feh_lim[1]-feh_lim[0])*np.random.random(nTry)+feh_lim[0]
		ages = (age_lim[1]-age_lim[0])*np.random.random(nTry)+age_lim[0]
		N = num_simulations - num_simulated
		print(f"Running {N} simulations")
		os.system('rm -rf current_*')
		textFile = open('current_'+str(N),'w')
		_theta = np.array([m1s,qs,ages,fehs])
		_x = simulate_data_sets(_theta)
		errs = get_errors(_x[:,0:8])
		x_arr = np.array(_x)
		x_arr[:,8:16] = errs
		x_means = x_arr[:,0:8]
		for ii in range(8):
			x_arr[:,ii] = np.random.normal(x_means[:,ii],x_arr[:,ii+8])
		zeros = np.where(errs==0)[0]
		zeros = list(set(zeros))
		keep = np.all(np.isfinite(_x).numpy(), axis=1)
		keep[zeros] = False
		theta.extend(np.transpose(np.array(_theta[:,keep])))
		x.extend(x_arr[keep])
		num_simulated+=sum(keep)
	theta = torch.tensor(np.vstack(np.float32(theta)))
	x = torch.tensor(np.vstack(np.float32(x)))
	return (theta, x)
nSim = 100
#sys.exit()
#num_injections = 100
#true_theta, true_obs = simulate_for_sbi_strict(sbi_simulator, sbi_prior, num_injections)
"""true_theta, true_obs = simulate_binaries(nSim)
pickle.dump((true_theta,true_obs),open('binary_data_no_dist.pkl','wb'))
sys.exit()"""
#(true_theta,true_obs) = pickle.load(open('binary_data.pkl','rb'))
#x = true_obs#[:,(0,1,2,3,9,10,11,12)]
(true_theta,true_obs) = pickle.load(open('binary_data_no_dist.pkl','rb'))
x = true_obs
#x[:,9:18] = 0
ii = 10
print(x[ii])
sample = posterior.sample(
		(num_samples,),
		x=x[ii],
		show_progress_bars=False
	)
pickle.dump((np.array(true_theta[ii]),np.array(x[ii]),np.array(sample)), open('sbi_ex_test_no_dist.pkl', 'wb'))
sys.exit()

#print(true_theta.shape,true_obs.shape)
#all_samples = []
for i, obs in enumerate(true_obs):
	section = i//(int(num_injections/nSplit))
	sample = posterior.sample(
		(num_samples,),
		x=obs,
		show_progress_bars=False
	)
	if sample.shape[0]<num_samples:
		sample = np.zeros((num_samples,true_theta.shape[1]))
	os.system('rm -rf current_*')
	textFile = open('current_'+str(i),'w')
	if i==int(section*num_injections/nSplit):
		all_samples = np.empty((int(num_injections/nSplit), num_samples, L))
	all_samples[i-section*int(num_injections/nSplit)] = sample
	if (i+1)%(num_injections/nSplit)==0:
		start = i+1-int(num_injections/nSplit)
		end = i+1
		pickle.dump((np.array(true_theta[start:end]),np.array(true_obs[start:end]),np.array(all_samples)), open(saveDir+'/sbi_values_binary'+str(int(section))+'.pkl', 'wb'))
