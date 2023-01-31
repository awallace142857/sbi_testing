import numpy as np
import os,sys,pickle
import matplotlib.pyplot as plt
import corner
from isochrones import get_ichrone
from astropy.table import Table
from chainconsumer import ChainConsumer
tracks = get_ichrone('mist', tracks=True)
def model_mags(samples):
	properties = tracks.generate_binary(
        samples[:,0],
        0,
        np.log10(samples[:,2]) + 9,
        samples[:,3],
    	bands=["G", "BP", "RP", "J", "H", "K", "W1", "W2"]
    )
	all_mags = np.array([properties.BP_mag.values,properties.G_mag.values,properties.RP_mag.values,properties.J_mag.values,properties.H_mag.values,properties.K_mag.values,properties.W1_mag.values,properties.W2_mag.values])
	all_mags+=5*np.log10(samples[:,4])-5
	return all_mags

def model_mags_bin(samples):
	properties = tracks.generate_binary(
        samples[:,0],
        samples[:,0]*samples[:,1],
        np.log10(samples[:,2]) + 9,
        samples[:,3],
    	bands=["G", "BP", "RP", "J", "H", "K", "W1", "W2"]
    )
	all_mags = np.array([properties.BP_mag.values,properties.G_mag.values,properties.RP_mag.values,properties.J_mag.values,properties.H_mag.values,properties.K_mag.values,properties.W1_mag.values,properties.W2_mag.values])
	all_mags+=5*np.log10(samples[:,4])-5
	return all_mags

(theta,x,sample) = pickle.load(open('sbi_ex_test_with_errors.pkl','rb'))
table_data = np.zeros((sample.shape[0],24))
mags_s = model_mags(sample)
mags_b = model_mags_bin(sample)
print(x[1]-x[3],x[2]-5*np.log10(100./x[0]))
if len(x)==9:
	x = np.concatenate((x,np.zeros(9)))
for jj in range(3):
	print(np.std(mags_b[jj]))
for jj in range(len(sample)):
	table_data[jj] = [sample[jj][0],sample[jj][1],sample[jj][2],sample[jj][3],sample[jj][4],mags_s[0][jj],mags_b[0][jj],mags_s[1][jj],mags_b[1][jj],mags_s[2][jj],mags_b[2][jj],x[0],x[9],x[1],x[10],x[2],x[11],x[3],x[12],theta[0],theta[1],theta[2],theta[3],theta[4]]

table_names = ['m1_sbi','q_sbi','age_sbi','feh_sbi','dist_sbi','b_single','b_binary','g_single','g_binary','r_single','r_binary','par_obs','par_err','b_obs','b_err','g_obs','g_err','r_obs','r_err','m1_true','q_true','age_true','feh_true','dist_true']
source_table = Table(data=table_data,names=table_names)
source_table.write('sample_ex_with_errors.csv',format='csv',overwrite=True)
basename = "sample_ex_with_errors.csv"

table = Table.read(basename)

sbi_labels = ["m1", "q", "age", "feh", "dist"]
bands = ["b", "g", "r"]# "j", "h", "k", "w1", "w2"]

binary_theta = np.array([table[f"{pn}_sbi"] for pn in sbi_labels]).T
single_theta = np.copy(binary_theta)
# ChainConsumer doesn't like all values being EXACTLY the same
single_theta[:, 1] = np.random.normal(1e-12, 1e-16, size=len(single_theta))

binary_photometry = np.array([table[f"{band}_binary"] for band in bands]).T
single_photometry = np.array([table[f"{band}_single"] for band in bands]).T

true_theta = np.array([table[f"{pn}_true"] for pn in sbi_labels]).T
true_theta[:, 0] += np.random.normal(0, 0.01, size=len(true_theta)) # m1
true_theta[:, 1] += np.random.normal(0, 0.01, size=len(true_theta)) # q 
true_theta[:, 2] += np.random.normal(0, 0.01, size=len(true_theta)) # age
true_theta[:, 3] += np.random.normal(0, 0.01, size=len(true_theta)) # feh
true_theta[:, 4] = 1000.0/np.random.normal(table["par_obs"][0], table["par_err"][0], size=len(true_theta)) # dist

obs_photometry = np.random.normal(
    [table[f"{band}_obs"][0] for band in bands],
    [table[f"{band}_err"][0] for band in bands],
    size=(len(true_theta), len(bands))
)
print([table[f"{band}_obs"][0] for band in bands],[table[f"{band}_err"][0] for band in bands])
true_chain = np.hstack([true_theta, obs_photometry])
binary_chain = np.hstack([binary_theta, binary_photometry])
single_chain = np.hstack([single_theta, single_photometry])
parameters = sbi_labels + bands

q = table["q_sbi"]

c = ChainConsumer()
#binary_chain+=np.random.normal(0,0.01,size=binary_chain.shape)
c.add_chain(binary_chain, parameters=parameters, name=f"$q = {np.median(q):.1f}\pm {np.std(q):.1f}$")
#c.add_chain(single_chain, parameters=parameters, name=r"$q=0$")
true_chain+=np.random.normal(0,0.01,size=true_chain.shape)
c.add_chain(true_chain, parameters=parameters, name="truth + data")
c.configure(max_ticks=2, diagonal_tick_labels=False, tick_font_size=8)
c.plotter.plot(filename=f"{basename}.png", figsize="column")
plt.show()
table = Table.read('sample_ex_with_errors.csv')
theta_with_errors = np.array([table[f"{pn}_sbi"] for pn in sbi_labels]).T
photometry_with_errors = np.array([table[f"{band}_binary"] for band in bands]).T
chain_with_errors = np.hstack([theta_with_errors, photometry_with_errors])
table = Table.read('sample_ex_no_errors.csv')
theta_no_errors = np.array([table[f"{pn}_sbi"] for pn in sbi_labels]).T
photometry_no_errors = np.array([table[f"{band}_binary"] for band in bands]).T
chain_no_errors = np.hstack([theta_no_errors, photometry_no_errors])

parameters = sbi_labels + bands

c = ChainConsumer()
#binary_chain+=np.random.normal(0,0.01,size=binary_chain.shape)
c.add_chain(chain_no_errors, parameters=parameters, name=f"No Magnitude Error")
c.add_chain(chain_with_errors, parameters=parameters, name=f"Magnitude Error = 0.01")
#c.add_chain(single_chain, parameters=parameters, name=r"$q=0$")
c.configure(max_ticks=2, diagonal_tick_labels=False, tick_font_size=8)
c.plotter.plot(filename=f"compare.png", figsize="column")
plt.show()
sys.exit()