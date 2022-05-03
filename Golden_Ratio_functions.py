# Load necessary packages
import numpy as np
from scipy.integrate   import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal      import hilbert
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Define the 8 damped, coupled harmonic oscillators #######################################
def damped_coupled_harmonic_oscillators(t,x,omega,beta,ft,f):
    f = interp1d(ft,f)(t)
    dx = [x[1], #-----------------------------------------------------------Oscillator 1
          -omega[0]**2*x[0] - 2*beta*x[1] + np.sum(x[[2,4,6,8,10,12,14]])*f,
          x[3], #-----------------------------------------------------------Oscillator 2
          -omega[1]**2*x[2] - 2*beta*x[3] + np.sum(x[[0,4,6,8,10,12,14]])*f,
          x[5], #-----------------------------------------------------------Oscillator 3
          -omega[2]**2*x[4] - 2*beta*x[5] + np.sum(x[[0,2,6,8,10,12,14]])*f,
          x[7], #-----------------------------------------------------------Oscillator 4
          -omega[3]**2*x[6] - 2*beta*x[7] + np.sum(x[[0,2,4,8,10,12,14]])*f,
          x[9], #-----------------------------------------------------------Oscillator 5
          -omega[4]**2*x[8] - 2*beta*x[9] + np.sum(x[[0,2,4,6,10,12,14]])*f,
          x[11],#-----------------------------------------------------------Oscillator 6
          -omega[5]**2*x[10]- 2*beta*x[11]+ np.sum(x[[0,2,4,6,8, 12,14]])*f,
          x[13],#-----------------------------------------------------------Oscillator 7
          -omega[6]**2*x[12]- 2*beta*x[13]+ np.sum(x[[0,2,4,6,8, 10,14]])*f,
          x[15],#-----------------------------------------------------------Oscillator 8
          -omega[7]**2*x[14]- 2*beta*x[15]+ np.sum(x[[0,2,4,6,8, 10,12]])*f]
    return dx

# Function to run the 8 damped, coupled harmonic oscillators ##############################
def run_damped_coupled_harmonic_oscillators(network, gain):
    
    # Network (fix perturbed oscillator as 4th, fix damping beta = 2)
    omega  = 2*np.pi*np.asarray(network["f_k"])
    beta   = 2
    
    # Gain (fix constant gain g_C = 50)
    f_S    = gain["f_S"]
    g_S    = gain["g_S"]
    g_C    = 50
    t_gain = np.arange(0,3,0.002)
    f_gain = g_C*np.ones(t_gain.shape) + g_S*np.cos(2*np.pi*f_S*t_gain)

    # Initialize (0.5 s, gain=0) to allow oscillators to reach equilibrium.
    Tmax  = 0.5; x0 = np.zeros(16)
    pre   = solve_ivp(damped_coupled_harmonic_oscillators, [0,Tmax], x0, args=(omega,beta,t_gain,0*f_gain), max_step=0.01)
    
    # Perturb oscillator (2.5 s, gain > 0)
    Tmax  = 2.5; x0 = np.zeros(16); x0[2*network["k_perturbed"]]=1
    post  = solve_ivp(damped_coupled_harmonic_oscillators, [0,Tmax], x0, args=(omega,beta,t_gain,  f_gain), max_step=0.01)
    
    # Concatenate results, perturbation at t=0.
    t = np.concatenate( (pre.t-pre.t[-1],post.t) )
    x = np.concatenate( (pre.y,post.y), axis=1 )
    
    return x,t

# Plot the model results. #################################################################
def plot_model_traces(t,x,network,gain):
    f_S    = gain["f_S"]
    g_S    = gain["g_S"]
    f_k    = network["f_k"]
    
    counter=0
    for k in np.arange(0,16,2):                         # Plot the "position" x for each oscillator
        if k == network["k_perturbed"]*2:
            plt.plot(t,counter+x[k],'r',linewidth=1)    # ... make the perturbed oscillator RED.
        else:
            plt.plot(t,counter+x[k]*10,'k',linewidth=1) # ... otherwise, make it BLACK.
        counter=counter+1                               # y-axis indicates freq of each oscillator.
    plt.yticks(np.arange(0,counter), ["%.1f" % number for number in f_k])
    plt.tick_params(axis='y',which='major',left=False,right=False) # ... and remove y-axis ticks.
    plt.xticks([0,1,2])                                 # x-axis indicates times at 0,1,2, s.
    plt.gca().xaxis.grid(True)                          # Add vertical lines for x axis.
    plt.gca().spines['top'].set_visible(False)          # Prettify plot.
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Time [s]')
    plt.ylabel('Oscillator frequency [Hz]')
        
# Plot the gain results. #################################################################
def plot_gain_traces(res):
    f_k    = res["network"][0]["f_k"][0][0]
    f_S    = res["f_S"][0]
    A      = res["A"]
    driver = res["network"][0]["k_perturbed"][0][0][0]
    
    counter=0
    y0 = np.zeros(8)
    for k in np.arange(0,8):     # For each oscillator,
        A0 = np.log10(A[2*k,:])  # ... get log10(amplitude),
        A0 = A0-np.min(A0)       # ... set min to 0.
        if k == driver:          # If it's the perturbed node, plot it RED,
            plt.semilogx(f_S,1.5*k+A0,'r',linewidth=1)
        else:                    # ... otherwise, plot it BLACK.
            plt.semilogx(f_S,1.5*k+A0,'k',linewidth=1)
        y0[k] = 1.5*k+A0[0]      # Offset in y to make plot look nice.
        counter=counter+1
    plt.plot([1,1], [1,2], 'k', 'LineWidth', 4)           # Include scale bar in y-direction (1).
    plt.yticks(y0, ["%.1f" %  number for number in f_k])  # Y-tick label = oscillator freq.
    plt.xticks(f_k, ["%.1f" % number for number in f_k])  # X-tick label = oscillator freq.
    plt.tick_params(axis='x',which='minor',bottom=False,top=False) # Clean up the tick marks.
    plt.tick_params(axis='y',which='major',left=False,right=False)
    plt.xlim([1,np.max(f_S)])
    plt.gca().xaxis.grid(True)                            # Add vertical lines for x axis.
    plt.gca().spines['top'].set_visible(False)            # Prettify plot.
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Gain frequency [Hz]')
    plt.ylabel('Oscillator frequency [Hz]')
        
# Plot the gain results. #################################################################
def plot_gain_traces_two_groups(res, f_k_ticks):
    f_k    = res["network"][0]["f_k"][0][0]
    f_S    = res["f_S"][0]
    A      = res["A"]
    driver = res["network"][0]["k_perturbed"][0][0][0]
    
    y0 = np.zeros(8)
    for k in np.arange(0,8):     # For each oscillator,
        A0 = np.log10(A[2*k,:])  # ... get log10(amplitude),
        A0 = A0-np.min(A0)       # ... set min to 0.
        if k == driver:          # If it's the perturbed node, plot it RED,
            plt.semilogx(f_S,1.5*k+A0,'r',linewidth=1)
        else:                    # ... otherwise, plot it BLACK.
            plt.semilogx(f_S,1.5*k+A0,'k',linewidth=1)
        y0[k] = 1.5*k+A0[0]      # Offset in y to make plot look nice.
    plt.plot([np.min(f_S),np.min(f_S)], [1,2], 'k', 'LineWidth', 4) # Include scale bar in y-direction (1).
    plt.yticks(y0,        ["%.1f" % number for number in f_k])      # Y-tick label = oscillator freq.
    plt.xticks(f_k_ticks, ["%.1f" % number for number in f_k_ticks])# X-tick label = user specified input.
    plt.tick_params(axis='x',which='minor',bottom=False,top=False)  # Clean up the tick marks.
    plt.tick_params(axis='y',which='major',left=False,right=False)
    plt.xlim([np.min(f_S), np.max(f_S)])
    plt.gca().xaxis.grid(True)                            # Add vertical lines for x axis.
    plt.gca().spines['top'].set_visible(False)            # Prettify plot.
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Gain frequency [Hz]')
    plt.ylabel('Oscillator frequency [Hz]')
