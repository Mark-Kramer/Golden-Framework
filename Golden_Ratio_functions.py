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
    osc_to_perturb = 4;
    
    # Gain (fix constant gain g_C = 50)
    f_S    = gain["f_S"]
    g_S    = gain["g_S"]
    g_C    = 50
    t_gain = np.arange(0,3,0.002)
    f_gain = g_C*np.ones(t_gain.shape) + g_S*np.cos(2*np.pi*f_S*t_gain)

    # Initialize (1 s, gain=0)
    Tmax  = 0.5;  x0 = np.zeros(16)
    pre   = solve_ivp(damped_coupled_harmonic_oscillators, [0,Tmax], x0, args=(omega,beta,t_gain,0*f_gain), max_step=0.01)
    
    # Perturb oscillator (4 s, gain > 0)
    Tmax  = 2.5;  x0 = np.zeros(16); x0[2*network["k_perturbed"]]=1
    post  = solve_ivp(damped_coupled_harmonic_oscillators, [0,Tmax], x0, args=(omega,beta,t_gain,f_gain), max_step=0.01)
    
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
    for k in np.arange(0,16,2):
        #p=plt.subplot(8,1,counter)
        if k == network["k_perturbed"]*2:
            plt.plot(t,counter+x[k],'r',linewidth=1)
        else:
            plt.plot(t,counter+x[k]*10,'k',linewidth=1)
        counter=counter+1
    plt.yticks(np.arange(0,counter), ["%.1f" % number for number in f_k])
    plt.xticks([0,1,2])
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    ax = plt.gca()
    ax.xaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel('Time [s]')
    plt.ylabel('Node frequency [Hz]')
        
# Plot the gain results. #################################################################
def plot_gain_traces(res):
    f_k    = res["network"][0]["f_k"][0][0]
    f_S    = res["f_S"][0]
    A      = res["A"]
    driver = res["network"][0]["k_perturbed"][0][0][0]
    
    #yspacing = np.arange(0,8)*3
    #yspacing[driver]=yspacing[driver]-2
    counter=0
    y0 = np.zeros(8)
    for k in np.arange(0,8):
        A0 = np.log10(A[2*k,:])  # Get trace.
        A0 = A0-np.min(A0)       # Set min to 0
        #p=plt.subplot(8,1,counter)
        #plt.tick_params(
        #    axis='x',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    bottom=False,      # ticks along the bottom edge are off
        #    top=False,         # ticks along the top edge are off
        #    labelbottom=True) # labels along the bottom edge are off
        if k == driver:
            plt.semilogx(f_S,1.5*k+A0,'r',linewidth=1)
        else:
            plt.semilogx(f_S,1.5*k+A0,'k',linewidth=1)
        y0[k] = 1.5*k+A0[0]
        #ax = plt.gca()
        #ylim = ax.get_ylim()
        #plt.ylim([-0.5, 2.5])
        #plt.text(f_k[0]+0.1,1,['f = '+"%.1f" % f_k[counter-1]][0])
        #niceaxis(p)
        #plt.xticks(f_k, f_k, rotation='vertical')
        #ax = plt.gca()
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #plt.xlim([f_k[0]-0.1, f_S[-1]])
        
        counter=counter+1
    plt.plot([1,1], [1,2], 'k', 'LineWidth', 4)
    plt.yticks(y0, ["%.1f" % number for number in f_k])
    plt.xticks(f_k, ["%.1f" % number for number in f_k])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)         # ticks along the top edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    plt.xlim([1,np.max(f_S)])
    ax = plt.gca()
    ax.xaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Gain frequency [Hz]')
    plt.ylabel('Node frequency [Hz]')
        
# Plot the gain results. #################################################################
def plot_gain_traces_two_groups(res, f_k_ticks):
    f_k    = res["network"][0]["f_k"][0][0]
    f_S    = res["f_S"][0]
    A      = res["A"]
    driver = res["network"][0]["k_perturbed"][0][0][0]
    
    y0 = np.zeros(8)
    for k in np.arange(0,8):
        A0 = np.log10(A[2*k,:])  # Get trace.
        A0 = A0-np.min(A0)       # Set min to 0
        if k == driver:
            plt.semilogx(f_S,1.5*k+A0,'r',linewidth=1)
        else:
            plt.semilogx(f_S,1.5*k+A0,'k',linewidth=1)
        y0[k] = 1.5*k+A0[0]
        #plt.tick_params(
        #    axis='x',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    bottom=False,      # ticks along the bottom edge are off
        #    top=False,         # ticks along the top edge are off
        #    labelbottom=True) # labels along the bottom edge are off
        #if k == res["network"]["k_perturbed"]*2:
        #    plt.plot(f_S,np.log10(A[k]),'r',linewidth=1)
        #else:
        #    plt.plot(f_S,np.log10(A[k]),linewidth=1)
        #ax = plt.gca()
        #ylim = ax.get_ylim()
        #plt.ylim([-0.5, 2.5])
        #plt.text(f_k[0]+0.1,1,['f = '+"%.1f" % f_k[counter-1]][0])
        #niceaxis(p)
        #ax = plt.gca()
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #plt.xlim([f_k[0]-1, f_S[-1]])
        #counter=counter+1
    plt.yticks(y0,        ["%.1f" % number for number in f_k])
    plt.xticks(f_k_ticks, ["%.1f" % number for number in f_k_ticks])
    plt.xlim([np.min(f_S), np.max(f_S)])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)         # ticks along the top edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False)         # ticks along the top edge are off
    ax = plt.gca()
    ax.xaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Gain frequency [Hz]')
    plt.ylabel('Node frequency [Hz]')
