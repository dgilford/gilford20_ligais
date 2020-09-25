"""
Created on Thurs. Feb. 7 2019.

This module code contains functions that we use throughout the LIG codebase, by
first appending the directory that contains this set of utilities, and then
calling the particular function we want to use.

original author: @dgilford
last updated: 5/8/19 by @dgilford
"""

## FUNCTION TO NORMALIZE AXES:
# ------------------------------------------------------------------

def normalize(x):
    x_out=(x-x.min())/(x.max()-x.min())
    return x_out

## FUNCTION TO INVERT AXES NORMALIZATION:
# ------------------------------------------------------------------

# to go backward on the axes normaliztion, provide both the array to invert and the original max and min values
def denormalize(x,xorig_max,xorig_min):
    x_out=x*(xorig_max-xorig_min)+xorig_min
    return x_out

## FUNCTION TO DRAW A STRAIGHT LINE:
# ------------------------------------------------------------------

def draw_straight_line(xyvec,val,mode,linestyle_in='k--'):
    
    # function notes:
    # xyvec: the x- or y-coordinate the horizontal line should be drawn across
    #           ... the line will exist over the length of this specified vector
    # val: the value of the line (with the opposite coordinate chosen in xyvec)
    # mode: the interpreter for whether xyvec is 'x' or 'y'
    # linestyle_in: the linestyle the user wants for the straight line
    #
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    if mode=='x':
        plt.plot(xyvec, np.array([val for i in range(len(xyvec))]), linestyle_in) 
    elif mode=='y':
        plt.plot(np.array([val for i in range(len(xyvec))]), xyvec, linestyle_in) 
    else:
        print('error in the chosen mode, straight line will not be drawn.')
        
## FUNCTION TO PLOT THE HISTOGRAM OF HOW WELL A GP MODEL FITS THE TRAINING DATA:
# ------------------------------------------------------------------
def train_data_fit(m_in,X_train,data_train):
    
    # m_in: the model object defined with GPflow (often optimized, but not necessarily needed)
    # X_train: the grid the data (target function training data) lies on (e.g. parameter space, space, time, ...)
    # data_train: the array of the target function training data
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # query at the exact locations of the station to see statistics
    train_gpr_mean,train_gpr_var=np.squeeze(m_in.predict_y(X_train))

    # plot the histogram of the differences
    plt.figure()
    plt.hist((np.squeeze(data_train)-train_gpr_mean))
    plt.title('Training Data minus GP model')
    plt.xlabel('Difference')
    plt.ylabel('Counts')
    plt.show()
    
    # the model fit range is 2 times the standard deviation of the difference array
    model_fit_range=2*np.std((np.squeeze(data_train)-train_gpr_mean))
    
    return model_fit_range

## FUNCTION TO LOAD PERCEPTUALLY UNIFORM COLORMAPS:
# ------------------------------------------------------------------
def call_scm_cmap(cmap_name='lapaz',software_path='./data/colormaps/'):
    
    # Perceptually Uniform Colormaps are from:
    # Scientific Colour-Maps 4
    # 
    # http://www.fabiocrameri.ch/colourmaps.php
    # Version 4.0.1 (21.08.2018)
    # 
    # Citations: 
    # Crameri, F. (2018). Scientific colour-maps. Zenodo. http://doi.org/10.5281/zenodo.1243862
    # Crameri, F. (2018), Geodynamic diagnostics, scientific visualisation and StagLab 3.0, Geosci. Model Dev., 11, 2541-2562, doi:10.5194/gmd-11-2541-2018
    
    # import what we need:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # load the colormap from its data file
    cm_data = np.loadtxt(software_path+cmap_name+".txt")
    
    # define the colormap
    cmap_object = LinearSegmentedColormap.from_list(cmap_name, cm_data)
    
    # go back to the above program level
    return(cmap_object)
    
## FUNCTION TO LOAD EVENLY SPACE COLORS FROM A COLORMAP:
# ------------------------------------------------------------------
def uniform_cmap_slices(cmap_object,ncols,col_range=[0,1.0]):
    
    # import what we need:
    import numpy as np
    import matplotlib.pyplot as plt
    
    # slice along the colormap to get our uniformly spaced colors
    cols = cmap_object(np.linspace(col_range[0],col_range[1],ncols))
    
    # go back to the above program level
    return(cols)

## FUNCTION TO ARCHIVE A FIT EMULATOR:
# ------------------------------------------------------------------

def archive_gpflow_model(m_in,kernel_code,tf_session,save_dir='./archived_models/'):
    
    # 
    # This function takes in a GPflow model object that has already been fit
    # using the optimization tools of GPflow and conditioned on data, and then
    # archives the model's properties (freezing them in time) to refer to later
    # if needed by the user.
    
    # Inputs:
    #         save_dir    --> (string) The directory to save the model into 
    #         m_in        --> (tensflow object) The fit GPflow model
    #         kernel_code --> (string) the code used to create the kernel the model was fit on
    #         tf_session  --> (tensor object) required to save the archived model
    #
    
    # import what we need:
    import pickle
    import pandas
    import gpflow
    from datetime import datetime
    import tensorflow as tf
    
    # compute and store the model's log-likelihood, given that the model has been fit on data
    log_likelihood=m_in.compute_log_likelihood()
    
    # store the model parameters
    model_hyperparams=m_in.read_values()
    
    # find and store the training data of the model (dependant and independant variables)
    X_train=m_in.X.value
    Y_train=m_in.Y.value
    
    # find and store the model name
    model_name=m_in.name

    # determine the timestamp for the save name
    time_string=str(datetime.utcnow())[:-7].replace(" ", "_")
    save_name='m_metadata_'+time_string+'.pk1'
    
        
    # build the dictionary to archive the model
    archive_dat={'X_train': X_train, 'Y_train': Y_train, 'model_name': model_name, 'kernel_code': kernel_code, \
                'time_of_creation': time_string, 'log_likelihood': log_likelihood, 'model_hyperparams': model_hyperparams}
    
    # archive the model
    pickle.dump(archive_dat, open( save_dir+save_name, "wb" ) )
    print("Model metadata saved in path: %s" % save_dir+save_name)
    
    # create the saver object and archive the model object
    saver = tf.train.Saver()
    m_save_path = saver.save(tf_session, save_dir+model_name+"_"+time_string+".ckpt")
    print("Model saved in path: %s" % m_save_path)
    
    return

## FUNCTION TO ESTIMATE THE POSTERIOR DENSITIES AND LIKELIHOOD OF EACH SAMPLE, GIVEN THE PRIOR
## DISTRIBUTION AND THE LIKELIHOOD DISTRIBUTION:
# ------------------------------------------------------------------

def posterior_densities(steps,prior_dist,likelihood_dist=0):
    
    #
    # This code calculates the posterior densities found by integrating over the densities of the GP
    # model prior distributions and a likelihood distribution. The finer the input steps mesh, the more exact the 
    # integration will be.
    #
    # If the likelihood distribution is not passed to the function (or is passed as "0") then the
    # density estimate over the steps mesh for the prior will be returned (the likelihood is treated
    # as ~inifinitely uniform with equal probabilities at every step).
    # 
    
    # Inputs:
    #     steps            --> (1d vector) The bins/locations over which to evaluate the densities.
    #                                      The step size determines how smooth the densities evaluated are
    #                                      (with finer being more exact/less of an estimate)
    #     prior_dist       --> (scipy object) The distribution of the prior probabilities
    #     likelihood_dist  --> (scipy object) The distribution of the likelihood to constrain the prior
    #
    # Outputs:
    #
    #     The code outputs a dictionary ("store_dat") containing:
    #
    #     posterior        --> (1d vector) The posterior distribution, with size=len(samps)
    #     param_density    --> (1d vector) The likelihood of each parameter value (as given by the values
    #                                      /length used to construct the scipy object "prior_dist") for the 
    #                                      samples used to construct the prior_dist scipy object.
    # 
    
    from scipy.stats import uniform, norm
    import numpy as np
    
    # derive what we need for the function
    nsamps=prior_dist.pdf(1).size
    nsteps=len(steps)
    step_size=steps[1]-steps[0]
    
    #  if likelihood_dist=0, return the prior by passing through a long uniform window
    if likelihood_dist==0:
        likelihood_dist=uniform(loc=-1e6, scale=2e6)
    
    # create arrays needed in our calculations
    step_prob=np.empty((nsteps,1),dtype='float')
    samplestep_prob=np.empty((nsteps,nsamps),dtype='float')
    
    # loop over each bin in "steps" to find probabilities in each bin
    for k in range(nsteps):
        
        # find the probability of each step at each (input) parameter value from the prior
        samplestep_prob[k,:]=np.squeeze(prior_dist.pdf(steps[k]))*np.squeeze(likelihood_dist.pdf(steps[k]))
    
        # find the integrated step probability over all prior parameter values
        step_prob[k,0]=np.sum(samplestep_prob[k,:])
        
    # find the likelihoods of each sample
    param_density=np.sum(samplestep_prob,axis=0)/np.sum(np.sum(samplestep_prob,axis=0))
        
    # normalize the probabilities
    posterior_pdf=step_prob/np.sum(step_prob*step_size)
    
    # go back to the above program level
    store_dat={'posterior':posterior_pdf,'param_density':param_density}
    return(store_dat)


## FUNCTION TO ESTIMATE THE POSTERIOR DENSITIES OF THE RCP8.5 EMULATOR OVER TIME, GIVEN THE LIKELIHOODS
## OF EACH SAMPLE IN MODEL PARAMETER SPACE:
# ------------------------------------------------------------------
def posterior_densities_rcp85(steps,time_grid,prior_dist,likelihoods):
    
    #
    # This code calculates the posterior densities of the RCP8.5 emulator, found by integrating 
    # over the densities of the GP model prior distributions at each sample point, weighted by the likelihoods
    # of each sample in the emulator parameter space. The finer the input steps mesh, the more exact the 
    # integration will be.
    # 
    
    # Inputs:
    #     steps            --> (1d vector) The bins/locations over which to evaluate the densities.
    #                                      The step size determines how smooth the densities evaluated are
    #                                      (with finer being more exact/less of an estimate)
    #     time_grid        --> (1d vector) The temporal grid on which the prior timeseries of the emulator is defined
    #     prior_dist       --> (scipy object) The distribution of the prior probabilities (ntime x nsamps)
    #     likelihoods       --> (1d vector) The likelihood probabilities of each parameter set (constrained 
    #                                      by something, usually the LIG) associated with samples in the prior
    #
    # Outputs:
    #
    #     The code outputs a dictionary containing:
    #
    #     prob_density_ts  --> (1d vector) The posterior distribution timeseries, with size=len(samps)
    # 
    
    from scipy.stats import uniform, norm
    import numpy as np
    
    # derive what we need for the function
    nsamps=prior_dist.pdf(1).shape[1]
    ntime=len(time_grid)
    nsteps=len(steps)
    step_size=steps[1]-steps[0]
    
    # create arrays needed in our calculations
    samplestep_prob=np.empty((nsteps,ntime,nsamps),dtype='float')
    cumulative_density_ts=np.empty((nsteps,ntime),dtype='float')

    # loop over each bin in "steps" to find probabilities in each bin
    for k in range(nsteps):

        # calculate the weighted densities in each bin and each time
        # where the time varying PDF is the prior and the parameter probabilities for each sample are the likelihood
        samplestep_prob[k,:,:]=np.squeeze(prior_dist.pdf(steps[k]))*np.squeeze(likelihoods)

    # Integrate over all samples to find the (un-normalized) probabilities at each step over time
    prob_time=np.sum(samplestep_prob,axis=2)

    # Normalize the probabilities
    prob_density_ts=prob_time[:,:]/np.sum(prob_time[:,:],axis=0)
    
    # derive the cumulative density over the steps at each point in time
    cumulative_density_ts=np.cumsum(prob_density_ts,axis=0)
    
    # go back to the above program level
    store_dat={'posterior_ts':prob_density_ts,'cumulative_posterior_ts':cumulative_density_ts}
    return(store_dat)


## FUNCTION TO SMOOTH NUMPY TIMESERIES
## FROM THE SCIPY-COOKBOOK: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
# ------------------------------------------------------------------

def cookbook_smoother(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    
    import numpy

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


## FUNCTION that finds the nearest quantile index/value called by the user with an array of quantiles:
# ------------------------------------------------------------------
def find_quantile_i(F,q):
    
    # F is the cumulative distribution function
    # q is the quantile of interest (in integer, 0-->100)
    
    import numpy as np
    
    # create the array we need
    found_i=np.empty((len(q),),dtype='int')
    found_val=np.empty((len(q),),dtype='float64')
    
    # loop through all quantiles requested
    for qi in range(len(q)):
    
        # find the quantile location and value
        found_i[qi]=np.floor(np.where(np.min(np.abs(F-q[qi]/100))==np.abs(F-q[qi]/100)))
        found_val[qi]=F[found_i[qi]]
    
    # go back to the above program level
    store_dat={'q_i':found_i,'q_vals':found_val}
    return(store_dat)


## FUNCTION that uses the nearest quantile index/value to find the values associated with the quantile from the
# steps used to generate "F"... output over time
# ------------------------------------------------------------------
def find_quantile_ts(F,q,steps,time_grid):
    
    # F is the cumulative distribution function
    # q is the quantile of interest (in integer, 0-->100)
    # steps are the steps along which to find the cumulative distribution
    # time_grid is the grid on which the native timeseries of F lies
    #
    # NOTE: This function requires the module "find_quantile_i" to run
    
    import numpy as np
    
    # create the necessary output array
    quants_output=np.empty((len(q),len(time_grid)),dtype='float64')
    
    # loop to find the quantile values over time
    for t in range(len(time_grid)):
        # call the module to find the index
        ii=find_quantile_i(F[:,t],q)
        # use the steps to find the value in F
        quants_output[:,t]=np.squeeze(steps[ii['q_i']])
    
    # go back to the above program level
    return(quants_output)



