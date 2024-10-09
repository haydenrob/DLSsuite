"""
Created on 2024-05-17

@author: Hayden Robertson

These functions are used to model and fit DLS data
 - model g(r) to the Cumulant function
 - extract Gamma from the Cumulant function
 - Extract diffusion coeff from Gamma vs q**2 plot
 - Determine radius of hydration from Stokes-Einstein equation

"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import constants

rows_to_skip = 18       #Attention: counting starts at 1
row_count = 231         #Number of maximum lines to be read: -1 = all lines. was 232
# end_string = 'Count Rate History'
# end_string = '\n\n'

x_col = 0   #Lag time
y_col = 1   #g2-1

wavelength = 660 # nm
n = 1.333
viscosity = 1e-3 # Pa
temp_err = 0.1



def Cumulant(tau, B, gamma, mu, beta):
    """
    Cumulant function.
    """

    y = B + beta*np.exp(-2*gamma*tau)*(1+(mu/2)*tau**2)**2
    return y



def grab_metadata(fh, yaml = False):
    """
    Function which reads in the corresponding yaml file of the relevant data file. Returns the temperature and scattering angle.
    You can choose to use the yaml file or use the meta data embedded within each measurement file.
    """

    if yaml:
        with open(fh[:-7] + '_measurementSummary.yml', 'r') as file:
            md = yaml.safe_load(file)

        temp = md['Temperature set point (°C)']
        theta = md['Scattering angle (°)']

    else:
        with open(fh, "r") as file:
            data = file.read()

        data = data.split('\n')
        temp = float(data[7].split('\t')[1]) - 273.15
        temp = round(temp, 0)

        theta = float(data[2].split('\t')[1])

    file.close()

    return temp, theta



def data_loader(fh):
    """
    Function to load in DLS .dat files containing correlation function. Returns `x` and `y` arrays.
    """

    xy_values = np.loadtxt(fh,
                        dtype=float,
                        comments='#',
                        delimiter='\t',
                        skiprows=rows_to_skip,
                        max_rows=row_count,
                        usecols=None
                        )

    x = xy_values[:row_count:,x_col]
    y = xy_values[:row_count:,y_col]

    return x, y



def plot_data(x, y, params, temp, theta):
    """
    Function to plot the experimental g(r) with superimposed Cumulant fit.
    """

    fig, ax = plt.subplots()

    y_fitted = Cumulant(x, params[0], params[1], params[2], params[3])

    ax.plot(x, y, 'o', label = 'Experimental')
    ax.plot(x, y_fitted, '-', label = 'Model')
    ax.set_xlabel('Lag time')
    ax.set_ylabel('g(r)')
    ax.set_xscale('log')
    ax.text(0.1, 0.1, f'{temp}°C\n\n{theta}°', transform=ax.transAxes)
    ax.legend()
    fig.tight_layout()
        
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig.savefig('plots/gr_' + str(temp) + 'C_' + str(round(theta)) + 'degrees.pdf')

    plt.close(fig)



def q(theta, wavelength = wavelength, n = n):
    """
    Calculates `q` from theta.
    """
    
    q = (4 * np.pi * n * np.sin(np.radians(theta) / 2)) / (wavelength * 10**(-9))
    return q
    

def plot_gamma(unique_values, m, b, r_value, temp, i):
    """
    Function to plot gamma values as a function of q.
    """

    fig, ax = plt.subplots()

    qs =  unique_values[:,0]
    gamma = unique_values[:,1]
    yerr = unique_values[:,2]

    ax.errorbar(qs[:-i], gamma[:-i], yerr=yerr[:-i], marker='o', label = 'Experiment', linestyle='', color='C0')
    ax.plot(qs, m * qs + b, '-', label = 'Model')

    fig.tight_layout()

    ax.set_xlabel('$q^2$')
    ax.set_ylabel('Gamma')
    ax.text(0.1, 0.75, f'{temp}°C\n\nD = {m}\n\n$R^2$={r_value ** 2}', transform=ax.transAxes)
    ax.legend()

    ax.errorbar(qs[-i:], gamma[-i:], yerr=yerr[-i:], marker='o', label = 'Experiment', fillstyle='none', linestyle='', color='C0')

    fig.tight_layout()
    fig.savefig('plots/gamma_' + str(temp) + 'C_.pdf')
    plt.close(fig)


def stokes_einstein(temp, D, D_err, viscosity = viscosity, temp_err = temp_err):
    """
    Function to calculate the radius of hydration using the diffusion coefficient and the Stokes-Einstein equation.
    """

    temp += 273.15
    # viscosity = D / (constants.k * temp)

    r = (constants.k * temp) / (6 * np.pi * D * viscosity)
    r *= 10**9

    r_err = (constants.k * temp_err) / (6 * np.pi * D * viscosity) + (constants.k * temp * D_err) / (6 * np.pi * viscosity * D**2)
    r_err *= 10**9

    return r, r_err
