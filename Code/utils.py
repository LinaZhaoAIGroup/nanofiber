from scipy.interpolate import splev, splrep
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from IPython import display
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
import copy
import sys
sys.path.append('/home/linazhao/mhsvn/diffrac_ring/Code/')
from simsingleshot import simulation

import seaborn
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import h5py
import torch

from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ShuffleSplit
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader


def filter_peak(x,y,threshold=0.15):

    from scipy.signal import argrelextrema
    import scipy
    box_x = []
    box_y = []
    for i in range(len(x)):
        tmp = x[i].copy()
#         peak_index = argrelextrema(tmp, comparator=np.greater)[0]
        peak_index = scipy.signal.find_peaks(tmp, width=2)[0]
        mini_maximum = np.min(tmp[peak_index])
        if mini_maximum >= threshold * np.max(tmp) and len(peak_index) == 4:
            box_x.append(tmp)
            box_y.append(y[i].copy())
    return np.array(box_x), np.array(box_y)
def search_peak(chi, intensity):
    """
    find extrema chi and I(chi). """
    from scipy.signal import argrelextrema
    chi = np.array(chi)
    intensity = np.array(intensity)
    indices = argrelextrema(data = intensity, comparator=np.greater)[0]
    return chi[indices].copy(), intensity[indices].copy()

def interpolate_func(x, y, x2):
    """
    chi1, I(chi1), chi2 => I(chi2). 
    """
    from scipy.interpolate import splrep, splev
    spl = splrep(x, y)
    y2 = splev(x2, spl)
    return y2

def plot_parity(y_true, y_pred, df_score, file_path ='./parity_plot.jpg', show=False):
    ### parity plot
    labels = ['alpha_1', 'beta_1', 'gamma_1', 'dgamma_1', 'alpha_2', 'beta_2', 'gamma_2', 'dgamma_2','scale_ratio']
    labels_math = [r'$\alpha_1$', r'$\beta_1$', r'$\gamma_1$', r'$\Delta \gamma_1$', r'$\alpha_2$', r'$\beta_2$', r'$\gamma_2$', r'$\Delta \gamma_2$', r'$\lambda_1 / \lambda_2$']
    label_to_icon={'alpha_1':"o",
                  'beta_1':"v",
                  'gamma_1':'^',
                  'dgamma_1':"s",
                  'alpha_2':"P",
                   'beta_2': "d",
                   'gamma_2': "+",
                   'dgamma_2': "x",
                   'scale_ratio': "<"
                 }

    colors_by_label = {'alpha_1':'orangered',
                      'beta_1':'darkorange',
                      'gamma_1':'gold',
                      'dgamma_1':'seagreen',
                      'alpha_2':'dodgerblue',
                       'beta_2': 'blue',
                       'gamma_2': 'c',
                       'dgamma_2': 'm',
                       'scale_ratio': 'y'
                     }


    fig= plt.figure(figsize = (12, 10), dpi=1000)
    for i, label in enumerate(labels):
        all_min = 0
        all_max = 1
        ax = fig.add_subplot(3, 3, i+1)
        number = len(y_true)
        all_min = min(all_min, min(y_true[:, i]))
        all_max = max(all_max, max(y_true[:, i]))

        plt.scatter((y_true[:, i])[0],
                        (y_pred[:, i])[0], s=1.5,
                        zorder=0,
                        marker=label_to_icon[label],
                        color= colors_by_label[label], 
                        alpha = 1,
                        label = labels_math[i] + r" MAE: " + str(np.round((df_score.loc[label, 'mae']),2)) +  '\n    ' + 
                    r'  $R^2$:' + str(np.round((df_score.loc[label, 'r2']),2)))

        plt.scatter(y_true[:, i],
                    y_pred[:, i], s=1.5,

                    marker=label_to_icon[label],
                    color= colors_by_label[label], 
                    alpha = 0.055)
        plt.plot((all_min,all_max),(all_min,all_max),color='black',ls='--')
        ax = plt.gca()
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_yticks())
        delta = (all_max - all_min) * 0.1
        plt.axis('equal')
        ax.set_xlim(all_min - delta, all_max + delta)
        ax.set_ylim(all_min - delta, all_max + delta)
        ax.set_aspect(1)
        plt.legend(fontsize=13, loc='upper left')
        plt.xlabel("Ground truth", fontsize=12)
        plt.ylabel("Prediction value", fontsize=12)
    # plt.suptitle("Regression Performance",)
    plt.tight_layout()
    plt.savefig(file_path, format='jpg', dpi=1000, transparent=True, bbox_inches='tight')
    if show:
        plt.show()

def align(chi_exp, I_exp, corruption_offset=None):
    """*** transform and reorder."""
    chi_exp = np.array(chi_exp)
    delta_ = [chi_exp[i + 1] - chi_exp[i] for i in range(len(chi_exp) - 1)]
    vals, counts = np.unique(delta_, return_counts=True)
    ind = np.argmax(counts)
    delta = vals[ind]

    arg = (chi_exp < 0)
    chi_exp[arg] = chi_exp[arg] + 360
    arg = (chi_exp > 360)
    chi_exp[arg] = chi_exp[arg] - 360

    arg = np.argsort(chi_exp)
    chi_exp = chi_exp[arg]
    I_exp = I_exp[:, arg]

    """    *** verify the gap intervals, and define a useful function. """
    f_extend = lambda start, delta, m: [start + delta * (i + 1) for i in range(m)]
    chi_bool = [];
    chi_whole = [];
    I_whole = []
    for i in range(len(chi_exp)):
        chi_bool.append(True);
        chi_whole.append(chi_exp[i]);
        I_whole.append(I_exp[:, i])
        m = int(np.round((chi_exp[i + 1] - chi_exp[i]) / delta))
        if m != 1:
            chi_bool = chi_bool + [False] * (m - 1)
            chi_whole = chi_whole + f_extend(chi_exp[i], delta, m - 1)
            I_whole = I_whole + f_extend(np.zeros(I_exp.shape[0]), 0, m - 1)
        if i+1 == len(chi_exp) - 1:
            chi_bool.append(True);
            chi_whole.append(chi_exp[i+1]);
            I_whole.append(I_exp[:, i+1])
            break
    if len(chi_whole) != int(np.round(360 / delta)):
        m = int(np.round((360 - (chi_whole[-1] - chi_whole[0])) / delta))
        chi_bool = chi_bool + [False] * (m - 1)
        chi_whole = chi_whole + f_extend(chi_whole[-1], delta, m - 1)
        I_whole = I_whole + f_extend(np.zeros(I_exp.shape[0]), 0, m - 1)


    """    *** transform and reorder again. """
    chi_whole = np.array(chi_whole)
    arg = chi_whole > 360
    chi_whole[arg] = chi_whole[arg] - 360

    arg = np.argsort(chi_whole)
    chi_whole = chi_whole[arg]
    I_whole = np.stack(I_whole, axis=1)
    I_whole = I_whole[:, arg]
    chi_bool = np.bool_(chi_bool)
    chi_bool = chi_bool[arg]
    """corruption data point around the gaps. """
    if corruption_offset:
        for _ in range(corruption_offset):
            flag=0
            chi_bool_ = chi_bool.copy()
            for i in range(len(chi_whole) - 1):
                flag = int(chi_bool[i+1]) - int(chi_bool[i])
                if flag == 1:
                    chi_bool_[i+1] = False
                    I_whole[:, i+1] = 0
                elif flag == -1:
                    chi_bool_[i] = False
                    I_whole[:, i] = 0
            i = len(chi_whole) - 1
            flag = int(chi_bool[0]) - int(chi_bool[i])
            if flag == 1:
                chi_bool_[0] = False
                I_whole[:, 0] = 0
            elif flag == -1:
                chi_bool_ [i]= False 
                I_whole[:, i] = 0
            chi_bool = chi_bool_                
    
    return chi_bool, chi_whole, I_whole

def SortFunc(t1=None):

    t1 = t1.copy()
    if len(t1.shape) != 2:
        t1 = t1.reshape(1, -1)
    idx = t1[:, 2] < t1[:, 6]
    t1[idx] = t1[idx][:, [4, 5, 6, 7, 0, 1, 2, 3, 8]]
    t1[idx, -1] = 1 / t1[idx, -1]
    t1[:, 3] = np.abs(t1[:, 3])
    t1[:, 7] = np.abs(t1[:, 7])

    return t1

def correct(x1):
    x = x1.copy()
    x[:, 3] = np.abs(x[:, 3])
    x[:, 7] = np.abs(x[:, 7])
    x[:, 8] = np.abs(x[:, 8])
    return x

def sortlabel(y, label='beta'):
    labels = ['alpha', 'beta', 'gamma', 'dgamma', 'scale']
    assert label in labels, f'label must be in {labels}'
    arg = labels.index(label)
    y = y.copy()
    idx1 = y[:, arg] < y[:, arg + 5]
    y[idx1] = y[idx1][:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]]
    return y.copy()

class Transit(nn.Module):
    def __init__(self, func='abs', scale=1):
        super(Transit, self).__init__()
        self.scale = scale
        self.func = func
    def forward(self, X):
        if self.func == 'abs':
            X = torch.abs(X)
        if self.func == 'sigmoid':
            X = self.scale * torch.sigmoid(X) - (self.scale - 1) / 2
        return X

def net_constructor(in_features, targets, num_hiddens=512, rate = 0.7, num_layers = 5, drop1=None, drop2=None):
    from models import MLP
    net_a = MLP(in_features, num_hiddens, targets, num_layers=num_layers, rate=rate, drop=drop1)
    net_b = MLP(targets, net_a.num_features, in_features, num_layers=num_layers, rate=1.0 / rate, drop = drop2)
    net = nn.Sequential(net_a, net_b)
    return net


def quality_identify(I_exp):
    """return qualified sample index. """
    quantile = 0.05
    sort_exp = np.sort(I_exp.copy(),  axis=1)
    ind = np.arange(I_exp.shape[0]).tolist();ind_ = ind.copy()
    for i in ind_:
        counts =sort_exp[i][-int(sort_exp.shape[1]*quantile)]
        if counts <=50:
            ind.remove(i)
    ind = np.array(ind, dtype=np.int32)
    return ind

def read_expdata(file='83744', corruption_offset=False):
    exp_files = ['83744_I_chi.h5', '190127_I_chi.h5', 'third_I_chi.h5']
    exp_dir =  '/home/linazhao/mhsvn/diffrac_ring/Data/experiment_data/'
    if file == '83744':
        exp_file = exp_files[0]
    elif file == '190127':
        exp_file = exp_files[1]
    else:
        exp_file = exp_files[2]
    exp_path = os.path.join(exp_dir, exp_file)
    with h5py.File(exp_path, 'r') as f:
        try:
            chi_exp_ = f['chi'][:]
            I_exp_ = f['intensity'][:]
        except KeyError:
            chi_exp_ = f['chi'][:]
            I_exp_ = f['I'][:]
        try:
            name_list = f['name_list'][:]
        except KeyError:
            name_list = np.arange(1, len(y) + 1).tolist()
    chi_exp_ = chi_exp_[0] if len(chi_exp_.shape) == 2 else chi_exp_
    identified_index = quality_identify(I_exp_)

    I_exp_ = (I_exp_.T/np.max(np.abs(I_exp_), axis=1)).T
    I_exp_[I_exp_ < 0.0] = 0.0
    mask_bool, chi_exp, I_exp = align(chi_exp_, I_exp_, corruption_offset=corruption_offset)
    return chi_exp_, I_exp_, name_list, identified_index, mask_bool, chi_exp, I_exp

def obtain_tradresult(file='83744'):
    data_dir = '/home/linazhao/mhsvn/diffrac_ring/Data/experiment_data/'
    sim_labels = ['alpha_1', 
                        'beta_1',
                   'gamma_1', 
                    'dgamma_1',
                     'alpha_2',
                    'beta_2',
                    'gamma_2',
                    'dgamma_2',
                    'scale_1',
                    'scale_2',
                       ]
    labels_tmp = ['Alpha1', 'Beta1', 'gamma0_1', 'dgamma0_1',
           'Alpha2', 'Beta2', 'gamma0_2',  'dgamma0_2', 'scale1', 'scale2']
    if file == '83744':
        tra_result = pd.read_csv(os.path.join(data_dir, '83744_tra_modeling_result.csv'), sep=',', index_col=0, header=0)
    else:
        tra_result = pd.read_csv(os.path.join(data_dir, '190127_tra_modeling_result.csv'), sep=',', index_col=0, header=0)
    tra_result = tra_result.loc[:, labels_tmp]
    tra_result.columns = sim_labels
    tra_result = pd.concat([tra_result, pd.DataFrame(columns=['scale_ratio', ], 
                                                     data = tra_result.loc[:, 'scale_1'] / tra_result.loc[:, 'scale_2'])], axis = 1)
    if file == '83744':
        df_tmp = -tra_result.loc[:, 'beta_1'].copy()
        tra_result.loc[:, 'beta_1'] = -tra_result.loc[:, 'alpha_1']
        tra_result.loc[:, 'alpha_1'] = df_tmp
        df_tmp = -tra_result.loc[:, 'beta_2'].copy()
        tra_result.loc[:, 'beta_2'] = -tra_result.loc[:, 'alpha_2']
        tra_result.loc[:, 'alpha_2'] = df_tmp
    else:
        tra_result.loc[:, 'beta_1'] = -tra_result.loc[:, 'beta_1']
        tra_result.loc[:, 'beta_2'] = -tra_result.loc[:, 'beta_2']
    tra_result = tra_result.loc[:, ['alpha_1', 
                        'beta_1',
                   'gamma_1', 
                    'dgamma_1',
                     'alpha_2',
                    'beta_2',
                    'gamma_2',
                    'dgamma_2',
                    'scale_ratio',
                    'scale_1',
                    'scale_2'
                       ]]
    return tra_result

def physics_recons(y_true, chi=None, interpolate_chi=None):
    sim_labels = ['alpha_1', 
                        'beta_1',
                   'gamma_1', 
                    'dgamma_1',
                     'alpha_2',
                    'beta_2',
                    'gamma_2',
                    'dgamma_2',
                    'scale_1',
                    'scale_2',
                       ]
    if chi == None and interpolate_chi == None:
        chi = np.arange(0, 2 * np.pi, 2 * np.pi / 360)
        interpolate_chi = np.arange(1, 360, 2) / 360 * 2 * np.pi
    recons = []
    for j in range(len(y_true)):
        params = {sim_labels[i]:y_true[j, i] for i in range(9)}
        params['scale_2'] = 1.0
        I_chi, _ = simulation(**params)
        I_chi = I_chi.reshape(-1)
        I_chi = I_chi/max(I_chi)
        I_chi = interpolate_func(chi, I_chi, interpolate_chi)
        recons.append(I_chi)
    recons = np.array(recons)
    return recons


def tolist(json_str):
    for key in json_str.keys():
        if isinstance(json_str[key], dict):
            tolist(json_str[key])
        if isinstance(json_str[key], np.ndarray):
            json_str[key] = json_str[key].tolist()

class Transform(object):

    def __init__(self, ):

        self.X = None
        self.y = None
    def get_xy(self, angle):
        angle = angle * 2
        x = 10 * np.cos(angle * np.pi / 180)
        y = 10 * np.sin(angle * np.pi / 180)
        return np.stack([x, y], axis=1)
    def get_angle(self, x, y):
        radius = np.sqrt(x ** 2 + y ** 2)
        angles1 = np.sort(np.stack([np.degrees(np.arccos(x / radius)), 
                                    - np.degrees(np.arccos(x / radius))], axis = 1), axis = 1)
        angles2 = np.sort(np.stack([np.sign(np.degrees(np.arcsin(y / radius))) * 180 - np.degrees(np.arcsin(y / radius)), 
                                    np.degrees(np.arcsin(y / radius))] , axis=1), axis = 1)
        
        arr_bool1 = (np.around((angles2 - angles1[:, 0].copy().reshape(-1, 1)), 1) == 0)
        arr_bool2 = (arr_bool1[:, 0] | arr_bool1[:, 1])
        idx = np.ones(len(x), dtype=np.int32)
        idx[arr_bool2] = 0
        idx = idx.reshape(-1, 1)
        angles = np.take_along_axis(angles1, idx, axis=1)
        return angles / 2

    def transform(self, x):
        self.X = x.copy()
        part1_idx = self.X[:, -1] <=1;part2_idx = self.X[:, -1]>1
        self.X[part1_idx, -1] = (self.X[part1_idx, -1] - 1/15.0) / (1.0 - 1/15.0) * 0.5
        self.X[part2_idx, -1] = (self.X[part2_idx, -1] -1) / (15.0 - 1.0)*0.5 + 0.5
        output1 = []
        for i in range(8):
            output1.append(self.get_xy(self.X[:, i]))
        output1 = np.concatenate(output1, axis=1)
        self.X2 = np.concatenate([output1, self.X[:, -1].reshape(-1, 1)], axis=1)
        return self.X2

    def inverse_transform(self, y):
        y = y.copy()
        part1_idx = y[:, -1] <= 0.5; part2_idx = y[:, -1] > 0.5
        y[part1_idx, -1] = y[part1_idx, -1]/0.5 * (1.0 - 1/15.0) + 1/15.0
        y[part2_idx, -1] = (y[part2_idx, -1] - 0.5) / 0.5 * (15.0 - 1.0) + 1.0
        output2 = []
        for i in range(8):
            output2.append(self.get_angle(y[:, 2*i], y[:, 2 * i + 1]))
        output2 = np.concatenate(output2, axis=1)
        self.y2 = np.concatenate([output2, y[:, -1].reshape(-1, 1)], axis=1)
        return self.y2

class Scaler1(object):


    def __init__(self, ):

        self.X = None
        self.scale_array = [(-5, 5), (-90, 90), (-90, 90), (8, 90), (-5, 5), (-90, 90), (-90, 90), (8, 90)]
        self.y = None
        
    def transform(self, x):
        self.X = x.copy()
        part1_idx = self.X[:, -1] <=1;part2_idx = self.X[:, -1]>1
        self.X[part1_idx, -1] = (self.X[part1_idx, -1] - 1/15.0) / (1.0 - 1/15.0) * 0.5
        self.X[part2_idx, -1] = (self.X[part2_idx, -1] -1) / (15.0 - 1.0)*0.5 + 0.5
        for i in range(len(self.scale_array)):
            self.X[:, i] = (self.X[:, i] - self.scale_array[i][0]) / (self.scale_array[i][1] - self.scale_array[i][0])

        return self.X

    def inverse_transform(self, y):

        y = y.copy()
        for i in range(len(self.scale_array)):
            y[:, i] = y[:, i] * (self.scale_array[i][1] - self.scale_array[i][0]) + self.scale_array[i][0]
        
        part1_idx = y[:, -1] <= 0.5; part2_idx = y[:, -1] > 0.5
        y[part1_idx, -1] = y[part1_idx, -1]/0.5 * (1.0 - 1/15.0) + 1/15.0
        y[part2_idx, -1] = (y[part2_idx, -1] - 0.5) / 0.5 * (15.0 - 1.0) + 1.0
        
        self.y = y
        return self.y

class Scaler2(object):

    def __init__(self, ):

        self.X = None
        self.scale_array = [(-5, 5), (-90, 90), (-90, 90), (8, 90), (-5, 5), (-90, 90), (-90, 90), (8, 90), (1, 15)]
        self.y = None

        
    def transform(self, x):
        self.X = x.copy()
        for i in range(len(self.scale_array)):
            self.X[:, i] = (self.X[:, i] - self.scale_array[i][0]) / (self.scale_array[i][1] - self.scale_array[i][0])

        return self.X

    def inverse_transform(self, y):

        y = y.copy()
        for i in range(len(self.scale_array)):
            y[:, i] = y[:, i] * (self.scale_array[i][1] - self.scale_array[i][0]) + self.scale_array[i][0]
        self.y = y.copy()
        return self.y



def save_model(net, name):
    path = os.path.join(".", name) 
    torch.save(net.state_dict(), path)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

