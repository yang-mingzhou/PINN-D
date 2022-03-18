import torch
from torch import nn, optim
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import csv
import time
import visdom
from tqdm import tqdm

import torch.profiler
import torch.utils.data

#from torchinterp1d import Interp1d
import math


class MapeLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, label, pred):
        mask = label >= 1e-4

        mape = torch.mean(torch.abs((pred[mask] - label[mask]) / label[mask]))
        # print(p, l, loss_energy)
        return mape


# #version 1 vd=>vt=>Same time interpolation
# def vd2vtWithInterpolation(velocityProfile, length):
#     lengthOfEachPart = length / (velocityProfile.shape[1] - 1)
#     averageV = velocityProfile[:,0:-1] + velocityProfile[:,1:]
#     tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
#     tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1],tOnSubPath.shape[1])).to(device))
#     tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0],1]).to(device),tAxis],dim=-1)
#     timeSum = tAxis[:, -1]
#     tNew = torch.arange(tParts).unsqueeze(0).to(device) * timeSum.unsqueeze(-1) / (tParts - 1)
#     #print('tAxis',tAxis,'velocityProfile',velocityProfile, 'tNew',tNew)
#     intetpResult = None
#     intetpResult = Interp1d()(tAxis, velocityProfile, tNew, intetpResult).to(device)
#     ##plot interpolation
#     # x = tAxis.cpu().numpy()
#     # y = velocityProfile.cpu().numpy()
#     # xnew = tNew.cpu().numpy()
#     # yq_cpu = intetpResult.cpu().numpy()
#     # print(x.T, y.T, xnew.T, yq_cpu.T)
#     # plt.plot(x.T, y.T, '-', xnew.T, yq_cpu.T, 'x')
#     # plt.grid(True)
#     # plt.show()
#     return intetpResult,tNew

#version 1 vd=>vt
def vd2vt(velocityProfile, length):
    '''

    :param velocityProfile: velocityProfile: velocity profiel (uniform length sampling)
    :param length: length: total length of the segment
    :return: tAxis: the axis of time of the velocity profile
    '''
    lengthOfEachPart = length / (velocityProfile.shape[1] - 1)
    averageV = velocityProfile[:,0:-1] + velocityProfile[:,1:]
    tOnSubPath = (2 * lengthOfEachPart).unsqueeze(-1) / averageV
    tAxis = torch.matmul(tOnSubPath, torch.triu(torch.ones(tOnSubPath.shape[1],tOnSubPath.shape[1])).to(device))
    tAxis = torch.cat([torch.zeros([tOnSubPath.shape[0],1]).to(device),tAxis],dim=-1)
    return velocityProfile,tAxis


#version 1 vt=>calculate t
def vt2t(velocityProfile, length):
    '''

    :param velocityProfile: velocity profiel (uniform time sampling)
    :param length: total length of the segment
    :return: tAxis: the axis of time of the velocity profile
    '''
    mul = torch.cat([torch.ones([1, 1]), 2 * torch.ones([velocityProfile.shape[1] - 2, 1]),\
                       torch.ones([1, 1])], dim=0).to(device)
    vAverage = torch.matmul(velocityProfile, mul) / 2
    tForOnePart = length.unsqueeze(-1) / vAverage
    tAxis = torch.arange(velocityProfile.shape[1]).unsqueeze(0).to(device) * tForOnePart
    return velocityProfile,tAxis


def timeEstimation(tNew):
    return tNew[:, -1]


def fuelEstimation(v, tNew, acc, m, height, length):
    sin_theta = height / length
    fuel = vt2fuel(v, acc, tNew, m, sin_theta)
    return fuel

def vt2a(v,t):
    # calculate a using 1-3, 2-4
    dv = v[:, 2:] - v[:, 0:-2]
    dt = t[:, 2].unsqueeze(-1)
    aMiddle = dv / dt
    dvHead = v[:, 1] - v[:, 0]
    dtSingle = t[:, 1]
    aHead = (dvHead / dtSingle).unsqueeze(-1)
    dvTail = v[:, -1] - v[:, -2]
    aTail = (dvTail / dtSingle).unsqueeze(-1)
    a = torch.cat([aHead, aMiddle, aTail], dim=-1)
    return a


def power(v,a,m,sin_theta,rho):
    R = 0.5003  # wheel radius (m)
    g = 9.81  # gravitational accel (m/s^2)
    A = 10.5  # frontal area (m^2)
    Cd = 0.5  # drag coefficient
    Crr = 0.0067  # rolling resistance
    Iw = 10  # wheel inertia (kg m^2)
    Nw = 10  # number of wheels

    Paccel = (m * a * v).clamp(0)
    #Pascent =(m * g * torch.sin(torch.tensor([theta * (math.pi / 180)]).to(device)) * v).clamp(0)
    Pascent = (m * g * sin_theta.unsqueeze(-1) * v).clamp(0)
    Pdrag = (0.5 * rho * Cd * A * v ** 3).clamp(0)
    Prr = (m * g * Crr * v).clamp(0)
    #Pinert = (Iw * Nw * (a / R) * v).clamp(0)
    # pauxA = torch.zeros(Pinert.shape).to(device)
    Paux = 1000
    #Paux = torch.where(v>0.1, pauxA, pauxB).to(device)
    #Paux = 0
    P = (Paccel + Pascent + Pdrag + Prr +Paux) / 1000
    return P


def vt2fuel(v,a,t,m,sin_theta ):

    #m = 10000  # mass (kg)
    rho = 1.225  # density of air (kg/m^3)
    #theta = 0  # road grade (degrees)
    fc = 40.3  # fuel consumption (kWh/gal) # Diesel ~40.3 kWh/gal
    eff = 0.56  # efficiency of engine

    P = power(v, a, m, sin_theta, rho)
    P_avg = (P[:, :-1] + P[:, 1:]) / 2
    f = P_avg / (fc * eff) * t[:, 1].unsqueeze(-1) / 3600
    #from galon => 10ml
    return torch.sum(f, dim=1)*3.7854*100