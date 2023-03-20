import torch
import torch.nn as nn
import math, pdb
from torch.autograd import Variable
import argparse
import numpy as np
import scipy.io
import torch.nn.functional as F


class PHY(torch.nn.Module):
    def __init__(self, args):
        super(PHY, self).__init__()
        self.args = args
        self.N = args.N
        self.M = args.M

    def compute_PAPR(self, x_t):
        # ---------------------------------------------------------- compute PAPR
        data_t_power = torch.square(torch.abs(x_t))
        meanPower = torch.mean(data_t_power, dim = 1)
        # power of x_t =(\approx) power of data_t / sps
        # print(self.power(x_t))
        maxPower = torch.max(data_t_power, dim = 1).values
        PAPR = maxPower/meanPower
        return PAPR

    def clip(self, x):
        # with torch.no_grad():
        x_power_mean_amp = torch.sqrt(torch.mean(torch.square(torch.abs(x)), dim = 1))
        thres = self.args.clip * x_power_mean_amp
        non_neg_diff = F.relu(torch.abs(x) - thres.unsqueeze(1))
        x = (1 - non_neg_diff/(torch.abs(x)+1e-8)) * x # scale the symbol with amplitude larger than thres
        return x

           
    def channel(self, data_x):
        # AWGN channel, noiseless feedback
        len_data_x = data_x.size(1)
        noise_std1 = 10 ** (-self.args.snr1 * 1.0 / 10 / 2)
        noise_std2 = 10 ** (-self.args.snr2 * 1.0 / 10 / 2)
        # feedforward channel
        AWGN1 = torch.normal(0, std=noise_std1, size=(self.inputBS, len_data_x, 2), requires_grad=False).to(self.args.device)
        AWGN2 = torch.normal(0, std=noise_std2, size=(self.inputBS, len_data_x, 2), requires_grad=False).to(self.args.device)
        AWGN1 = torch.view_as_complex(AWGN1)
        AWGN2 = torch.view_as_complex(AWGN2)
        if self.args.snr2 == 100:
            AWGN2 = AWGN2 * 0

        data_r_B = data_x + AWGN1
        data_r_A = data_r_B + AWGN2
        return data_r_B, data_r_A

    def RX(self, data_r):
        # FFT
        y = data_r.unsqueeze(1)
        y = torch.fft.fft(y, n=None, dim=-1, norm="ortho")
        # demapping
        y = torch.index_select(y, 2, self.loc)
        # IDFT, if necessary
        if self.args.precoding == 1:
            # y = torch.fft.fftshift(y, dim = -1)
            y = torch.fft.ifft(y, dim=-1, norm="ortho")
        # reshape to a packet, BS*4*64 -> BS*numOFDMsymbol
        y = y.view(self.inputBS, self.numOFDM * self.N)
        # complex to real, BS*numOFDMsymbol*2
        y = torch.view_as_real(y)
        return y

    def forward(self, dataTx):
        self.inputBS = dataTx.size(0)
        # input is real with shape(x) = (inputBS, ell, 1)
        if np.mod(self.args.ell, 2) == 1:
            dataTx = torch.cat([dataTx, torch.zeros(size = (self.inputBS, 1, 1), requires_grad=False).to(self.args.device)],dim=1)
        # shape(x) = (inputBS, self.N*2, 1)
        dataTx = dataTx.reshape(self.inputBS, self.N, 2)
        x = torch.view_as_complex(dataTx)
        # x is complex with shape(x) = (inputBS, self.N, 1)
        # =================================================================================== modulation
        # total subcarriers = M = 128, #allocated subcarriers = N = 64
        self.numOFDM = 1
        # ---------------------------------------------------------- modulation
        x = x.view(self.inputBS, self.numOFDM, self.N)

        # DFT precoding, if necessary
        if self.args.precoding == 1:
            x = torch.fft.fft(x, dim=-1, norm="ortho")
            # x = torch.fft.fftshift(x, dim = -1)
        # ------------------------------------------------ mapping to subcarriers
        # determine locations of subcarriers
        if self.args.mapping == 0:
            # OFDMA, random mapping
            loc = np.random.permutation(self.M)[:self.N]
            self.loc = torch.randperm(self.M)[:self.N]
        elif self.args.mapping == 1:
            # LFDMA, localized mapping
            # startloc = np.random.randint(self.M-self.N+1)
            # loc = np.arange(startloc, startloc+self.N)
            startloc = torch.randint(self.M-self.N+1, size=())
            self.loc = torch.arange(startloc, startloc+self.N)
        self.loc = self.loc.to(self.args.device)
        # power of x = 2; power of data_f = 1
        data_f = torch.zeros(self.inputBS, self.numOFDM, self.M, dtype = torch.complex64).to(self.args.device)
        data_f[:, :, self.loc] = x
        # IFFT; power of data_f = 1, power of data_t = 1
        # ---------------------------------------------------------- oversampling
        data_f = torch.cat([data_f, torch.zeros(size = (self.inputBS, 1, self.M *(self.args.sps-1)), requires_grad=False).to(self.args.device)], dim = 2)
        # ---------------------------------------------------------- OFDM modulation
        data_t = torch.fft.ifft(data_f, n=None, dim=-1, norm="ortho")

        # ---------------------------------------------------------- clipping
        data_x = data_t[:,0,:]
        if self.args.clip != 0.0:
            data_x = self.clip(data_x)
        # ---------------------------------------------------------- compute PAPR
        # PAPRdB.size() = (BatchSize,); PAPRloss.size() = (1,)
        PAPR = self.compute_PAPR(data_x)


        # =================================================================================== Channel
        data_r_B, data_r_A = self.channel(data_x)

        # =================================================================================== RX
        data_r_B = self.RX(data_r_B)
        noise_A = self.RX(data_r_A) - dataTx

        data_r_B = data_r_B.reshape(self.inputBS, -1)
        noise_A = noise_A.reshape(self.inputBS, -1)
        if np.mod(self.args.ell, 2) == 1:
            data_r_B = data_r_B[:,:-1]
            noise_A = noise_A[:,:-1]

        return PAPR, noise_A, data_r_B.unsqueeze(-1)
