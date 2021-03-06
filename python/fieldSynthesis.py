from typing import List, Union
import numpy as np
import scipy.fftpack as ft
import scipy.signal as sig
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from PIL import Image
import pandas as pd
from scipy.signal import convolve2d

''' Field Synthesis
Python-based demonstration of Field Synthesis

supplementary material to:
    Universal Light-Sheet Generation with Field Synthesis
    Bo-Jui Chang, Mark Kittisopikul, Kevin M. Dean, Phillipe Roudot, Erik Welf and Reto Fiolka.

Mark Kittisopikul
Goldman Lab
Northwestern University

November 2018

Field Synthesis Demonstration -
Python code to demonstrate field synthesis light sheet microscopy
Copyright (C) 2019 Reto Fioka,
              University of Texas Southwestern Medical Center
Copyright (C) 2019 Mark Kittisopikul,
              Northwestern University
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


def createAnnulus(n=256, r=32, w=4):
    ''' createAnnulus - create a ring-like structure
    INPUT
    n - size of square array or vector
    r - radius of the ring
    w - width of the ring
    OUTPUT
    an array n x n
    '''
    if np.isscalar(n):
        v = np.arange(n)
        v = v - np.floor(n / 2)
    else:
        v = n

    y, x = np.meshgrid(v, v)
    q = np.hypot(x, y)
    annulus = abs(q - r) < w

    return annulus


def doConventionalScan(Fsqmod, Lsqmod):
    '''Simulate Conventional digital scanning / dithering
        INPUT
        F_sqmod - Square modulus of F at the front focal plane
        L_sqmod - Square modulus of L at the front focal plane
        OUTPUT
        scanned - Scanned (dithered) intensity of Fsqmod by Lsqmod
    '''
    # Manually scan by shifting Fsqmod and multiplying by Lsqmod
    scanned = np.zeros(Fsqmod.shape)
    center = Lsqmod.shape[1] // 2

    for x in range(np.size(Fsqmod, 1)):
        scanned = scanned + np.roll(Fsqmod, x - center, 1) * Lsqmod[center, x]

    return scanned


def doConventionalScanHat(F_hat, L_hat):
    '''Simulate Conventional digital scanning / dithering from frequency space representations
       INPUT
       F_hat - Mask at back focal plane
       L_hat - Line scan profile in frequency space at the back focal plane
       OUTPUT
       scanned - Scanned (dithered) intensity of Fsqmod by Lsqmod at front focal plane
    '''
    F_hat = ft.ifftshift(F_hat)
    F = ft.ifft2(F_hat)
    F = ft.fftshift(F)
    # This is the illumination intensity pattern
    Fsqmod = np.real(F * np.conj(F))

    L_hat = ft.ifftshift(L_hat)
    L = ft.ifft2(L_hat)
    L = ft.fftshift(L)
    Lsqmod = L * np.conj(L)

    scanned = doConventionalScan(Fsqmod, Lsqmod)
    return scanned


def doFieldSynthesisLineScan(F_hat, L_hat):
    '''Simulate Field Synthesis Method
        INPUT
        F_hat - Frequency space representation of illumination pattern, mask at back focal plane
        L_hat - Line scan profile in frequency space at the back focal plane
        OUTPUT
        fieldSynthesis - Field synthesis construction by doing a line scan in the back focal plane
    '''
    # Do the Field Synthesis method of performing a line scan at the back focal plane
    fieldSynthesis = np.zeros_like(F_hat)

    for a in range(fieldSynthesis.shape[1]):
        # Instaneous scan in frequency space
        T_hat_a = F_hat * np.roll(L_hat, a - fieldSynthesis.shape[1] // 2, 1)
        # Instaneous scan in object space
        T_a = ft.fftshift(ft.fft2(ft.ifftshift(T_hat_a)))
        # Incoherent summing of the intensities
        fieldSynthesis = fieldSynthesis + np.abs(T_a) ** 2

    return fieldSynthesis


def demoFieldSynthesis():
    '''Demonstrate Field Synthesis Method with Plots
        INPUT
         None
        OUTPUT
         None
    '''
    # FixMe
    # plt.rc('text', usetex=True)
    fig, ax = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(16, 9))

    # Create F, the illumination pattern
    F_hat = createAnnulus()
    F_hat = ft.ifftshift(F_hat)
    F = ft.ifft2(F_hat)
    F = ft.fftshift(F)
    # This is the illumination intensity pattern
    Fsqmod = np.real(F * np.conj(F))

    # FIXME
    # plt.figure()
    # plt.title('F')
    # plt.imshow(Fsqmod, cmap='plasma')
    # plt.show(block=False)
    ax[0, 0].imshow(Fsqmod, cmap='plasma')
    ax[0, 0].set_title('F(x,z)')

    # Create L, the scan profile
    L = np.zeros_like(Fsqmod)
    center = L.shape[1] // 2
    sigma = 30
    L[center, :] = norm.pdf(np.arange(-center, center), 0, sigma)
    # L[L.shape[1]//2,:] = 1
    # The square modulus of L is the object space
    Lsqmod = L * np.conj(L)
    # This is the line scan profile used in Field Synthesis
    L_hat = ft.fftshift(ft.fft2(ft.ifftshift(L)))

    ax[0, 1].imshow(L, cmap='plasma')
    ax[0, 1].set_title('$ L(x)\delta(z) $')

    ax[0, 2].imshow(Lsqmod, cmap='plasma')
    ax[0, 2].set_title('$ |L(x)\delta(z)|^2 $')

    ax[0, 3].imshow(np.abs(L_hat), cmap='plasma')
    ax[0, 3].set_title('$\hat{L}(k_x) $')

    # Manually scan by shifting Fsqmod and multiplying by Lsqmod
    scanned = doConventionalScan(Fsqmod, Lsqmod)

    ax[1, 0].imshow(scanned, cmap='plasma')
    ax[1, 0].set_title('Scanned: $ \sum_{x\'} |F(x\',z)|^2|L(x-x\')|^2 $')

    # Manually scanning is a convolution operation
    # There are potentially boundary effects here
    convolved = sig.convolve(Fsqmod, Lsqmod, 'same')

    ax[1, 1].imshow(convolved, cmap='plasma')
    ax[1, 1].set_title('Convolved: $ |F(x,z)|^2 ** |L(x)\delta(z)|^2 $')

    # This manual implementation of Fourier transform based convolution
    # actually does circular convolution
    convolvedft = ft.fftshift(ft.fft2(ft.ifft2(ft.ifftshift(Fsqmod)) * ft.ifft2(ft.ifftshift(Lsqmod))))
    convolvedft = np.real(convolvedft)

    ax[1, 2].imshow(convolvedft, cmap='plasma')
    ax[1, 2].set_title(
        r'Convolved FT: $ \mathcal{F}^{-1} \{ \mathcal{F}\{|F|^2\} \mathcal{F}\{|L(x)\delta(z)|^2\} \} $')

    # Do the Field Synthesis method of performing a line scan at the back focal plane
    fieldSynthesis = doFieldSynthesisLineScan(F_hat, L_hat)

    ax[1, 3].imshow(fieldSynthesis, cmap='plasma')
    ax[1, 3].set_title('Field Synthesis: $ \sum_a |\mathcal{F}^{-1}\{ \hat{F}(k_x,k_z)\hat{L}(k_x-a) \}|^2 $')

    plt.show()
    plt.pause(0.001)


def shiftMatrixByK(mat, k):
    N = 4095
    matrix = [[]]
    j = 0
    if k < 0:
        k = -k
    else:
        k = N - k
    while j < N:
        # Print elements from
        # index k
        for i in range(k, N):
            matrix[j].append(mat[j][i])

        # Print elements before
        # index k
        for i in range(0, k):
            matrix[j].append(mat[j][i])

        j = j + 1
    return matrix


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def createAnnulus(n=256, r=32, w=4):
    """
    createAnnulus - create a ring-like structure
    INPUT
    n - size of square array or vector
    r - radius of the ring
    w - width of the ring
    OUTPUT
    an array n x n
    """
    if np.isscalar(n):
        v = np.arange(n)
        v = v - np.floor(n / 2)
    else:
        v = n

    y, x = np.meshgrid(v, v)
    q = np.hypot(x, y)
    annulus = abs(q - r) < w

    return annulus


def compute_kspace(width):
    n = 4096
    r = 256

    dispRange: List[Union[int, float]] = []
    for i in range(-600, 601):
        dispRange.append(i + math.floor(n / 2) + 1)
    v = []
    for i in range(0, n):
        v.append(i - math.floor(n / 2))
    kspace = createAnnulus(v, r, width)

    return kspace


def compute_mask(w, pos):
    offset = 256
    n = 4096
    v = []
    for i in range(0, n):
        v.append(i - math.floor(n / 2))
    initial_v = []
    for i in range(0, n):
        initial_v.append(-v[i])
    mask = []
    for i in range(0, n):
        if (offset * 0.99 / 1.35 + pos + w / 2 > initial_v[i] > offset * 0.99 / 1.35 + pos - w / 2) or \
                (offset * 0.99 / 1.35 - pos + w / 2 > initial_v[i] > offset * 0.99 / 1.35 - pos - w / 2):
            mask.append(1)
        else:
            mask.append(0)
    return mask


def compute_masked_kspace(kspace, mask):
    n = 4096
    masked_kspace = kspace

    for c in range(0, n):
        if not mask[c]:
            masked_kspace[:, c] = False
    masked_kspace = masked_kspace.astype(float)
    return masked_kspace


if __name__ == "__main__":
    n = 4096
    r = 256
    w = 5
    offset = 256
    dispRange: List[Union[int, float]] = []
    for i in range(-600, 601):
        dispRange.append(i + math.floor(n / 2) + 1)
    v = []
    for i in range(0, n):
        v.append(i - math.floor(n / 2))
    annulus = createAnnulus(v, r, w)
    abs_v = np.absolute(v)
    selected_columns = []
    for i in range(0, n):
        if (offset + w / 2 > abs_v[i] > offset - w / 2) or (w / 2 > v[i] > -w / 2):
            selected_columns.append(1)
        else:
            selected_columns.append(0)
    latticeFourierMask = annulus
    for c in range(0, n):
        if not selected_columns[c]:
            latticeFourierMask[:, c] = False
    latticeFourierMask = latticeFourierMask.astype(float)

    fieldSynthesisProfile = ft.fftshift(ft.ifft(ft.ifftshift(latticeFourierMask)))
    df = pd.DataFrame(fieldSynthesisProfile)
    sum_rows = df.sum(axis=1)
    fieldSynthesisProfile = sum_rows ** 2
    lattice_efield = ft.fftshift(ft.ifft2(ft.ifftshift(latticeFourierMask)))
    lattice = abs(lattice_efield) ** 2
    df1 = pd.DataFrame(lattice)
    latticeLineProfile = df1.sum(axis=1)
    latticeLineProfile = latticeLineProfile * n

    lattice_hat = ft.fftshift(ft.fft2(ft.ifftshift(lattice)))

    period = n / offset
    if period == round(period):
        # latticeDithered = np.zeros((4096, 4096))
        # for s in range(math.floor(-period / 2), math.floor(period / 2) - 1):
        #     max = np.roll(lattice, s, axis=1)
        #     latticeDithered = np.add(latticeDithered, max)
        # latticeDithered = latticeDithered / period
        latticeDithered = conv2(lattice, np.ones((1, int(period))) / period, 'same')
    else:
        for x in lattice_hat:
            for y in x:
                latticeDithered = y * (np.sinc(v / period))
        latticeDithered = ft.fftshift(ft.ifft2(ft.ifftshift(latticeDithered)))

    latticeDithered_hat = ft.fftshift(ft.fft2(ft.ifftshift(latticeDithered)))

    fig = plt.figure()
    plt.figure(figsize=(16, 9))
    plt.gca().get_autoscale_on()

    ax1 = plt.subplot(231)
    ax1.set_xlim([1600, 2500])
    ax1.set_ylim([1600, 2500])
    aa = abs(latticeFourierMask)
    aa[aa > 1e-6] = 1e-6
    ax1.imshow(latticeFourierMask, cmap='hot')
    ax1.set_title('Electric field in pupil')

    aaa = abs(lattice_hat)
    aaa[aaa > 1e-6] = 1e-6
    ax2 = plt.subplot(232)
    ax2.set_xlim([1250, 2800])
    ax2.set_ylim([1250, 2800])
    ax2.imshow(abs(aaa), cmap='hot')
    ax2.set_title('Fourier components of lattice intensity')

    bbb = abs(latticeDithered_hat)
    bbb[bbb > 1e-6] = 1e-6
    ax3 = plt.subplot(233)
    ax3.set_xlim([1300, 2800])
    ax3.set_ylim([1300, 2800])
    ax3.imshow(abs(bbb), cmap='hot')
    ax3.set_title('Fourier components of dithered lattice intensity')

    ax4 = plt.subplot(234)
    ax4.imshow(np.real(lattice_efield), cmap='gray')
    ax4.set_title('Electric field of lattice at focal plane')
    ax4.set_xlim([1950, 2150])
    ax4.set_ylim([1950, 2150])

    ax5 = plt.subplot(235)
    ax5.imshow(lattice, cmap='hot')
    ax5.set_title('Intensity of lattice')
    ax5.set_xlim([1950, 2150])
    ax5.set_ylim([1950, 2150])

    ax6 = plt.subplot(236)
    ax6.imshow(latticeDithered, cmap='hot')
    ax6.set_title('Averaged Intensity of dithered lattice')
    ax6.set_xlim([1950, 2150])
    ax6.set_ylim([1950, 2150])

    plt.show()
    plt.pause(0.001)

    ##demoFieldSynthesis()
