import numpy as np
import scipy.fftpack as ft
from scipy.signal import convolve2d
import math
import pandas as pd
from typing import List, Union


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
    field_synthesis = np.zeros_like(F_hat)

    for a in range(field_synthesis.shape[1]):
        # Instaneous scan in frequency space
        T_hat_a = F_hat * np.roll(L_hat, a - field_synthesis.shape[1] // 2, 1)
        # Instaneous scan in object space
        T_a = ft.fftshift(ft.fft2(ft.ifftshift(T_hat_a)))
        # Incoherent summing of the intensities
        field_synthesis = field_synthesis + np.abs(T_a) ** 2

    return field_synthesis


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def osWidth_gui(prof):
    # df = pd.DataFrame(prof)

    # gives no. of rows along x-axis
    prof = prof.reshape((len(prof), 1))

    e = np.exp(1)

    # sum over rows for each of the column
    prof2 = np.cumsum(prof, axis=0)
    # np.amax(prof) = Maximum of the flattened array
    prof2_n = prof2 / np.amax(prof2)

    # Find the indices of the maximum values along each column
    p_maxint = np.argmax(prof, axis=0)

    num_par_y_array_size = 4096
    ny = num_par_y_array_size

    total_area = prof.sum(axis=0)
    guess_thickness_pxl = 3
    thickness63percent = 0

    while thickness63percent == 0:
        guess_thickness_pxl = guess_thickness_pxl + 1
        if guess_thickness_pxl == ny:
            thickness63percent = float('nan')
        # csum_norm is a 1xguessthickness_pxl array with all zeros
        # csum_norm = np.zeros(1, guessthickness_pxl)
        csum_norm = np.zeros((1, ny))
        guess_start_point = max(p_maxint - guess_thickness_pxl + 1, 1)
        guess_end_point = min(p_maxint + guess_thickness_pxl - 1, ny) - guess_thickness_pxl

        for ii in range(guess_start_point[0], guess_end_point[0] + 1):
            prof_partial = np.array(prof)
            indices = list(range(ii, ii + guess_thickness_pxl + 1))
            prof_partial_sum = prof_partial[indices].sum()
            csum_norm[0, ii] = prof_partial_sum / total_area
            if not thickness63percent and csum_norm[0, ii] >= (1 - 1 / e):
                thickness63percent = guess_thickness_pxl

    return thickness63percent * 1


def osWidth_gui_2(prof):
    # df = pd.DataFrame(prof)

    e = np.exp(1)

    # sum over rows for each of the column
    prof_row = np.sum(prof, axis=1)

    num_par_y_array_size = 4096
    ny = num_par_y_array_size

    total_area = prof_row.sum(axis=0)
    guess_thickness_pxl = 0
    thickness63percent = 0

    while thickness63percent == 0:
        # csum_norm is a 1xguessthickness_pxl array with all zeros
        # csum_norm = np.zeros(1, guessthickness_pxl)
        csum_norm = np.zeros((1, ny))

        for ii in range(1, 2048):
            guess_thickness_pxl = guess_thickness_pxl + 2
            prof_partial = np.array(prof_row)
            indices = list(range(2048 - ii, 2048 + ii + 1))
            prof_partial_sum = prof_partial[indices].sum()
            csum_norm[0, ii] = prof_partial_sum / total_area
            if not thickness63percent and csum_norm[0, ii] >= (1 - 1 / e):
                thickness63percent = guess_thickness_pxl

    return thickness63percent


def analysisbeam_gui(PSFSumY):
    cy = 2048
    centerline = PSFSumY[cy - 1]
    num_par_dx = 1
    pMaxPos = pWidth_gui(centerline, num_par_dx)

    prof_peak = PSFSumY[:][pMaxPos - 1]
    os_peak_e = osWidth_gui_2(prof_peak)

    return os_peak_e


def pWidth_gui(prof, dy):
    df = pd.DataFrame(prof)

    # gives no. of rows along x-axis
    if len(df) == 1:
        prof = np.transpose(prof)

    # Find the indices of the maximum values along each column
    pMaxPos = np.argmax(prof, axis=0)

    return pMaxPos


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


def compute_scores(model):
    # df = pd.DataFrame(prof)
    e = np.exp(1)

    # sum over rows for each of the column
    prof_row = np.sum(model, axis=1)

    num_par_y_array_size = 4096
    ny = num_par_y_array_size

    totalarea = prof_row.sum(axis=0)
    guessthickness_pxl = 0
    thickness63percent = 0

    while thickness63percent == 0:
        # csum_norm is a 1xguessthickness_pxl array with all zeros
        # csum_norm = np.zeros(1, guessthickness_pxl)
        csum_norm = np.zeros((1, ny))

        for ii in range(1, 2048):
            guessthickness_pxl = guessthickness_pxl + 2
            prof_partial = np.array(prof_row)
            indices = list(range(2048 - ii, 2048 + ii + 1))
            prof_partial_sum = prof_partial[indices].sum()
            csum_norm[0, ii] = prof_partial_sum / totalarea
            if not thickness63percent and csum_norm[0, ii] >= (1 - 1 / e):
                thickness63percent = guessthickness_pxl

    return thickness63percent


def load_model(width, position):
    n = 4096
    offset = 256
    v = []
    for i in range(0, n):
        v.append(i - math.floor(n / 2))

    kspace = compute_kspace(width)
    mask = compute_mask(width, position)
    masked_kspace = compute_masked_kspace(kspace, mask)

    field_synthesis_profile = ft.fftshift(ft.ifft(ft.ifftshift(masked_kspace)))
    df = pd.DataFrame(field_synthesis_profile)
    lattice_efield = ft.fftshift(ft.ifft2(ft.ifftshift(masked_kspace)))
    lattice = abs(lattice_efield) ** 2
    lattice_hat = ft.fftshift(ft.fft2(ft.ifftshift(lattice)))

    period = n / offset
    if period == round(period):
        recon = conv2(lattice, np.ones((1, int(period))) / period, 'same')
    else:
        for x in lattice_hat:
            for y in x:
                recon = y * (np.sinc(v / period))
        recon = ft.fftshift(ft.ifft2(ft.ifftshift(recon)))
    return recon