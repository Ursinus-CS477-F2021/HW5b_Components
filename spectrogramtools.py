"""
Programmer: Chris Tralie
Purpose: To provide some basic spectrogram tools, including Griffin-Lim inversion,
         so that no dependencies other than numpy are needed
"""
import numpy as np
import matplotlib.pyplot as plt

""" WINDOW FUNCTIONS """
halfsine_window = lambda w: np.sin(np.pi*np.arange(w)/float(w))
hann_window = lambda w: 0.5*(1 - np.cos(2*np.pi*np.arange(w)/w))
def blackman_harris_window(w):
    t = 2*np.pi*np.arange(w)/w
    return 0.35875 - 0.48829*np.cos(t) + 0.14128*np.cos(2*t) - 0.01168*np.cos(3*t)

def plot_specgram(S, sr, win, hop):
    A = np.log10(np.abs(S + 1e-3))
    ts = (0, hop*S.shape[1]/sr)
    ys = (0, sr*S.shape[0]/win)
    plt.imshow(A, cmap='magma_r', extent=(ts[0], ts[1], ys[1], ys[0]), aspect='auto')
    plt.gca().invert_yaxis()
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (hz)")


def stft(x, win, hop, winfunc = blackman_harris_window):
    """
    Parameters
    ----------
    x: ndarray(N)
        The audio signal
    win: int
        Window length
    hop: int
        Hop length
    winfunc: function w->ndarray(w)
        Handle to a window function
    
    Returns
    -------
    ndarray(NBins, NWindows)
        The non-redundant complex half of the spectrogram
    """
    Q = win/hop
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    winfn = winfunc(win)
    NWin = int(np.floor((x.size - win)/float(hop)) + 1)
    S = np.zeros((win, NWin), dtype = np.complex)
    for i in range(NWin):
        S[:, i] = np.fft.fft(winfn*x[np.arange(win) + (i-1)*hop])
    #Second half of the spectrum is redundant for real signals
    if win%2 == 0:
        #Even Case
        S = S[0:int(win/2)+1, :]
    else:
        #Odd Case
        S = S[0:int((win-1)/2)+1, :]
    return S

def istft(pS, win, hop, winfunc = blackman_harris_window):
    """
    Inverse short time Fourier Transform

    Parameters
    ----------
    pS: ndarray(NBins, NWindows)
        A complex non-redundant half of an STFT to invert
    win: int
        Window length
    hop: int
        Hop length
    winfunc: function w->ndarray(w)
        Handle to a window function
    
    Returns
    -------
    x: ndarray(N)
        Inverted audio
    """
    #First put back the entire redundant STFT
    S = np.array(pS, dtype = np.complex)
    if win%2 == 0:
        #Even Case
        S = np.concatenate((S, np.flipud(np.conj(S[1:-1, :]))), 0)
    else:
        #Odd Case
        S = np.concatenate((S, np.flipud(np.conj(S[1::, :]))), 0)
    
    #Figure out how long the reconstructed signal actually is
    N = win + hop*(S.shape[1] - 1)
    x = np.zeros(N, dtype = np.complex)
    
    #Setup the window
    Q = win/hop
    if Q - np.floor(Q) > 0:
        print('Warning: Window size is not integer multiple of hop size')
    winfn = winfunc(win)/(Q/2.0)

    #Do overlap/add synthesis
    for i in range(S.shape[1]):
        x[i*hop:i*hop+win] += winfn*np.fft.ifft(S[:, i])
    return x

def griffin_lim(SAbs, win, hop, n_iters=10, winfunc = blackman_harris_window):
    """
    Do Griffin Lim phase retrieval

    Parameters
    ----------
    Parameters
    ----------
    SAbs: ndarray(NBins, NWindows)
        A complex non-redundant half of an STFT to invert
    win: int
        Window length
    hop: int
        Hop length
    n_iters: int
        Number of iterations to do
    winfunc: function w->ndarray(w)
        Handle to a window function
    
    Returns
    -------
    ndarray(N):
        Inverted audio
    """
    S = SAbs
    for i in range(n_iters):
        A = stft(istft(S, win, hop, winfunc), win, hop, winfunc)
        Phase = np.arctan2(np.imag(A), np.real(A))
        S = SAbs*np.exp(np.complex(0, 1)*Phase)
    x = istft(S, win, hop, winfunc)
    return np.real(x)