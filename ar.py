from scipy.io import wavfile
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
import peakutils
from scipy.fftpack import rfft, irfft, fftfreq

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

'''
y = np.array([1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,
       1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,
       2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1])

# Settings: lag = 30, threshold = 5, influence = 0
lag = 30
threshold = 5
influence = 0

# Run algo with settings from above
result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

# Plot result
pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"], color="cyan", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.ylim(-1.5, 1.5)
pylab.show()

print(result["signals"])
'''

fs, data = wavfile.read('SDRSharp_20170830_073907Z_145825000Hz_IQ_autogain.wav')

d=np.abs(data.T[0][3900000:4000000]+1j*data.T[1][3900000:4000000])
#d=np.abs(data.T[0]+1j*data.T[1])
plt.plot(d)
plt.show()
#d=np.abs(data.T[0]+1j*data.T[1])
d_=d.copy()

N  = 3    # Filter order
Wn = 0.1 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')
d = signal.filtfilt(B,A, d)

print(signaltonoise(d))

fd=np.abs(np.fft.fft(d-np.mean(d)))
#fd=fd[:int(len(fd)/2)]
fd=fd[:800]
#fd=np.append([1*np.mean(fd[:15])]*15,fd[:1000])
#fd=np.append([0]*30,fd[:800])
plt.plot(fd)
plt.show()

fd_=np.reshape(fd,(len(fd),1))

FD=fd_.T*fd_
print(np.shape(FD))
plt.pcolormesh(FD)
plt.show()
print(np.argmax(FD,axis=0))
print(np.argmax(FD,axis=1))

indexes = peakutils.indexes(fd, thres=0.02/max(fd), min_dist=30)

print(indexes)

freq=np.fft.fftfreq(len(fd),1/fs)
print(freq[:10])


cut_f_signal = []
for i in range(500):
    cut_f_signal.append(fd[i])
cut_f_signal=cut_f_signal+[0]*(500)+cut_f_signal[::-1]
cut_f_signal_=np.reshape(cut_f_signal,(len(cut_f_signal),1))
Cut_f_signal=cut_f_signal_.T*cut_f_signal_
plt.pcolormesh(Cut_f_signal)
plt.show()

'''
cut_f_signal=np.array(cut_f_signal)
plt.plot(cut_f_signal)
plt.show()
cut_signal = np.abs(np.fft.ifft(cut_f_signal))
plt.plot(cut_signal)
plt.show()
print(signaltonoise(cut_signal))
'''

f, t, Sxx = signal.spectrogram(d, fs)
print(np.shape(Sxx))
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

'''
fig, (ax1) = plt.subplots(nrows=1)
Pxx, freqs, bins, im = plt.specgram(d, NFFT=10000, Fs=fs, noverlap=900)
plt.show()
'''

for i in range(0,len(f)-10,10): 
    f, t, Sxx = signal.spectrogram(d, fs)
    plt.pcolormesh(t, f[i:i+10], Sxx[:][i:i+10])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    

lag = 15
threshold = 5
influence = 0

result = thresholding_algo(fd, lag=lag, threshold=threshold, influence=influence)

ft=np.where(result["signals"]==1)
print(ft)

# Plot result
y=fd.copy()
pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)

#pylab.plot(np.arange(1, len(y)+1),
#           result["avgFilter"], color="cyan", lw=2)

#pylab.plot(np.arange(1, len(y)+1),
#           result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

#pylab.plot(np.arange(1, len(y)+1),
#           result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

#pylab.plot(np.arange(1, len(y)+1),
#           threshold * result["stdFilter"], color="green", lw=2)

pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.ylim(-1.5, 1.5)
pylab.show()

print(freq[108])
print(freq[162])
print(freq[169])
print(freq[545])
print(freq[int(len(fd)/2)-1])


peaks, _ = find_peaks(fd, distance=20)
print(peaks)
peaks2, _ = find_peaks(fd, prominence=1)      # BEST!
peaks3, _ = find_peaks(fd, width=20)
peaks4, _ = find_peaks(fd, threshold=0.4)     # Required vertical distance to its direct neighbouring samples, pretty useless
plt.subplot(2, 2, 1)
plt.plot(peaks, fd[peaks], "xr"); plt.plot(fd); plt.legend(['distance'])
plt.subplot(2, 2, 2)
plt.plot(peaks2, fd[peaks2], "ob"); plt.plot(fd); plt.legend(['prominence'])
plt.subplot(2, 2, 3)
plt.plot(peaks3, fd[peaks3], "vg"); plt.plot(fd); plt.legend(['width'])
plt.subplot(2, 2, 4)
plt.plot(indexes, fd[indexes], "xk"); plt.plot(fd); plt.legend(['threshold'])
plt.show()

#print(peaks)















