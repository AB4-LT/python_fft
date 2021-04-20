#https://habr.com/ru/post/451278/

import numpy as np
from pylab import *
from scipy import *
import scipy.io.wavfile as wavfile
import pywt


#Фурье
if 1:
    M=501
    hM1=int(np.floor((1+M)/2))
    hM2=int(np.floor(M/2))
    (fs,x)=wavfile.read('WebSDR.wav')
    x1=x[5000:5000+M]*np.hamming(M)
    N=511
    fftbuffer=np.zeros([N])
    fftbuffer[:hM1]=x1[hM2:]
    fftbuffer[N-hM2:]=x1[:hM2]
    X=fft(fftbuffer)
    mX=abs(X)
    pX=np.angle(X)
    suptitle("Анализ радиосигналаWebSDR")
    subplot(3, 1, 1)
    st='Входной сигнал (WebSDR.wav)'
    plot(x,linewidth=2, label=st)
    legend(loc='center')
    subplot(3, 1, 2)
    st=' Частотный спектр входного сигнала'
    plot(mX,linewidth=2, label=st)
    legend(loc='best')
    subplot(3, 1, 3)
    st=' Фазовый спектр входного сигнала'
    pX=np.unwrap(np.angle(X))
    plot(pX,linewidth=2, label=st)
    legend(loc='best') 
    show()
    
    
#Вейвлет
if 1:
    # Найти наибольшую мощность двух каналов, которая меньше или равна входу.
    def lepow2(x):    
        return int(2 ** floor(log2(x)))
    #Скалограмма с учетом дерева MRA.
    def scalogram(data):
        bottom = 0
        vmin = min(map(lambda x: min(abs(x)), data))
        vmax = max(map(lambda x: max(abs(x)), data))
        gca().set_autoscale_on(False)
        for row in range(0, len(data)):
            scale = 2.0 ** (row - len(data))
            imshow(
                array([abs(data[row])]),
                interpolation = 'nearest',
                vmin = vmin,
                vmax = vmax,
                extent = [0, 1, bottom, bottom + scale])
            bottom += scale
    # Загрузите сигнал, возьмите первый канал.
    rate, signal = wavfile.read('WebSDR.wav')
    signal = signal[0:lepow2(len(signal))]
    tree = pywt.wavedec(signal, 'coif5')
    gray()
    scalogram(tree)
    show()    
    
    