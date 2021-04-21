import sys
import re
from datetime import datetime
from datetime import timedelta
import time
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.dates import num2date
import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy import array, arange, abs as np_abs
from math import sin, pi, cos
import inspect
import csv
import pylab# модуль построения поверхности
from mpl_toolkits.mplot3d import Axes3D# модуль построения поверхности
from scipy.signal import butter, lfilter, freqz
import statistics
import pywt
from pywt import wavedec
from pylab import *
from scipy import *


FD = 3200
G_SCALE_FACTOR = 0.004
G_MM_S2 = 9.80665
MM_IN_METER = 1000

DEAD_ZONE =2*G_SCALE_FACTOR
CALC_START_POS = 1
deadzone_graph =[]

py_velocity = .0

velocity_fft =[]


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    print('b', b, sep = '   ')
    print('a', a, sep = '   ')
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    print('b', b, sep = '   ')
    print('a', a, sep = '   ')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

    
def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"
    

def calc_python_velocity(freq_array, fft_array):
    global py_velocity
    global velocity_fft
    #print('python velocity')
    velocity = 0
    for i in range(CALC_START_POS, len(freq_array)):
        if (freq_array[i] > 9.5) & (freq_array[i] < 1000) & (fft_array[i] > DEAD_ZONE) :
            velocity_fft[i] =  (fft_array[i] * G_MM_S2 * MM_IN_METER) / (2*pi * freq_array[i]) #перевод из амплитуды в дискретах ускорения в СКЗ виброскорости на этой частоте
            velocity += pow( velocity_fft[i]/pow(2, 0.5), 2)
            print("[", i, "] ", toFixed(freq_array[i],2), " Hz   ", toFixed(fft_array[i],5), " g   ", toFixed(velocity_fft[i],3), " mm/s  ", toFixed(pow(velocity,0.5),3) , " mm/s", sep='')

    velocity = pow(velocity,0.5)
    print('velocity', velocity, sep='   ')
    py_velocity = velocity





# варианты сигналов

#1 сигнал со стенда, заданная виброскорость 1 мм/с
input_file = "AB4-LT F88A5EA2A577_0.97_80Гц_1.csv"
tune_coeff = 10140 #1577

#2 сигнал со стенда, заданная виброскорость 10 мм/с
input_file = "AB4-LT F88A5EA2A577_9.91_10Гц_10.csv"
tune_coeff = 10140 #1577

#3 сигнал со стенда, заданная виброскорость 10 мм/с
input_file = "AB4-LT F88A5EA2A577_9.94_160Гц_10.csv"
tune_coeff = 10140 #1577

#4 сигнал с двигателя, результат правильный
input_file = "AB4-LT F88A5EA2ABA5_3.12.csv"
tune_coeff = 10300 #ABA5

#5 сигнал с двигателя, сильно завышен результат
input_file = "AB4-LT F88A5EA2ABA5_10.18.csv"
tune_coeff = 10300 #ABA5

#6 сигнал с двигателя, сильно завышен результат
input_file = "AB4-LT F88A5EA2ABA5_14.22.csv"
tune_coeff = 10300 #ABA5




# подставить сюда нужный файл
# или см. дальше, где "модель сигнала вместо данных из файла"

#4 сигнал с двигателя, результат правильный
input_file = "AB4-LT F88A5EA2ABA5_3.12.csv"
tune_coeff = 10300 #ABA5





file_descr = input_file

reader = csv.reader(open(input_file, 'r'),delimiter=';', quotechar=',')
input_points = []
for row in reader:
   k, v = row
   input_points.append([float(k.replace(',','.')), float(v.replace(',','.'))])


#модель сигнала вместо данных из файла
if 0 :
    #USE_POINTS = 4*2048   
    FD = 3200
    #CALC_START_POS = int(10*USE_POINTS/FD+1)
    input_points = []
    input_signal_hz = 400
    file_descr = " model " + str(input_signal_hz) +" Hz [" + str(USE_POINTS) + " points " + str(FD) + " Hz]"
    for i in range(USE_POINTS):
        model_signal = 2.4464590727776745/G_SCALE_FACTOR*sin(input_signal_hz*i/FD*2*pi+pi/2)
        input_points.append([float(i), float(model_signal)])


n=[input_points[i][0] for i in range(len(input_points))]
signal = [input_points[i][1] for i in range(len(input_points))]

#window = np.hamming(len(signal))
#window = np.hanning(len(signal))
#signal = signal*window


# Filter requirements.
order = 6
fs = FD       # sample rate, Hz
lowcut = 10.0
highcut = 1000.0

if 1 :
    signal -= np.mean(signal)
    signal = butter_lowpass_filter(signal, highcut, fs, order)
    

if 0 :
    signal -= np.mean(signal)
    signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order)
    

#FFT
if 1:
    spectrum = rfft(signal)
    furie_amplitudes = np_abs(spectrum)
    furie_norm_amplitudes = 2 * G_SCALE_FACTOR * furie_amplitudes / len(signal)
    furie_norm_amplitudes[0] /= 2

    furie_freqs = []
    for i in range(len(furie_norm_amplitudes)) :
        furie_freqs.append( i*(((FD*10000)/tune_coeff)/2)/((len(furie_norm_amplitudes)) - 1) )

    for i in range(len(furie_freqs)) :
        velocity_fft.append(0)

    DEAD_ZONE = 1*G_SCALE_FACTOR

    print(file_descr)

    calc_python_velocity(furie_freqs, furie_norm_amplitudes)
    print("DEAD_ZONE", DEAD_ZONE, sep='   ')

    py_velocity = toFixed(py_velocity,3)

    for i in range(len(furie_freqs)):
        if i < CALC_START_POS:
            deadzone_graph.append(0)
        else: 
            deadzone_graph.append(float(DEAD_ZONE))

#wavelet
if 1:
    print('\r\n\r\n\r\n\r\n')
    #print(pywt.wavelist())
    #['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'haar', 'mexh', 'morl', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'shan', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
    wavelet_level = 8
    coeffs = wavedec(signal, 'db1', level=wavelet_level)
    
    rms_array = []
    rms_g_array = []
    rms_velocity_array = []
    wavelet_signal = []
    #cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    real_sample_freq = (FD*10000)/tune_coeff
    print('real sample freq   ' + str(toFixed(real_sample_freq,3)))
    for i in range(1,len(coeffs)-1):
        rms = np.sqrt(np.mean(coeffs[i]**2))
        rms_g = np.sqrt(np.mean((G_SCALE_FACTOR*coeffs[i])**2))
        right_freq =  (real_sample_freq/(2**(wavelet_level-i)))
        left_freq =  (real_sample_freq/(2**((wavelet_level-i)+1)))
        middle_freq = left_freq + (right_freq - left_freq)/2
        rms_velocity = (rms_g* G_MM_S2 * MM_IN_METER) /  (2*pi*right_freq)
        print('freq  [' + str(toFixed(left_freq,2)) + '..' + str(toFixed(middle_freq,2)) + '..' + str(toFixed(right_freq,2)) + ']    len(' + str(len(coeffs[i])) + ')   ' + str(toFixed(rms,5)) + '   ' + str(toFixed(rms_g,5)) + ' g   ' + str(toFixed(rms_velocity,3)) + '   mm/s')
        rms_array.append(rms)
        rms_g_array.append(rms_g)
        rms_velocity_array.append(rms_velocity)
        
        for j in range(len(coeffs[i])):
            wavelet_signal.append(coeffs[i][j])

    summ_acc = 0 
    for i in range(len(rms_g_array)):
        summ_acc += pow(rms_g_array[i],2)
    rms_result = toFixed(pow(summ_acc, 0.5), 3)
    print('summ_acc   ' + str(summ_acc))
        
    rms_result = 0 
    for i in range(len(rms_velocity_array)):
        rms_result += pow(rms_velocity_array[i],2)
    rms_result = toFixed(pow(rms_result, 0.5), 3)
    print('rms_result   ' + str(rms_result))
        
    plt.subplot(3, 1, 1)
    st='Входной сигнал   ' + file_descr
    plt.plot(signal,linewidth=2, label=st)
    plt.legend(loc='center')
    plt.subplot(3, 1, 2)
    st=' FFT   ' + str(py_velocity)
    plt.plot(furie_freqs, furie_norm_amplitudes,linewidth=2, label=st)
    plt.legend(loc='best')
    plt.subplot(3, 1, 3)
    st='DWT(' + str(wavelet_level) + ')    [' + str(len(signal)) + ']    ' + str(rms_result)
    plt.plot(wavelet_signal,linewidth=2, label=st)
    plt.legend(loc='best') 
    plt.show()
    
