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

tune_coeff = 10000 #default

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

input_file = "AB4-LT F88A5EA2A7F7_3200_1.4_20Hz.csv"     #1  сигнал 20 Hz,sampling 3200
input_file = "AB4-LT F88A5EA2A7F7_1600_1.38_20Hz.csv"    #2  сигнал 20 Hz,sampling 1600
input_file = "AB4-LT F88A5EA2A7F7_3200_0_1590Hz.csv"     #3  сигнал 1590 Hz,sampling 3200 
input_file = "AB4-LT F88A5EA2A7F7_1600_35.47_1590Hz.csv" #4  сигнал 1590 Hz,sampling 1600
input_file = "AB4-LT F88A5EA2A7F7_3200_40.26_3190Hz.csv" #5  сигнал 3190 Hz,sampling 3200
input_file = "AB4-LT F88A5EA2A7F7_1600_39.09_3190Hz.csv" #6  сигнал 3190 Hz,sampling 1600
input_file = "AB4-LT F88A5EA2A7F7_3200_0_4790Hz.csv"     #7  сигнал 4790 Hz,sampling 3200
input_file = "AB4-LT F88A5EA2A7F7_1600_28.93_4790Hz.csv" #8  сигнал 4790 Hz,sampling 1600
input_file = "AB4-LT F88A5EA2A7F7_3200_4.09_6390Hz.csv"  #9  сигнал 6390 Hz,sampling 3200
input_file = "AB4-LT F88A5EA2A7F7_1600_4.07_6390Hz.csv"  #10 сигнал 6390 Hz,sampling 1600



# подставить сюда нужный файл
# или см. дальше, где "модель сигнала вместо данных из файла"
input_file = "AB4-LT F88A5EA2A7F7_1600_35.47_1590Hz.csv" #4  сигнал 1590 Hz,sampling 1600
FD = 1600





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
highcut = 200.0

if 0 :
    signal -= np.mean(signal)
    signal = butter_lowpass_filter(signal, highcut, fs, order)
    

if 1 :
    signal -= np.mean(signal)
    signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order)


    
#прореживание сигнала
if 0 :
    decimation_signal = []    
    decimation_coefficient = 8
    i = 0

    FD /= decimation_coefficient
    
    lowcut = 10.0
    highcut = FD/3
    fs = FD
    order = 6
    signal -= np.mean(signal)
    signal = butter_lowpass_filter(signal, highcut, fs, order)
    
    while i < len(signal):
        decimation_signal.append(signal[i])
        i += decimation_coefficient
    signal = decimation_signal

    #signal -= np.mean(signal)
    #signal = butter_lowpass_filter(signal, highcut, fs, order)



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
            
            
plt.subplot(2, 1, 1)
st='Входной сигнал   ' + file_descr
plt.plot(signal,linewidth=2, label=st)
plt.legend(loc='center')
plt.subplot(2, 1, 2)
st=' FFT   ' + str(py_velocity)
plt.plot(furie_freqs[1:], furie_norm_amplitudes[1:],linewidth=2, label=st)
plt.legend(loc='best')
plt.show()
    
