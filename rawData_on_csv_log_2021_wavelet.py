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
        
        


FD = 3200
G_SCALE_FACTOR = 0.004
G_MM_S2 = 9.80665
MM_IN_METER = 1000
USE_POINTS = 2*4096
START_POINT_INDEX = 0

DEAD_ZONE =2*G_SCALE_FACTOR
CALC_START_POS = 1
deadzone_graph =[]

py_velocity = .0

velocity_histoty =[]
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
    
def s16(value):
    return -(value & 0x8000) | (value & 0x7fff)

def get_xyz_textfile(textfile_name):
    xyz_points = []
    f = open(textfile_name, "r")
    for line in f:
        point = line.strip('\n').split(' ')
#        point = re.split(r" +",line.strip('\n'))
#        print(point)
        if len(point) >= 2:
            if point[0] == "[XYZ]" :
                xyz_points.append([int(point[1]), int(point[2]) ])
    f.close()
    return xyz_points

def get_fft_textfile(textfile_name):
    fft_points = []
    f = open(textfile_name, "r")
    for line in f:
        point = line.strip('\n').split(' ')
#        point = re.split(r" +",line.strip('\n'))
        if len(point) >= 2:
            if point[0] == "[FFT]" :
                fft_points.append([int(point[1]), int(point[2]) ])
    f.close()
    return fft_points

def get_descr_textfile(textfile_name):
    f = open(textfile_name, "r")
    data = f.readlines()
    return data[0]

def plot_signal_fft_and_history(adxl_signal, freqs, history, velocity, python_freqs, python_fft, label_name):
    START_POS = 1
    plt.subplot(311)
    plt.plot(adxl_signal)
    plt.subplot(312)
    plt.plot(python_freqs[START_POS:], python_fft[START_POS:], label=label_name)
    plt.plot(python_freqs[START_POS:], deadzone_graph[START_POS:], color='green', label=velocity + " mm/s,   DEAD_ZONE = " + str(toFixed(DEAD_ZONE,5)))
    plt.legend(loc='best')
    plt.subplot(313)
    plt.plot(freqs[START_POS:], history[START_POS:], label=velocity)
    plt.legend(loc='best')
    plt.show()

def plot_fft_and_history(adxl_signal, freqs, history, velocity, python_freqs, python_fft, label_name):
    START_POS = 1
    plt.subplot(211)
    plt.plot(python_freqs[START_POS:], python_fft[START_POS:], label=label_name)
    plt.plot(python_freqs[START_POS:], deadzone_graph[START_POS:], color='green', label=velocity + " mm/s,   DEAD_ZONE = " + str(toFixed(DEAD_ZONE,5)))
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(freqs[START_POS:], history[START_POS:], label=velocity)
    plt.legend(loc='best')
    plt.show()

def plot_signal_fft_and_fftvelocity(python_freqs, adxl_signal, python_fft, fft_velocity, velocity, label_name):
    START_POS = 1
    plt.subplot(311)
    plt.plot(adxl_signal)
    plt.subplot(312)
    plt.plot(python_freqs[START_POS:], python_fft[START_POS:], label=label_name)
    plt.plot(python_freqs[START_POS:], deadzone_graph[START_POS:], color='green', label=velocity + " mm/s,   DEAD_ZONE = " + str(toFixed(DEAD_ZONE,5)))
    plt.legend(loc='best')
    plt.subplot(313)
    plt.plot(python_freqs[START_POS:], fft_velocity[START_POS:], label=velocity)
    plt.legend(loc='best')
    plt.show()


def calc_python_velocity(freq_array, fft_array):
    global py_velocity
    global velocity_histoty
    global velocity_fft
    #print('python velocity')
    velocity = 0
    count = 0
    for i in range(CALC_START_POS, len(freq_array)):
        if (freq_array[i] > 9.5) & (freq_array[i] < 1000) & (fft_array[i] > DEAD_ZONE) :
            velocity_fft[i] =  (fft_array[i] * G_MM_S2 * MM_IN_METER) / (2*pi * freq_array[i]) #перевод из амплитуды в дискретах ускорения в СКЗ виброскорости на этой частоте
            count += 1
            velocity += pow( velocity_fft[i]/pow(2, 0.5), 2)
            print("[", i, "] ", toFixed(freq_array[i],2), " Hz   ", toFixed(fft_array[i],5), " g   ", toFixed(velocity_fft[i],3), " mm/s  ", toFixed(pow(velocity,0.5),3) , " mm/s", sep='')
        velocity_histoty[i] =  pow(velocity,0.5)

    velocity = pow(velocity,0.5)
    print('velocity', velocity, sep='   ')
    py_velocity = velocity



def save_in_point_to_c_file(input_points):
    f_out = open("in_array.c", "w")
    f_out.write("const int16_t z_in_array[" + str(len(input_points)) +"][2] = {\r")
    for i in range(len(input_points)):
        f_out.write('{ ' + str(int(input_points[i][0])) + ' , ' + str(int(input_points[i][1])) + ' },\r')
    f_out.write("};\r")
    f_out.close()


def save_signal_to_c_file(signal):
    f_out = open("in_array.c", "w")
    f_out.write("const int16_t z_in_array[" + str(len(signal)) +"][2] = {\r")
    for i in range(len(signal)):
        f_out.write('{ ' + str(i) + ' , ' + str(int(signal[i])) + ' },\r')
    f_out.write("};\r")
    f_out.close()



# указка в сборе, с магнитным щупом, 3200 Гц, двигатель №1
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_0.65.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_0.84.csv"
tune_coeff = 10300
FD = 3200


# указка в сборе, с магнитным щупом, 1600 Гц, двигатель №1
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1600_2.89.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1600_4.54.csv"
tune_coeff = 10300
FD = 1600


# указка в сборе, штатный щуп, 3200 Гц, двигатель №1
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_0.73.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1.07.csv"
tune_coeff = 10300
FD = 3200

# указка в сборе, штатный щуп, 1600 Гц, двигатель №1
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1600_2.13.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1600_1.99.csv"
tune_coeff = 10300
FD = 1600

# выносной магнитный щуп, 3200 Гц, двигатель №1
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_1.13.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_0.6.csv"
tune_coeff = 9970
FD = 3200

# выносной магнитный щуп, 1600 Гц, двигатель №1
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_1600_2.4.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_1600_2.4 (1).csv"
tune_coeff = 9970
FD = 1600


# выносной магнитный щуп, 3200 Гц, двигатель №2
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_12.39.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_9.24.csv"
tune_coeff = 9970
FD = 3200

# выносной магнитный щуп, 1600 Гц, двигатель №2
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_1600_16.09.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2A9E5_1600_20.6.csv"
tune_coeff = 9970
FD = 1600


# указка в сборе, с магнитным щупом, 3200 Гц, двигатель №2
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_5.28.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_4.19.csv"
tune_coeff = 10300
FD = 3200

# указка в сборе, с магнитным щупом, 1600 Гц, двигатель №2
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1600_9.26.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1600_15.4.csv"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_1600_13.59.csv"
tune_coeff = 10300
FD = 1600



input_file = "office_1\\AB4-LT F88A5EA2A7F7_1600_0_TASKYIELD.csv"
input_file = "office_1\\AB4-LT F88A5EA2A7F7_1600_FIFO-3.csv"
tune_coeff = 10000
FD = 1600

input_file = "office_1\\AB4-LT F88A5EA2A7F7_0_FIFO_8.csv"
input_file = "office_1\\AB4-LT F88A5EA2A7F7_0_FIFO_16.csv"
input_file = "office_1\\AB4-LT F88A5EA2A7F7_0_FIFO-3.csv"
input_file = "office_1\\AB4-LT F88A5EA2A7F7_0_TASKYIELD.csv"
tune_coeff = 10000
FD = 3200




input_file = "2021.04.05\\AB4-LT_7.94 _566Гц_10.csv"
input_file = "2021.04.05\\AB4-LT F88A5EA2A577_9.91_10Гц_10.csv"
input_file = "2021.04.05\\AB4-LT F88A5EA2A577_10.14_40Гц_10.csv"
input_file = "2021.04.05\\AB4-LT F88A5EA2A577_0.42_8Гц_10.csv"
input_file = "2021.04.05\\AB4-LT F88A5EA2A577_1.76_25Гц_2.csv"
input_file = "2021.04.05\\AB4-LT F88A5EA2A577_3.87_10Гц_4.csv"
tune_coeff = 10140 #1577
FD = 3200



input_file = "2021.04.07\\AB4-LT F88A5EA2A9E5_2.62.csv"
input_file = "2021.04.07\\AB4-LT F88A5EA2A9E5_4.92.csv"
input_file = "2021.04.07\\AB4-LT F88A5EA2A9E5_2.02.csv"
tune_coeff = 9970 #A9E5
FD = 3200

input_file = "2021.04.07\\AB4-LT F88A5EA2ABA5_2.78.csv"
input_file = "2021.04.07\\AB4-LT F88A5EA2ABA5_4.15.csv"
input_file = "2021.04.07\\AB4-LT F88A5EA2ABA5_13.4.csv"
tune_coeff = 10300 #BA5
FD = 3200



input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_2.41.csv"
input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_0.44.csv"
input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_0.csv"
input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_7.67.csv"
input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_0.12.csv"
input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_0.13.csv"
input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_1.36.csv"
input_file = "2021.04.08\\AB4-LT F88A5EA2A7F7_1.14.csv"
tune_coeff = 10300 #BA5
FD = 3200



#Северная ТЭЦ
#Двигатель №1
input_file = "2021.04.13\\AB4-LT F88A5EA2ABA5_11.57.csv"
input_file = "2021.04.13\\AB4-LT F88A5EA2ABA5_11.56.csv"
tune_coeff = 10300 #BA5
FD = 3200

#Двигатель №1
input_file = "2021.04.13\\AB4-LT F88A5EA2A9E5_7.52.csv"
input_file = "2021.04.13\\AB4-LT F88A5EA2A9E5_7.84.csv"
tune_coeff = 9970 #A9E5
FD = 3200

#Двигатель №2
input_file = "2021.04.13\\AB4-LT F88A5EA2A9E5_2.79.csv"
input_file = "2021.04.13\\AB4-LT F88A5EA2A9E5_2.78.csv"
tune_coeff = 9970 #A9E5
FD = 3200

#Двигатель №2
input_file = "2021.04.13\\AB4-LT F88A5EA2ABA5_4.72.csv"
input_file = "2021.04.13\\AB4-LT F88A5EA2ABA5_1.31.csv"
input_file = "2021.04.13\\AB4-LT F88A5EA2ABA5_1.73.csv"
tune_coeff = 10300 #BA5
FD = 3200




# подставить сюда нужный файл
# или см. дальше, где "модель сигнала вместо данных из файла"


input_file = "2021.04.05\\AB4-LT F88A5EA2A577_10.14_40Гц_10.csv"
input_file = "2021.04.13\\AB4-LT F88A5EA2ABA5_11.56.csv"
tune_coeff = 10140 #1577
FD = 3200




file_descr = input_file

reader = csv.reader(open(input_file, 'r'),delimiter=';', quotechar=',')
input_points = []
for row in reader:
   k, v = row
   i = float(k.replace(',','.')) 
   START_POINT_INDEX
   if (i >= START_POINT_INDEX) & ((i - START_POINT_INDEX) < USE_POINTS) :
       input_points.append([float(k.replace(',','.')), float(v.replace(',','.'))])


#модель сигнала вместо данных из файла
if 0 :
    #USE_POINTS = 4*2048   
    FD = 3200
    #CALC_START_POS = int(10*USE_POINTS/FD+1)
    input_points = []
    input_signal_hz = 600
    file_descr = " model " + str(input_signal_hz) +" Hz [" + str(USE_POINTS) + " points " + str(FD) + " Hz]"
    for i in range(USE_POINTS):
        model_signal = 2.4464590727776745/G_SCALE_FACTOR*sin(input_signal_hz*i/FD*2*pi+pi/2)
        #model_signal = 0.1601766481718932/G_SCALE_FACTOR*sin(input_signal_hz*i/FD*2*pi+pi/2)
        #model_signal += 0.03844239556125437/G_SCALE_FACTOR*sin(20*i/FD*2*pi)
        #if i > G_SCALE_FACTOR/20 : model_signal += 1/G_SCALE_FACTOR
        #model_signal += 0.08/G_SCALE_FACTOR*sin(77*i/FD*2*pi)
        input_points.append([float(i), float(model_signal)])



   

n=[input_points[i][0] for i in range(len(input_points))]
signal = [input_points[i][1] for i in range(len(input_points))]

#window = np.hamming(len(signal))
#window = np.hanning(len(signal))
#signal = signal*window


# Filter requirements.
order = 6
fs = 3200.0       # sample rate, Hz
lowcut = 10.0
highcut = 50.0
# Get the filter coefficients so we can check its frequency response. # b, a = butter_lowpass(cutoff, fs, order)
level = 2000
if 1 :
    signal -= np.mean(signal)
    for i in range(len(signal)):
        if signal[i] > level :
            signal[i] = level
        if signal[i] < -level :
            signal[i] = -level

if 0 :
    signal -= np.mean(signal)
    signal = butter_lowpass_filter(signal, highcut, fs, order)
    

if 0 :
    signal -= np.mean(signal)
    signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order)
    

save_signal_to_c_file(signal)


#FFT
if 1:
    spectrum = rfft(signal)
    furie_amplitudes = np_abs(spectrum)
    furie_norm_amplitudes = 2 * G_SCALE_FACTOR * furie_amplitudes / len(signal)
    furie_norm_amplitudes[0] /= 2



    #furie_freqs = rfftfreq(len(signal), 1. / FD)
    furie_freqs = []
    for i in range(len(furie_norm_amplitudes)) :
        furie_freqs.append( i*(((FD*10000)/tune_coeff)/2)/((len(furie_norm_amplitudes)) - 1) )

    for i in range(len(furie_freqs)) :
        velocity_histoty.append(0)
        velocity_fft.append(0)

    #DEAD_ZONE = statistics.median(furie_norm_amplitudes[3328:3840]) # 1300..1500 Hz (4096/3200)*freq  
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

    #plot_signal_fft_and_history(signal, furie_freqs, velocity_histoty, py_velocity, furie_freqs, furie_norm_amplitudes,  str(len(signal)) + " points from " + str(START_POINT_INDEX) + "  in  " + file_descr )
    #plot_fft_and_history(signal, furie_freqs, velocity_histoty, py_velocity, furie_freqs, furie_norm_amplitudes,  str(len(signal)) + " points from " + str(START_POINT_INDEX) + "  in  " + file_descr )
    #plot_signal_fft_and_fftvelocity(furie_freqs, signal, furie_norm_amplitudes, velocity_fft, py_velocity, str(len(signal)) + " points from " + str(START_POINT_INDEX) + "  in  " + file_descr )

if 0:
    reduced_signal=[]
    for i in range(0, 3200):
        reduced_signal.append(signal[i])
    signal = reduced_signal

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
    
    
#FFT по восстановленному сигналу
if 1:
    icoeffs = []
    for i in range(0, len(coeffs)-3):
        icoeffs.append(coeffs[i])
        
    isignal = pywt.waverec(icoeffs, 'db1') 
    signal = isignal
    
    spectrum = rfft(signal)
    furie_amplitudes = np_abs(spectrum)
    furie_norm_amplitudes = 2 * G_SCALE_FACTOR * furie_amplitudes / len(signal)
    furie_norm_amplitudes[0] /= 2



    #furie_freqs = rfftfreq(len(signal), 1. / FD)
    furie_freqs = []
    for i in range(len(furie_norm_amplitudes)) :
        furie_freqs.append( i*(((FD*10000)/tune_coeff)/2)/((len(furie_norm_amplitudes)) - 1) )

    for i in range(len(furie_freqs)) :
        velocity_histoty.append(0)
        velocity_fft.append(0)

    #DEAD_ZONE = statistics.median(furie_norm_amplitudes[3328:3840]) # 1300..1500 Hz (4096/3200)*freq  
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

    
    
gray()
scalogram(coeffs)
show()
    
sys.exit()
