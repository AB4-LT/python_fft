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


FD = 3200
G_SCALE_FACTOR = 0.004
G_MM_S2 = 9.80665
MM_IN_METER = 1000
USE_POINTS = 1*4096
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

def plot_signal_and_fft(python_freqs, adxl_signal, python_fft, velocity, label_name):
    START_POS = 1
    plt.subplot(211)
    plt.plot(adxl_signal)
    plt.subplot(212)
    plt.plot(python_freqs[START_POS:], python_fft[START_POS:], label=label_name)
    plt.plot(python_freqs[START_POS:], deadzone_graph[START_POS:], color='green', label=velocity + " mm/s,   DEAD_ZONE = " + str(toFixed(DEAD_ZONE,5)))
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




input_file = "2021.04.22\\AB4-LT F88A5EA2A577_-1.12    [-4.11,   -3.38,   -1.12]    1288  lines 916.csv"
input_file = "2021.04.22\\AB4-LT F88A5EA2A9BA_2.39    [-4.28,   -4.83,   2.39]    1289  lines 373.csv"
input_file = "2021.04.22\\AB4-LT F88A5EA2A9E5_1.66    [3.89,   4.97,   1.66]    1283  lines 952.csv"
input_file = "2021.04.22\\AB4-LT F88A5EA2ABA5_-3.91    [-3.59,   -3.1,   -3.91]    1307  lines 1216.csv"  # датчик развёрнут на Y
input_file = "2021.04.22\\AB4-LT F88A5EA2A577_-3.04    [-8.52,   -9.82,   -3.04]    1291  lines 1953.csv"
input_file = "2021.04.22\\AB4-LT F88A5EA2A9BA_-3.41    [-7.14,   -11.3,   -3.41]    1291  lines 1707.csv"
input_file = "2021.04.22\\AB4-LT F88A5EA2A9BA_-1    [-1.35,   -2.87,   -1]    1291  lines 1094.csv"
input_file = "2021.04.22\\AB4-LT F88A5EA2A9E5_1.09    [3.3,   3.88,   1.09]    1283  lines 12.csv"
input_file = "2021.04.22\\AB4-LT F88A5EA2A9E5_1.3    [2.16,   2.51,   1.3]    1280  lines 90.csv"  # раскачивание по Y
input_file = "2021.04.22\\AB4-LT F88A5EA2A9BA_1.03    [-1.91,   1.61,   1.03]    1296  lines 169.csv"   # не прогрузились все точки


# подставить сюда нужный файл
# или см. дальше, где "модель сигнала вместо данных из файла"
input_file = "2021.04.22\\AB4-LT F88A5EA2A9E5_1.3    [2.16,   2.51,   1.3]    1280  lines 90.csv"  # раскачивание по Y
tune_coeff = 10000 
FD = 3200



file_descr = input_file

reader = csv.reader(open(input_file, 'r'),delimiter=';', quotechar=',')
input_points = []
input_xyz = []
for row in reader:
   k, v = row
   input_points.append([float(k.replace(',','.')), float(v.replace(',','.'))])
   input_xyz.append(float(v.replace(',','.')))
   

#модель сигнала вместо данных из файла
if 0 :
    #USE_POINTS = 4*2048   
    FD = 3200
    #CALC_START_POS = int(10*USE_POINTS/FD+1)
    input_points = []
    input_signal_hz = 50
    file_descr = " model " + str(input_signal_hz) +" Hz [" + str(USE_POINTS) + " points " + str(FD) + " Hz]"
    for i in range(USE_POINTS):
        model_signal = 0.1601766481718932/G_SCALE_FACTOR*sin(input_signal_hz*i/FD*2*pi+pi/2)
        model_signal += 0.03844239556125437/G_SCALE_FACTOR*sin(20*i/FD*2*pi)
        #if i > G_SCALE_FACTOR/20 : model_signal += 1/G_SCALE_FACTOR
        #model_signal += 0.08/G_SCALE_FACTOR*sin(77*i/FD*2*pi)
        input_points.append([float(i), float(model_signal)])



   
x = []
y = []
z = [] 
n = []
for i in range(USE_POINTS):
    n.append(input_points[i][0])
    x.append(input_points[i][1])
    y.append(input_points[i+USE_POINTS][1])
    z.append(input_points[i+2*USE_POINTS][1])

middle_x = np.mean(x)    
middle_y = np.mean(y)    
middle_z = np.mean(z)    
for i in range(USE_POINTS):
    x[i] -= middle_x
    y[i] -= middle_y
    z[i] -= middle_z

signal = []    
vector_len = []    
for i in range(USE_POINTS):
    signal.append( z[i] )
    vector_len.append( pow((pow(x[i],2) + pow(y[i],2) + pow(z[i],2)),0.5) )
    #signal.append( y[i] *  pow((pow(x[i],2) + pow(y[i],2) + pow(z[i],2)),0.5) / (1/G_SCALE_FACTOR))
    
#for i in range(USE_POINTS):
    #signal[i] = signal[i] * pow( pow(signal[i], 2) / pow(vector_len[i], 2), 0.5)
    #signal[i] = vector_len[i]



#window = np.hamming(len(signal))
#window = np.hanning(len(signal))
#signal = signal*window


# Filter requirements.
order = 6
fs = 3200.0       # sample rate, Hz
lowcut = 800.0
highcut = 1600.0
# Get the filter coefficients so we can check its frequency response. # b, a = butter_lowpass(cutoff, fs, order)
level = 2000
if 0 :
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
#plot_signal_and_fft(furie_freqs, signal, furie_norm_amplitudes, py_velocity, str(len(signal)) + " points from " + str(START_POINT_INDEX) + "  in  " + file_descr )


st='Входной сигнал   ' + file_descr
plt.plot(input_xyz,linewidth=2, label=st)
plt.legend(loc='best')
plt.show()


sys.exit()
