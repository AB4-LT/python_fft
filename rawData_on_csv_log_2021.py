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

FD = 3200
G_SCALE_FACTOR = 0.004
G_MM_S2 = 9.80665
MM_IN_METER = 1000
USE_POINTS = 2*4096
START_POINT_INDEX = 0

DEAD_ZONE =2*G_SCALE_FACTOR
CALC_START_POS = 13
deadzone_graph =[]

py_velocity = .0

velosity_histoty =[]

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
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
    plt.plot(python_freqs[START_POS:], deadzone_graph[START_POS:], color='green', label=velocity + " mm/s")
    plt.legend(loc='best')
    plt.subplot(313)
    plt.plot(freqs[START_POS:], history[START_POS:], label=velocity)
    plt.legend(loc='best')
    plt.show()

def plot_fft_and_history(adxl_signal, freqs, history, velocity, python_freqs, python_fft, label_name):
    START_POS = 1
    plt.subplot(211)
    plt.plot(python_freqs[START_POS:], python_fft[START_POS:], label=label_name)
    plt.plot(python_freqs[START_POS:], deadzone_graph[START_POS:], color='green', label=velocity + " mm/s")
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(freqs[START_POS:], history[START_POS:], label=velocity)
    plt.legend(loc='best')
    plt.show()


def calc_python_velocity(freq_array, fft_array):
    global py_velocity
    global velosity_histoty
    #print('python velocity')
    velocity = 0
    count = 0
    for i in range(CALC_START_POS, len(freq_array)):
    #for i in range(CALC_START_POS, 1280):
    #for i in range(CALC_START_POS, 512):
        if fft_array[i] > DEAD_ZONE:
            velocity += pow(fft_array[i] / freq_array[i], 2)
            count += 1
            print("[", i, "] ", toFixed(freq_array[i],2), " Hz   ", toFixed(fft_array[i],5), " g   ", toFixed(pow(velocity * pow( G_MM_S2 * MM_IN_METER, 2) / (4 * pow(pi, 2)), 0.5) / pow(2, 0.5),3), " mm/s", sep='')
        velosity_histoty[i] =  velocity

    velocity = pow(velocity * pow( G_MM_S2 * MM_IN_METER, 2) / (4 * pow(pi, 2)), 0.5) / pow(2, 0.5)

    for i in range(CALC_START_POS, len(freq_array)):
        velosity_histoty[i] = pow(velosity_histoty[i]* pow( G_MM_S2 * MM_IN_METER, 2) / (4 * pow(pi, 2)), 0.5) / pow(2, 0.5)

    #print('velocity_xyz_points', count,sep='   ')
    print('velocity', velocity, sep='   ')

    py_velocity = velocity



def save_in_point_to_c_file(input_points):
    f_out = open("in_array.c", "w")
    f_out.write("int16_t z_in_array[4096][2] = {\r")
    for i in range(len(input_points)):
        f_out.write('{ ' + str(int(input_points[i][0])) + ' , ' + str(int(input_points[i][1])) + ' },\r')
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


# подставить сюда нужный файл
# или см. дальше, где "модель сигнала вместо данных из файла"
input_file = "2021.04.02\\AB4-LT F88A5EA2ABA5_0.84.csv"
tune_coeff = 10300
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
   #else :
       #input_points.append([float(k.replace(',','.')), 0.0])
   

#модель сигнала вместо данных из файла
if 1 :
    USE_POINTS = 4*2048   
    FD = 3200
    CALC_START_POS = int(10*USE_POINTS/FD+1)
    input_points = []
    input_signal_hz = 9
    file_descr = " model " + str(input_signal_hz) +" Hz [" + str(USE_POINTS) + " points " + str(FD) + " Hz]"
    for i in range(USE_POINTS):
        model_signal = 1/G_SCALE_FACTOR*sin(input_signal_hz*i/FD*2*pi)
        #model_signal += 0.5/G_SCALE_FACTOR*sin(100*i/FD*2*pi)
        #model_signal += 0.08/G_SCALE_FACTOR*sin(77*i/FD*2*pi)
        input_points.append([float(i), float(model_signal)])



   
#print(input_points)
save_in_point_to_c_file(input_points)


n=[input_points[i][0] for i in range(len(input_points))]
signal = [input_points[i][1] for i in range(len(input_points))]
#window = np.hamming(len(signal))
window = np.hanning(len(signal))
signal = signal*window


# Filter requirements.
order = 6
fs = 3200.0       # sample rate, Hz
cutoff = 600      # desired cutoff frequency of the filter, Hz
# Get the filter coefficients so we can check its frequency response. # b, a = butter_lowpass(cutoff, fs, order)

#signal = butter_lowpass_filter(signal, cutoff, fs, order)

#print(signal)

spectrum = rfft(signal)
furie_amplitudes = np_abs(spectrum)
furie_norm_amplitudes = 2 * G_SCALE_FACTOR * furie_amplitudes / len(signal)
furie_norm_amplitudes[0] /= 2



#furie_freqs = rfftfreq(len(signal), 1. / FD)
furie_freqs = []
for i in range(len(furie_norm_amplitudes)) :
    furie_freqs.append( i*(((FD*10000)/tune_coeff)/2)/((len(furie_norm_amplitudes)) - 1) )

for i in range(len(furie_freqs)) :
    velosity_histoty.append(0)

calc_python_velocity(furie_freqs, furie_norm_amplitudes)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(n,signal,0,label='accel z')
#ax.legend(loc='best')
#ax.grid(True)
#plt.show()

py_velocity = toFixed(py_velocity,3)

for i in range(len(furie_freqs)):
    if i < CALC_START_POS:
        deadzone_graph.append(0)
    else: 
        deadzone_graph.append(float(DEAD_ZONE))

plot_signal_fft_and_history(signal, furie_freqs, velosity_histoty, py_velocity, furie_freqs, furie_norm_amplitudes,  str(len(signal)) + " points from " + str(START_POINT_INDEX) + "  in  " + file_descr )
#plot_fft_and_history(signal, furie_freqs, velosity_histoty, py_velocity, furie_freqs, furie_norm_amplitudes,  str(len(signal)) + " points from " + str(START_POINT_INDEX) + "  in  " + file_descr )


sys.exit()



WINDOW_LEN = 4096
windowed_graph = []
ADDED_LABEL = ""
for window_shift in range(len(input_points) - WINDOW_LEN) :
    print('window_shift', window_shift, sep='   ')
    window_points = []
    for i in range(WINDOW_LEN):
        window_points.append(float(input_points[i+window_shift][1]))

    #hann_window = np.hamming(len(window_points))
    #window_points = window_points*hann_window
    #ADDED_LABEL = "+ HANN"

    order = 6
    fs = 3200.0       # sample rate, Hz
    cutoff = 600      # desired cutoff frequency of the filter, Hz
    window_points = butter_lowpass_filter(window_points, cutoff, fs, order)
    ADDED_LABEL = "+ LOWPASS FILTER"
        
    spectrum = rfft(window_points)
    furie_amplitudes = np_abs(spectrum)
    furie_norm_amplitudes = 2 * G_SCALE_FACTOR * furie_amplitudes / len(window_points)
    furie_norm_amplitudes[0] /= 2
    
    furie_freqs = []
    for i in range(len(furie_norm_amplitudes)) :
        furie_freqs.append( i*(((FD*10000)/tune_coeff)/2)/((len(furie_norm_amplitudes)) - 1) )
        
    velosity_histoty = []
    for i in range(len(window_points)) :
        velosity_histoty.append(0)

    calc_python_velocity(furie_freqs, furie_norm_amplitudes)
    
    windowed_graph.append(float(py_velocity))

plt.plot(range(len(input_points) - WINDOW_LEN), windowed_graph, label= input_file + "   window[" + str(WINDOW_LEN) + "]" + ADDED_LABEL)
plt.legend(loc='best')
plt.show()
