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

FD = 3200
G_SCALE_FACTOR = 0.004
G_MM_S2 = 9.80665
MM_IN_METER = 1000
USE_POINTS = 4096
START_POINT_INDEX = 3000

DEAD_ZONE =2*G_SCALE_FACTOR
CALC_START_POS = 13
deadzone_graph =[]

py_velocity = .0

velosity_histoty =[]

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


def calc_python_velocity(freq_array, fft_array):
    global py_velocity
    global velosity_histoty
    print('python velocity')
    velocity = 0
    count = 0
    for i in range(CALC_START_POS, len(freq_array)):
    #for i in range(CALC_START_POS, 1280):
    #for i in range(CALC_START_POS, 512):
        if fft_array[i] > DEAD_ZONE:
            velocity += pow(fft_array[i] / freq_array[i], 2)
            count += 1
            print(i, freq_array[i], fft_array[i], velocity, sep='   ')
        velosity_histoty[i] =  velocity

    velocity = pow(velocity * pow( G_MM_S2 * MM_IN_METER, 2) / (4 * pow(pi, 2)), 0.5) / pow(2, 0.5)

    for i in range(CALC_START_POS, len(freq_array)):
        velosity_histoty[i] = pow(velosity_histoty[i]* pow( G_MM_S2 * MM_IN_METER, 2) / (4 * pow(pi, 2)), 0.5) / pow(2, 0.5)

    print('velocity_xyz_points', count,sep='   ')
    print('velocity', velocity, sep='   ')

    py_velocity = velocity



def save_in_point_to_c_file(input_points):
    f_out = open("in_array.c", "w")
    f_out.write("int16_t z_in_array[4096][2] = {\r")
    for i in range(len(input_points)):
        f_out.write('{ ' + str(int(input_points[i][0])) + ' , ' + str(int(input_points[i][1])) + ' },\r')
    f_out.write("};\r")
    f_out.close()



input_file = "AB4-LT F88A5EA2A9E5_0.51.csv"
tune_coeff = 10000

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
   
   
print(input_points)
save_in_point_to_c_file(input_points)


n=[input_points[i][0] for i in range(len(input_points))]
signal = [input_points[i][1] for i in range(len(input_points))]
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

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(n,signal,0,label='accel z')
ax.legend(loc='best')
ax.grid(True)
#plt.show()

py_velocity = toFixed(py_velocity,3)

for i in range(len(furie_freqs)):
    if i < CALC_START_POS:
        deadzone_graph.append(0)
    else: 
        deadzone_graph.append(float(DEAD_ZONE))

plot_signal_fft_and_history(signal, furie_freqs, velosity_histoty, py_velocity, furie_freqs, furie_norm_amplitudes,  str(len(signal)) + " points from " + str(START_POINT_INDEX) + "  in  " + file_descr )
