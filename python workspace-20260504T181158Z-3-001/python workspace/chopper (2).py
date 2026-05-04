import libtiepie
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, butter, filtfilt
import os
import math
import sys
print(sys.path)
#------constants------#

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
DEFAULT_BACKGROUND_COLOR='#1a1a1a'

#------constants------#

#------global variables------#

chopperRadius = 42.74

osc = 0
sample_time = 1 #sample time in [s]
sample_rate = int(10e4) #sampling frequency [Hz]
record_length = sample_rate*sample_time
coupling= "DCV" #coupling type, can be ACV or DCV.
ch_range = [6,6] #oscilloscope range
resolution = 16
rawData = [[],[]]
powerData = [[],[]]
intensityData = [[],[]]

#------global variables------#

#------global variables------#

minPow = 0
maxPow = 0
rpm = 0
period = 0
centerTime = 0
oversample = 0

#------global variables------#


#------functions-----#
def connectTiePie():
    
    if 'scp' in globals():
        
        global scp
        
        scp.stop()
        
        return scp
        
    # Enable network search:
    libtiepie.network.auto_detect_enabled = True

    # Search for devices:
    libtiepie.device_list.update()

    # Try to open an oscilloscope with stream measurement support:
    scp = None
    for item in libtiepie.device_list:
        if item.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE):
            scp = item.open_oscilloscope()
            if scp.measure_modes & libtiepie.MM_STREAM:
                break
            else:
                scp = None
    
    # Try to open a generator:
    gen = None
    for item in libtiepie.device_list:
        if item.can_open(libtiepie.DEVICETYPE_GENERATOR):
            gen = item.open_generator()
            if gen:
                break
                
    return scp, gen

def oscParameters(scp,sample_rate,record_length,ch_range,coupling,resolution):
    
    assert resolution in [8,12,14,16], "resolution can be 8, 12, 14 or 16 bit"
    
    # Set measure mode:
    scp.measure_mode = libtiepie.MM_STREAM

    # Set sample rate:
    scp.sample_rate = sample_rate

    # Set record length:
    scp.record_length = record_length

    # For all channels:
    for i, ch in enumerate(scp.channels):
        # Enable channel to measure it:
        ch.enabled = True
        
        # Set range:
        ch.range = ch_range[i]

        # Set coupling:
        if coupling[i] == "ACV":
            ch.coupling = libtiepie.CK_ACV  # AC Volt
        elif coupling[i] == "DCV":
            ch.coupling = libtiepie.CK_DCV  # DC Volt
            
    scp.resolution = resolution

def normalizePerCent(min, max, t):
    return 100*(t-min)/(max-min)

def normalizePerCentData(data, min, max):
    for i,e in enumerate(data[1]):
        data[1][i] = normalizePerCent(min, max, e)
    return data

def normalizeIntegral(data):
    newData = [[],[]]
    total = riemSum(data)
    for i in range(0, len(data[0])):
        newData[0].append(data[0][i])
        newData[1].append(data[1][i]/total)
    return newData

def inInterval(a,b,x):
    return (a-x)*(b-x)<=0

def sumList(list):
    sum = 0
    for e in list:
        sum += e
    return sum
    
def riemSum(data):
    sum = 0
    for i in range(1,len(data[0])):
        sum += (data[0][i]-data[0][i-1])*mean([data[1][i],data[1][i-1]])
    print(sum)
    return sum

def gauss(x, a, b):
    y= a*np.exp(-1*b*x**2)
    return y

def interpolate(p1, p2, t):
    if(p2[0]-p1[0])==0:
        print('interpolation error')
        return False
    alpha = (p2[1]-p1[1])/(p2[0]-p1[0])
    return p1[1] + (t-p1[0])*alpha

def lowPass(data, sample_rate, cutoff, order):
    nyq = 0.5 *sample_rate
    normal_cutoff = cutoff/nyq
    b,a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b,a,data)
    return y

def measure():

    global osc
    global sample_rate
    global record_length
    global rawData
    if not osc:
        print("tiepie not connected!")
        return False
    osc.start()

    while not (osc.is_data_ready or osc.is_data_overflow):
        time.sleep(0.01)  # 10 ms delay, to save CPU time

    if osc.is_data_overflow:
        print('Data overflow!')

    # Get data:
    data = osc.get_data()

    # Stop stream:
    osc.stop()
    
    t = np.linspace(0,record_length/sample_rate,record_length)
    df_data = [t,data[0]]
    rawData = df_data
    return df_data

def measureBt():
    measure()
    updateData()

def mean(list):
    sum = 0
    for i in list:
        sum+=i
    return sum/len(list)

def calibrateMin():
    global minPow
    data = measure()
    minPow = min(data[1])
    print("new lower bond set to : %d"%minPow)
    return

def calibrateMax():
    global maxPow
    data = measure()
    maxPow = mean(data[1])
    print("new higher bond set to : %d"%maxPow)

def meanFilter(data, n):
    newData = [[],[]]
    print("len/n %d"%math.floor(len(data[0])/n))
    for i in range(0, math.floor(len(data[0])/n)):
        
        newData[0].append(mean(data[0][n*i:n*(i+1)]))
        newData[1].append(mean(data[1][n*i:n*(i+1)]))

    return newData


def powerToIntensityNum(data):
    newData = [[],[]]
    for i in range(1, len(data[1])):
        if (data[0][i]-data[0][i-1])==0:
            continue
        dP = (data[1][i]-data[1][i-1])/(data[0][i]-data[0][i-1])
        newData[0].append((data[0][i-1]+data[0][i])/2)
        newData[1].append(dP)
    return newData        

def translateList(list, t):
    for i in range(len(list)):
        list[i]+=t

def separateBlock(data):
    state = 0
    start = 0
    block = [[],[]]
    for i,e in enumerate(data[1]):
        if state:
            block[0].append(data[0][i])
            block[1].append(e)
            if e <= minPow or e >= maxPow:
                break
        if inInterval(minPow, maxPow, e) and not state:
            start = i
    translateList(block[0], -data[0][start])
    return block

def sortPoints(data):
    points = list(zip(data[0], data[1]))
    points.sort(key= lambda x:x[0])
    data[0], data[1] = zip(*points)
    data[0] = list(data[0])
    data[1] = list(data[1])
    return data

def invertPower(data1, target):

    for i,e in enumerate(data1):
        data1[i] = 2*target-e
    return data1

def filterInvalidPoints(data):
    i = 1
    while(i<len(data[0])):
        if (abs(data[0][i]-data[0][i-1])<=1e-19):
            data[0].pop(i-1)
            data[1][i] = mean([data[1].pop(i-1),data[1][i]])
        else:
            i+=1
    return data

def filterOversample(data):
    global oversample
    newData = [[],[]]
    temp = [[data[0][0]], [data[1][0]]]
    print("size of data: %d, oversample: %d"%(len(data[0]),oversample))
    for i in range(1, len(data[0])):
        if not i%oversample:
            newData[0].append(mean(temp[0]))
            newData[1].append(mean(temp[1]))
            temp = [[],[]]
        temp[0].append(data[0][i])
        temp[1].append(data[1][i])
    return newData


def getProcessedDataNew(data):
    global oversample
    global rpm
    global centerTime
    global label
    parity = 1
    blockList = []
    tempBlock = [[],[]]
    newData = [[],[]]
    tplot = []
    waist = 0
    target = (maxPow+minPow)/2
    last = -10
    finalPeriod = 0
    finalPeriodCounter = 0
    waistCounter = 0
    print("midPower: %.4f"%target)
    for e in data[0]:
        tplot.append(target)
    #plt.plot(data[0], tplot)
    plt.figure()
    for i,e in enumerate(data[1][1:]):
        
        if inInterval(data[1][i-1], data[1][i], target) and (i-last)>10:
            print("i-1: %.5f | e: %.5f | target: %.5f"%(data[1][i-1], data[1][i], target))
            #tempBlock[0].append(interpolate([data[1][i-1],data[0][i-1]], [data[1][i], data[0][i]], target))
            #tempBlock[1].append(target)
            plt.plot(tempBlock[0], tempBlock[1])
            blockList.append(tempBlock)
            tempBlock = [[],[]]
            last = i

        tempBlock[0].append(data[0][i])
        tempBlock[1].append(data[1][i])
    centerTime = blockList[1][0][0]
    plt.show()
    #plt.plot(data[0], tplot)

    plt.figure()
    for i,e in enumerate(blockList):
        global period
        if not i:
            continue
        period1 = blockList[i-1][0][-1]-blockList[i-1][0][0]
        period2 = e[0][-1] - e[0][0]
        tempCenter = blockList[i][0][0]
        blockList[i-1][0] =list(blockList[i-1][0][-math.floor(len(blockList[i-1][0])/2):]) + list(blockList[i][0][:math.floor(len(blockList[i][0])/2)])
        blockList[i-1][1] =list(blockList[i-1][1][-math.floor(len(blockList[i-1][1])/2):]) + list(blockList[i][1][:math.floor(len(blockList[i][1])/2)])
        period = mean([period1, period2])*8
        if i == 1:
            period = period2*8
            newData[0] = list(blockList[i-1][0])
            newData[1] = list(blockList[i-1][1])
        else:
            translateList(blockList[i-1][0], centerTime-tempCenter)
            if parity < 0:
                invertPower(blockList[i-1][1], target)
                
                plt.plot(blockList[i-1][0],blockList[i-1][1])
                newData[0] = list(newData[0])+list(blockList[i-1][0])
                newData[1] = list(newData[1])+list(blockList[i-1][1])
        parity *= -1
        finalPeriod+=period
        finalPeriodCounter+=1
        rpm = 60/period
        try:
            if i < len(blockList)-1 and i > 2:
                waist += measureWaistTime(list(blockList[i-1]))
                waistCounter+=1
        except:
            print(i-1)
            plt.plot(blockList[i-1][0],blockList[i-1][1])
            plt.show()
    plt.show()
    oversample = len(blockList)
    waist = timeToSpace(waist/waistCounter)
    label.config(text="mean waist: %.3f"%(waist))
    newData = sortPoints(newData)
    finalPeriod /= finalPeriodCounter
    print("final period counter: %d"%finalPeriodCounter)
    period = finalPeriod
    return newData


def getProcessedData(data):
    global centerTime
    newData = [[],[]]
    mod = period/4
    blockSize=mod*sample_rate
    for i in range(0,math.floor(record_length/(blockSize))):
        block = [ data[0][math.floor(i*blockSize):math.floor((i+1)*blockSize)], data[1][math.floor(i*blockSize):math.floor((i+1)*blockSize)] ]
        if not i:
            calibrateCenterTime(block)
        else:
            tempCenterTime = getCenterTime(block)
            translateList(block[0], centerTime-tempCenterTime)
        newData[0] = newData[0] + block[0]
        newData[1] = newData[1] + block[1]
    newData = sortPoints(newData)
    return newData
    #data[1] = savgol_filter(data[1], window_length=int(4/period), polyorder=1)
    
def getFirstBlock(data):
    mod = period/4
    return [data[0][:math.floor(mod*sample_rate)], data[1][:math.floor(mod*sample_rate)]]

def calibrateRpm():
    state = 0
    start = 0
    global rpm
    global period
    global rawData
    for i,e in enumerate(rawData[1]):
        if i==0: continue
        cond1 = (e-minPow)*(rawData[1][i-1]-minPow)
        cond2 = (e-maxPow)*(rawData[1][i-1]-maxPow)
        if cond1 <= 0 and e>rawData[1][i-1]:
            if state == 0:
                state = 1
                start = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], minPow)
            elif state == 1:
                end = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], minPow)
                period = (end-start)*4
                rpm = 60/period
                print("start: %.3f, end: %.3f, period: %.3f, rpm: %.3f"%(start, end, period, rpm))
                return
        if cond1 <= 0 and e<=rawData[1][i-1]:
            if state == 0:
                state = 2
                start = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], minPow)
            elif state == 2:
                end = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], minPow)
                period = (end-start)*4
                rpm = 60/period
                print("start: %.3f, end: %.3f, period: %.3f, rpm: %.3f"%(start, end, period, rpm))
                return
        if cond2 <=0 and e > rawData[1][i-1]:
            if state == 0:
                state = 3
                start = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], maxPow)
            elif state == 3:
                end = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], maxPow)
                period = (end-start)*4
                rpm = 60/period
                print("start: %.3f, end: %.3f, period: %.3f, rpm: %.3f"%(start, end, period, rpm))
                return
        if cond2 <=0 and e <= rawData[1][i-1]:
            if state == 0:
                state = 4
                start = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], maxPow)
            elif state == 4:
                end = interpolate([rawData[1][i-1],rawData[0][i-1]], [e,rawData[0][i]], maxPow)
                period = (end-start)*4
                rpm = 60/period
                print("start: %.3f, end: %.3f, period: %.3f, rpm: %.3f"%(start, end, period, rpm))
                return
    print("calibration error")

def measureWaistTime(processedPowerData):
    start = 0
    end = 0
    for i,e in enumerate(processedPowerData[1]):
        if not i:
            continue
        normalizedLast = normalizePerCent(minPow, maxPow, processedPowerData[1][i-1])
        normalizedAtual = normalizePerCent(minPow, maxPow, e)
        if inInterval(normalizedLast, normalizedAtual, 84.1):
            start = interpolate([normalizedLast, processedPowerData[0][i-1]], [normalizedAtual, processedPowerData[0][i]], 84.1)
        if inInterval(normalizedLast, normalizedAtual, 15.9):
            end = interpolate([normalizedLast, processedPowerData[0][i-1]], [normalizedAtual, processedPowerData[0][i]], 15.9)
        if start and end:
            return abs(end-start)


def calibrateCenterTime(powerProcessedData):
    global centerTime
    target = (maxPow+minPow)/2
    for i,e in enumerate(powerProcessedData[1]):
        if not i:
            continue
        if inInterval(powerProcessedData[1][i-1], e, target):
            centerTime = interpolate([powerProcessedData[1][i-1], powerProcessedData[0][i-1]], [e, powerProcessedData[0][i]], target)
            return centerTime
    print("center time not found")
    return False

def getCenterTime(block):
    target = (maxPow+minPow)/2
    for i,e in enumerate(block[1]):
        if not i:
            continue
        if inInterval(block[1][i-1], e, target):
            centerTime = interpolate([block[1][i-1], block[0][i-1]], [e, block[0][i]], target)
            return centerTime
    print("center time not found")
    return False

def timeToSpace(time):
    angularSpeed = 2*math.pi*rpm/60
    theta = angularSpeed*time
    return 2*chopperRadius*math.sin(theta/2)

def timeToAngle(time):
    angularSpeed = 2*math.pi*rpm/60
    return angularSpeed*time

def timeToSpaceData(data):
    newData = [[],[]]
    for i,e in enumerate(data[0]):
        if inInterval(centerTime-period/16, centerTime+period/16, e):
            x = chopperRadius*math.sin(timeToAngle(e-centerTime))
            newData[0].append(x)
            newData[1].append(data[1][i])
    #newData = meanFilter(newData, 10)
    newData[1] = lowPass(newData[1], sample_rate, 1e4,2)
    #newData[1] = savgol_filter(data[1], window_length=100, polyorder=3)
    newData = filterOversample(newData)
    return newData

def absList(data):
    newData = []
    for e in data:
        newData.append(abs(e))
    return newData


def plotPowerSpaceData():
    global powerData
    global period
    print("powerData: %d"%len(powerData[0]))
    pData = powerData
    print("center time: %.4f"%centerTime)
    print("period: %.5f"%period)
    pData = timeToSpaceData(pData)
    pData = normalizePerCentData(pData, minPow, maxPow)
    plt.figure()
    plt.plot(pData[0], pData[1])
    plt.xlabel("distance from power midpoint [mm]")
    plt.ylabel("Power [%]")
    plt.show()

def plotRawData():
    global rawData
    plt.figure()
    plt.plot(rawData[0], rawData[1])
    plt.xlabel("measurement time [s]")
    plt.ylabel("Power Detector Signal [V]")
    plt.show()
    return

def plotIntensity():
    global intensityData
    iData = intensityData
    iData = normalizeIntegral(iData)
    npDatax = np.asarray(iData[0])
    npDatay = np.asarray(iData[1])
    parameters, covariance = curve_fit(gauss, npDatax, npDatay) 
  
    fit_A = parameters[0] 
    fit_B = parameters[1]
    fit_y = gauss(npDatax, fit_A, fit_B)
    plt.figure()
    plt.plot(iData[0], iData[1])
    plt.plot(npDatax, fit_y)
    plt.xlabel("distance from power midpoint [mm]")
    plt.ylabel("Linear Intensity [%]")
    plt.show()


def connect():
    global osc
    osc = connectTiePie()[0]
    oscParameters(osc, sample_rate, record_length, ch_range, coupling, resolution)
    print("connected")

def updateData():
    global rawData
    global powerData
    global intensityData
    powerData = getProcessedDataNew(rawData)
    #plt.figure()
    #plt.plot(powerData[0], powerData[1])
    #plt.show()
    #powerData = filterInvalidPoints(powerData)
    #powerData[1] = lowPass(powerData[1], sample_rate, 8e3,2)
    
    #data[1] = lowPass(data[1], sample_rate, 5e2,2)
    #data[1] = savgol_filter(data[1], window_length=100, polyorder=3)
    #plotPowerSpaceData()
    #data = getFirstBlock(data)
    calibrateCenterTime(powerData)
    #waistTime = measureWaistTime(data)
    #print("waist time: %f"%waistTime)
    #print("waist: %f"%timeToSpace(waistTime))
    #-Intensity
    #intensityData =  normalizePerCentData(powerData, minPow, maxPow)
    #calibrateRpm()
    intensityData = timeToSpaceData(powerData)
    intensityData = powerToIntensityNum(intensityData)
    intensityData[1] = absList(intensityData[1])
    #intensityData[1] = savgol_filter(intensityData[1], window_length=max(int(record_length/100),4), polyorder=1)
    return True

def saveData():
    csv_file = open('chopperData.csv', 'w')
    global rawData
    data = rawData
    try:
        for i in range(len(data[0])):
            csv_file.write(str(i))
            for j in range(len(data)):
                csv_file.write(';' + str(data[j][i]))
            csv_file.write('\n')
        print('Data written to: ' + csv_file.name)

    finally:
        csv_file.close()

def saveCalibration():
    file = open('chopperCalibration.data', 'w')
    file.write("%f\n"%minPow)
    file.write("%f\n"%maxPow)
    file.write("%f\n"%rpm)
    file.close()

def loadCalibration():
    global minPow
    global maxPow
    global rpm
    file = open('chopperCalibration.data', 'r')
    minPow = float(file.readline())
    print(minPow)
    maxPow = float(file.readline())
    print(maxPow)
    rpm = float(file.readline())
    print(rpm)
    file.close()

def loadData():
    global rawData
    file = open('chopperData.csv', 'r')
    time = []
    power = []
    for line in file:
        info = line.split(';')
        info[2]=info[2].rstrip()
        time.append(float(info[1]))
        power.append(float(info[2]))
    rawData = [time, power]
    updateData()
    return rawData
#------functions-----#

#--------mainblock---------#
    
root = tk.Tk()
root.title('Optical Chopper Profiler')
root.geometry('%dx%d'%(WINDOW_WIDTH, WINDOW_HEIGHT))
root.configure(background=DEFAULT_BACKGROUND_COLOR)



#--connect tiepie--#

"""


#--connect tiepie--#

#--calibration--#

input("calibrate min")
calibrateMin()

input("calibrate max")
calibrateMax()




input("calibrate rpm")
data = measure()
"""
"""
saveData(data)

"""
#data[1] = lowPass(data[1], sample_rate, 1e3,2)

#--calibration--#

#--plots--#

#-Raw

#-Processed

"""
"""


label = tk.Label(root, text = "No data loaded")
label.place(x=int(6*WINDOW_WIDTH/16), y=int(16*WINDOW_HEIGHT/18))

rawPlotButton = tk.Button(root, text="Plot Raw Data", command=plotRawData)
rawPlotButton.place(x=int(8*WINDOW_WIDTH/16), y=int(4*WINDOW_HEIGHT/18))

powerPlotButton = tk.Button(root, text="Plot Power", command=plotPowerSpaceData)
powerPlotButton.place(x=int(8*WINDOW_WIDTH/16), y=int(6*WINDOW_HEIGHT/18))

intensityPlotButton = tk.Button(root, text="Plot Intensity", command=plotIntensity)
intensityPlotButton.place(x=int(8*WINDOW_WIDTH/16), y=int(8*WINDOW_HEIGHT/18))

measureButton = tk.Button(root, text="Measure", command=measureBt)
measureButton.place(x=int(5*WINDOW_WIDTH/16), y=int(4*WINDOW_HEIGHT/18))

calibrateMaxButton = tk.Button(root, text="Calibrate Max", command=calibrateMax)
calibrateMaxButton.place(x=int(5*WINDOW_WIDTH/16), y=int(6*WINDOW_HEIGHT/18))

calibrateMinButton = tk.Button(root, text="Calibrate Min", command = calibrateMin)
calibrateMinButton.place(x=int(5*WINDOW_WIDTH/16), y=int(8*WINDOW_HEIGHT/18))

saveCalibrationButton = tk.Button(root, text="Save Calibration", command= saveCalibration)
saveCalibrationButton.place(x=int(5*WINDOW_WIDTH/16), y=int(10*WINDOW_HEIGHT/18))

saveDataButton = tk.Button(root, text= "Save Data", command = saveData)
saveDataButton.place(x=int(5*WINDOW_WIDTH/16), y=int(12*WINDOW_HEIGHT/18))

loadCalibrationButton = tk.Button(root, text="Load Calibration", command= loadCalibration)
loadCalibrationButton.place(x=int(8*WINDOW_WIDTH/16), y=int(10*WINDOW_HEIGHT/18))

loadDataButton = tk.Button(root, text= "Load Data", command = loadData)
loadDataButton.place(x=int(8*WINDOW_WIDTH/16), y=int(12*WINDOW_HEIGHT/18))

connectButton = tk.Button(root, text="Connect TiePie", command=connect)
connectButton.place(x=int(8*WINDOW_WIDTH/16), y=int(14*WINDOW_HEIGHT/18))

calibrateRpmButton = tk.Button(root, text="Calibrate Rpm", command=calibrateRpm)
calibrateRpmButton.place(x=int(5*WINDOW_WIDTH/16), y=int(14*WINDOW_HEIGHT/18))



root.mainloop()
#--plots--#
