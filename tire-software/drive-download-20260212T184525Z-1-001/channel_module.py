"""

Ver.: 0.1

"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class LowPassFilter:
    def __init__(self, sample_rate, cutoff_frequency, order):
        self.sample_rate = sample_rate
        self.cutoff_frequency = cutoff_frequency
        self.order = order
        return

    def filter(self, data):
        nyq = 0.5 *self.sample_rate
        normal_cutoff = self.cutoff_frequency/nyq
        b,a = butter(self.order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b,a,data)
        return y

class Channel:

    def __init__(self, name, 
                 data=[], color="FF0000", 
                 visibility = 0, filtered = 0, 
                 plotingType = "line", scaleble = True,
                 label = None,
                 xdata = None,
                 visibilityCheckbutton = None,
                 filteredCheckbutton = None,
                 filter = LowPassFilter(100, 2, 2)):
        self.name = name
        self.data = data
        self.color = color
        self.visibility = tk.IntVar(value = visibility)
        self.filtered = tk.IntVar(value = filtered)
        self.plotingType = plotingType
        self.scaleble = scaleble
        self.filter = filter
        self.visibilityCheckbutton = visibilityCheckbutton
        self.filteredCheckbutton = filteredCheckbutton
        self.xdata = xdata
        if label is None:
            label = name
        self.label = label
        return

    def setFilter(self, filter):
        self.filter = filter
        return
    def setXData(self, xdata):
        self.xdata = xdata
        return

    def setScaleble(self, scaleble):
        self.scaleble = scaleble
        return
    
    def setColor(self, newColor):
        self.color = newColor
        return
    
    def getName(self):
        return self.name

    def getVisibility(self):
        return bool(self.visibility.get())
    
    def getFiltered(self):
        return bool(self.filtered.get())

    def setData(self, newData):
        self.data = newData
        return

    def setButtons(self, root, x, y, visibilityButton = None, filteredButton = None, text = None):
        if text is None:
            text = self.name

        if visibilityButton is None:
            visibilityButton = ttk.Checkbutton(root, text = text, variable = self.visibility)

        self.visibilityCheckbutton = visibilityButton

        if filteredButton is None:
            filteredButton = ttk.Checkbutton(root, variable = self.filtered)

        self.visibilityCheckbutton = visibilityButton
        
        visibilityButton.place(x = x, y = y)
        filteredButton.place(x = x-22, y = y)
        return

    def plot(self, axis, xdata = None, parity = False):
        if xdata is None:
            xdata = self.xdata
        if not self.visibility.get():
            return parity
        if (len(axis.lines) or len(axis.collections)) and self.scaleble:
            axis = axis.twinx()
        data = self.data
        if self.filtered.get():
            data = self.filter.filter(self.data)
        if self.plotingType == "scatter":
            axis.scatter(xdata, data, color = self.color, label = self.name, linewidth = 1, s = 1)
        else:
            axis.plot(xdata, data, color = self.color, label = self.name, linewidth = 1)
        axis.tick_params(axis = "y", labelcolor = self.color)
        axis.axhline(y=0, color = 'black', linewidth=0.5)
        axis.axvline(x=0, color = 'black', linewidth=0.5)
        if parity:
            axis.yaxis.tick_right()
        else:
            axis.yaxis.tick_left()

        return not parity