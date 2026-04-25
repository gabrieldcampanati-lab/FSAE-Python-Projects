"""

Ver.: 0.5

"""

import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import ttk
import os
import master_builder as mb
import glob
from scipy.signal import butter, filtfilt
import index_tables as it
import channel_module as cm
import magic_formula as mf


WIDTH = 720
HEIGHT = 480



def lowPass(data, sample_rate, cutoff, order):
    nyq = 0.5 *sample_rate
    normal_cutoff = cutoff/nyq
    b,a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b,a,data)
    return y

def getBlockIndexOld():
    constant = 37
    p = it.pressureIndex[selections["pressure"].get()]
    ia = it.inclinationAngleIndex[selections["inclinationAngle"].get()]
    fz = it.fzIndex[selections["fz"].get()]
    return constant + 16*p + 5*ia + fz

def getBlockIndexFromString(string):
    string = string.split(":")[0]
    return int(string)

def getImmediateSubfolders(directory_path):
    """
    Returns a list of names of immediate subfolders within the specified directory.
    """
    subfolders = [
        name for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    ]
    return subfolders

def getRunFiles(path):
    pattern = os.path.join(path, f"*{".dat"}")
    return [os.path.basename(file) for file in glob.glob(pattern)]

def updateRunFileOptions(tireSelection):
    global runOptions
    global runSelectionBox
    runOptions = getRunFiles(".data_files/tires/%s/run"%tireSelection)
    runSelectionBox['menu'].delete(0, 'end')
    # Add new options
    for option in runOptions:
        runSelectionBox['menu'].add_command(label=option, command=tk._setit(selections["run"], option))
    # Set a new default selected option
    selections["run"].set(runOptions[0])
    return

def getBlocksNames(blocks):
    names = []
    if blocks.empty:
        return names
    for index, block in blocks.iterrows():
        name = "%d: V:%d file:%s"%(block["block_id"], block["V"], block["run_id"])
        names.append(name)
    return names

def updateBlockOptions(na):
    global blockOptions
    global blockSelectionBox
    blockGuide = loadData(".data_files/tires/%s/run/blockGuide.csv"%selections["tire"].get())
    blocks = blockGuide[
        (blockGuide["FZ"] == it.fzSelectionValues[selections["fz"].get()]) &
        (blockGuide["IA"] == it.inclinationAngleSelectionValues[selections["inclinationAngle"].get()]) &
        (blockGuide["P"] == it.pressureSelectionValues[selections["pressure"].get()])
        ]
    print(blocks)
    blockOptions = getBlocksNames(blocks)
    blockSelectionBox['menu'].delete(0, 'end')
    # Add new options
    for option in blockOptions:
        blockSelectionBox['menu'].add_command(label=option, command=tk._setit(selections["block"], option))
    # Set a new default selected option
    selections["block"].set(blockOptions[0])

    return

def loadRunData(path):
    data = pd.read_csv(path, sep = '\t', skiprows=[0,2], header = 0)
    return data

def loadData(path):
    data = pd.read_csv(path, sep = '\t', header = 0)
    return data

def plotChannel(channel, ax, totalCounter, confirmedCounter, data, ploted):
    
    if not channels[channel].get():
        return (ploted,0)
    
    if ploted:
        ax = ax.twinx()

    if(selections["lowpass"].get()):
        data[channel] = lowPass(data[channel], 1e2, 2, 2)

    ax.plot(data["ET"], data[channel], color = "C%d"%totalCounter, label = channel)
    ax.tick_params(axis = "y", labelcolor = "C%d"%totalCounter)
    if not confirmedCounter%2:
        ax.yaxis.tick_left()

    return (True, 1)

def plotData(data, title="graph"):

    fig, ax = plt.subplots(num=title)

    parity = False
    for channel in channels.values():
        channel.setData(data[channel.getName()])
        parity = channel.plot(ax, xdata = data["ET"], parity = parity)

    ax.set_xlabel("ET")
    ax.set_title(title)
    fig.legend()
    fig.show()
    return

def plotRunGraph():
    data = loadRunData(".data_files/tires/%s/run/%s"%(selections["tire"].get(), selections["run"].get()))

    plotData(data, "Run Elapsed Time Plot")
    
    return

def getBlockData(selections):
    data = loadData(".data_files/tires/%s/run/master_all.csv"%selections["tire"].get())
    index = getBlockIndexFromString(selections["block"].get())
    data = data[data["block_id"] == index]
    return data

def plotBlockGraph():
    global selections
    data = getBlockData(selections)
    plotData(data, "FZ: %s - IA: %s - P: %s"%(selections["fz"].get(), selections["inclinationAngle"].get(), selections["pressure"].get()))

    return

def plotMasterGraph():
    data = loadData(".data_files/tires/%s/run/master_all.csv"%selections["tire"].get())

    plotData(data, "Master File Elapsed Time Plot")

    return

def buildMasterFile():
    path = ".data_files/tires/%s/run"%selections["tire"].get()
    mb.buildMaster(path, path+"/master_all.csv", path+"/blockGuide.csv")
    return

def buildMFFile():
    data = loadData("./data_files/tires/%s/run/master_all.csv"%selections["tire"].get())
    lat_df = mf.getPureLateral(data)
    FZ0 = mf.getReferenceFz(lat_df)
    p0 = mf.guess_params(lat_df, FZ0)
    mf.fyFitting(p0, FZ0)


def renderButtons(root):
    ypos = 4
    for channel in channels.values():
        channel.setButtons(root, 2*WIDTH/16, ypos*HEIGHT/18)
        ypos+=1
    return

def initializeUserInterface(root):

    global channels
    global selections
    global runSelectionBox
    global blockSelectionBox
    
    root.geometry("%dx%d"%(WIDTH, HEIGHT))
    root.configure(bg="lightblue")
    label = tk.Label(root, text = "Channels", bg = "lightblue", font= "Arial 20")
    label.place(x=int(2*WIDTH/16), y=int(1*HEIGHT/18))
    label = tk.Label(root, text = "Filter", bg = "lightblue")
    label.place(x=int(1*WIDTH/16), y=int(3*HEIGHT/18))

    tireSelection = tk.StringVar(value = "")
    runSelection = tk.StringVar(value = "")
    pressureSelection = tk.StringVar(value = "")
    fzSelection = tk.StringVar(value = "")
    inclinationAngleSelection = tk.StringVar(value = "")
    blockSelection = tk.StringVar(value = "")
    
    channels = {
        "SA": cm.Channel("SA", color = "C0"),
        "SR": cm.Channel("SR", color = "C1"),
        "FY": cm.Channel("FY", color = "C2"),
        "FZ": cm.Channel("FZ", color = "C3"),
        "FX": cm.Channel("FX", color = "C4"),
        "IA": cm.Channel("IA", color = "C5"),
        "V": cm.Channel("V", color = "C6"),
        "P": cm.Channel("P", color = "C7"),
        "MZ": cm.Channel("MZ", color = "C8"),
        "TSTI": cm.Channel("TSTI", color = "C9"),
        "TSTC": cm.Channel("TSTC", color = "C10"),
        "TSTO": cm.Channel("TSTO", color = "C11"),
        "AmbTmp": cm.Channel("AmbTmp", color = "C12")
    }

    selections = {
        "tire": tireSelection,
        "run": runSelection,
        "pressure": pressureSelection,
        "fz": fzSelection,
        "inclinationAngle": inclinationAngleSelection,
        "block": blockSelection
    }

    runOptions = []
    tireOptions = getImmediateSubfolders("C:/Users/gabri/OneDrive/Documentos/Pegasus/P03/FSAE-Python-Projects/Tire Software/drive-download-20260212T184525Z-1-001/data_files/tires")
    pressureOptions= ["8 (psi)", "10 (psi)", "12 (psi)", "14 (psi)"]
    fzOptions= ["222 (N)", "445 (N)", "667 (N)", "890 (N)", "1112 (N)"]
    inclinationAngleOptions= ["0 (deg)", "2 (deg)", "4 (deg)"]

    
    tireSelectionLabel = ttk.Label(root, text = "Tire Selection", background = "lightblue")
    tireSelectionLabel.place(x=int(7*WIDTH/16), y=int(4*HEIGHT/18))
    tireSelectionBox = ttk.OptionMenu(root, tireSelection, tireOptions[0], *tireOptions, command = updateRunFileOptions)
    tireSelectionBox.place(x=int(7*WIDTH/16), y=int(5*HEIGHT/18))

    runSelectionLabel = ttk.Label(root, text = "Run Selection", background = "lightblue")
    runSelectionLabel.place(x=int(7*WIDTH/16), y=int(6*HEIGHT/18))
    runSelectionBox = ttk.OptionMenu(root, runSelection)
    updateRunFileOptions(tireSelection.get())
    runSelectionBox.place(x=int(7*WIDTH/16), y=int(7*HEIGHT/18))

    pressureSelectionLabel = ttk.Label(root, text = "Pressure", background = "lightblue")
    pressureSelectionLabel.place(x=int(7*WIDTH/16), y=int(8*HEIGHT/18))
    pressureSelectionBox = ttk.OptionMenu(root, pressureSelection, pressureOptions[0], *pressureOptions, command=updateBlockOptions)
    pressureSelectionBox.place(x=int(7*WIDTH/16), y=int(9*HEIGHT/18))

    fzSelectionLabel = ttk.Label(root, text = "FZ", background = "lightblue")
    fzSelectionLabel.place(x=int(9*WIDTH/16), y=int(8*HEIGHT/18))
    fzSelectionBox = ttk.OptionMenu(root, fzSelection, fzOptions[0], *fzOptions, command=updateBlockOptions)
    fzSelectionBox.place(x=int(9*WIDTH/16), y=int(9*HEIGHT/18))

    inclinationAngleSelectionLabel = ttk.Label(root, text = "InclinationAngle", background = "lightblue")
    inclinationAngleSelectionLabel.place(x=int(11*WIDTH/16), y=int(8*HEIGHT/18))
    inclinationAngleSelectionBox = ttk.OptionMenu(root, inclinationAngleSelection, inclinationAngleOptions[0], *inclinationAngleOptions, command=updateBlockOptions)
    inclinationAngleSelectionBox.place(x=int(11*WIDTH/16), y=int(9*HEIGHT/18))


    blockSelectionLabel = ttk.Label(root, text = "Block", background = "lightblue")
    blockSelectionLabel.place(x=int(7*WIDTH/16), y=int(10*HEIGHT/18))
    blockSelectionBox = ttk.OptionMenu(root, blockSelection)
    updateBlockOptions("NA")

    blockSelectionBox.place(x=int(7*WIDTH/16), y=int(11*HEIGHT/18))

    renderButtons(root)

    plotBlockGraphButton = ttk.Button(root, text = "plot block graph", command = plotBlockGraph)
    plotBlockGraphButton.place(x=int(12*WIDTH/16), y=int(11*HEIGHT/18))

    plotGraphButton = ttk.Button(root, text = "plot graph", command = plotRunGraph)
    plotGraphButton.place(x=int(12*WIDTH/16), y=int(14*HEIGHT/18))

    plotMasterGraphButton = ttk.Button(root, text = "plot master graph", command = plotMasterGraph)
    plotMasterGraphButton.place(x=int(12*WIDTH/16), y=int(13*HEIGHT/18))

    buildMasterFileButton = ttk.Button(root, text = "build master file", command = buildMasterFile)
    buildMasterFileButton.place(x=int(12*WIDTH/16), y=int(12*HEIGHT/18))
