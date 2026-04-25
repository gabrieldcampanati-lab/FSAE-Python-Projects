"""

Ver.: 0.1

"""

import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import ttk
from tkinter import font
import os
import index_tables as it
import channel_module as cm
import raw_data_manager as rdm
import magic_formula as mf
import numpy as np
import math
import glob

WIDTH, HEIGHT = 1280, 720
PADX, PADY = 2, 2
FRAME_BACKGROUND_COLOR = "#D9D9D9"
FRAME_TITLE_FONT_COLOR = "#081E26"
FRAME_FONT_COLOR = "#737373"
WINDOW_BACKGORUND_COLOR = "#081E26"

def updateBlockOptions(na):
    global blockOptions
    global blockSelectionBox
    blockGuide = rdm.loadData("./data_files/tires/%s/run/blockGuide.csv"%selections["tire"].get())
    blocks = blockGuide[
        (blockGuide["FZ"] == it.fzSelectionValues[selections["fz"].get()]) &
        (blockGuide["IA"] == it.inclinationAngleSelectionValues[selections["inclinationAngle"].get()]) &
        (blockGuide["P"] == it.pressureSelectionValues[selections["pressure"].get()])
        ]
    print(blocks)
    blockOptions = rdm.getBlocksNames(blocks)
    blockSelectionBox['menu'].delete(0, 'end')
    # Add new options
    for option in blockOptions:
        blockSelectionBox['menu'].add_command(label=option, command=tk._setit(selections["block"], option))
    # Set a new default selected option
    selections["block"].set(blockOptions[0])
    updateMFTyreName()

    return

def addSlipAngleChannel():
    global selections
    global mfTyres
    data = rdm.getBlockData(selections)
    data = data.sort_values(by="SA")

    sa = np.linspace(-45, 45, num=1000)
    fy = []
    fz = -float(selections["mf_fz"].get())
    ia = float(selections["mf_inclinationAngle"].get())
    tyre = mfTyres[selections["mf_selection"].get()]
    print(tyre.fz0)
    for alpha in sa:
        tempData = data[abs(data["SA"]-alpha)<0.1]
        temp = np.mean(tempData["TIRE_TEMP"])
        fy.append(tyre.getFy(math.radians(alpha), fz, math.radians(ia), 83))
    tyre.printMFP(math.radians(-12), fz, math.radians(ia), 83)
    tyre.printMFP(math.radians(12), fz, math.radians(ia), 83)
    mfChannel = cm.Channel("%s: %s (N) - %s (deg)"%(tyre.name, selections["mf_fz"].get(), selections["mf_inclinationAngle"].get()), data = fy, color="C%d"%len(channels[selections["plot_type"].get()]), visibility=1,
                           filtered=0, scaleble=False, xdata = sa)
    
    return mfChannel

def addSlipAngleChannel2():
    global selections
    global mfTyres
    data = rdm.getBlockData(selections)
    data = data.sort_values(by="SA")

    sa = np.linspace(-45, 45, num=1000)
    fy = []
    fz = -float(selections["mf_fz"].get())
    ia = float(selections["mf_inclinationAngle"].get())
    tyre = mfTyres[selections["mf_selection"].get()]
    for alpha in sa:
        tempData = data[abs(data["SA"]-alpha)<0.1]
        temp = np.mean(tempData["TIRE_TEMP"])
        fy.append(tyre.getFy3(math.radians(alpha), fz, math.radians(ia), 83))
    mfChannel = cm.Channel("%s: %s (N) - %s (deg)"%(tyre.name, selections["mf_fz"].get(), selections["mf_inclinationAngle"].get()), data = fy, color="C%d"%len(channels["FY x SA"]), visibility=1,
                           filtered=0, scaleble=False, xdata = sa)
    
    return mfChannel

def addCorneringStiffnessChannel():
    global selections
    global mfTyres
    sa = np.linspace(-20, 20, num=1000)
    fy = []
    fz = -float(selections["mf_fz"].get())
    ia = float(selections["mf_inclinationAngle"].get())
    tyre = mfTyres[selections["mf_selection"].get()]
    for alpha in sa:
        fy.append(tyre.getCorneringStiffness(math.radians(alpha), fz, math.radians(ia), 83, 1e-5))
    mfChannel = cm.Channel("%s: %s (N) - %s (deg)"%(tyre.name, selections["mf_fz"].get(), selections["mf_inclinationAngle"].get()), data = fy, color="C%d"%len(channels[selections["plot_type"].get()]), visibility=1,
                           filtered=0, scaleble=False, xdata = sa)
    return mfChannel

def addCamberStiffnessChannel():
    global selections
    global mfTyres
    sa = np.linspace(-20, 20, num=1000)
    fy = []
    fz = -float(selections["mf_fz"].get())
    ia = float(selections["mf_inclinationAngle"].get())
    tyre = mfTyres[selections["mf_selection"].get()]
    for alpha in sa:
        fy.append(tyre.getCamberStiffness(math.radians(alpha), fz, math.radians(ia), 83, 1e-5))
    mfChannel = cm.Channel("%s: %s (N) - %s (deg)"%(tyre.name, selections["mf_fz"].get(), selections["mf_inclinationAngle"].get()), data = fy, color="C%d"%len(channels[selections["plot_type"].get()]), visibility=1,
                           filtered=0, scaleble=False, xdata = sa)
    return mfChannel

def addPeakFyxFzChannel():
    fz = np.linspace(100, 2000, num= 2000)
    tyre = mfTyres[selections["mf_selection"].get()]
    print(tyre.name)
    print(tyre.fyParameters[-1112])
    for t in mfTyres.values():
        print(t.fyParameters == tyre.fyParameters)
    #tyre.fz0 = it.fzSelectionValues[selections["fz"].get()]
    ia = math.radians(float(selections["mf_inclinationAngle"].get()))
    fy = []
    for n in fz:
        fy.append(tyre.getMu(-n, ia, 83))
    mfChannel = cm.Channel("%s: %s (N) - %s (deg)"%(tyre.name, selections["fz"].get(), selections["mf_inclinationAngle"].get()), data = fy, color="C%d"%len(channels[selections["plot_type"].get()]), visibility=1,
                           filtered=0, scaleble=False, xdata = fz)
    return mfChannel

def plotChannels():
    fig, ax = plt.subplots()
    """
    for channel in channels[selections["plot_type"].get()].values():
        channel.plot(ax)
    """
    for channelType in channels.values():
        for channel in channelType.values():
            channel.plot(ax)
    ax.tick_params(axis = "both", labelcolor = "#000000")
    ax.set_xlabel("SA (deg)")
    ax.set_ylabel(selections["plot_type"].get())
    ax.grid(True)
    fig.legend()
    fig.show()
    return

def plotPeakSlipAngle():
    return



def addChannel():
    global channels
    data = rdm.getBlockData(selections)
    data = data.sort_values(by="SA")
    """
    fyChannel = cm.Channel("Tyre: %s\nP: %s - IA: %s - FZ: %s - %s"%(selections["tire"].get(),
                                                                      selections["pressure"].get(),
                                                                        selections["inclinationAngle"].get(),
                                                                          selections["fz"].get(), selections["block"].get()), data = data["FY"],
                                                                            color="C%d"%len(channels[selections["plot_type"].get()]), visibility=1, filtered=0, scaleble=False, xdata = data["SA"], plotingType="scatter")
    channels[selections["plot_type"].get()][fyChannel.name] = fyChannel
    fyChannel = cm.Channel("Tyre: %s\nP: %s - IA: %s - FZ: %s - %s Filtered"%(selections["tire"].get(),
                                                                      selections["pressure"].get(),
                                                                        selections["inclinationAngle"].get(),
                                                                          selections["fz"].get(), selections["block"].get()), data = data["FY"],
                                                                            color="C%d"%len(channels[selections["plot_type"].get()]), visibility=1, filtered=1, scaleble=False, xdata = data["SA"])
    channels[selections["plot_type"].get()][fyChannel.name] = fyChannel
    """
    fz = float(it.fzSelectionValues[selections["fz"].get()])
    fyChannel = cm.Channel("Tyre: %s\nP: %s - IA: %s - FZ: %s - %s Normalized"%(selections["tire"].get(),
                                                                      selections["pressure"].get(),
                                                                        selections["inclinationAngle"].get(),
                                                                          selections["fz"].get(), selections["block"].get()), data = (data["FY"]/data["FZ"])*fz,
                                                                            color="C%d"%len(channels[selections["plot_type"].get()]), visibility=1, filtered=0, scaleble=False, xdata = data["SA"], plotingType="scatter")
    channels[selections["plot_type"].get()][fyChannel.name] = fyChannel
    
    return

def addMFTyre():
    global mfTyreOptions
    global mfTyreSelectionBox
    global mfTyres
    if selections["mf_name"].get() in mfTyreOptions:
        return
    tyreName = selections["mf_name"].get()

    data = rdm.loadData("./data_files/tires/%s/run/master_all.csv"%selections["tire"].get())
    blockGuide = rdm.loadData("./data_files/tires/%s/run/blockGuide.csv"%selections["tire"].get())
    blocks = blockGuide[
        (blockGuide["P"] == it.pressureSelectionValues[selections["pressure"].get()]) &
        (abs(blockGuide["V"]-40) < 5)
        ]
    data = data[data["block_id"].isin(blocks["block_id"])]

    newTyre = mf.MFTire(tyreName, data=data, fyParameters = {}, blockGuide=blockGuide, path="./data_files/tires/%s"%selections["tire"].get(), fz0 = -1112)
    newTyre.calculateFyParameters()
    mfTyreSelectionBox['menu'].add_command(label=tyreName, command=tk._setit(selections["mf_selection"], tyreName))
    selections["mf_selection"].set(tyreName)
    newTyre.saveLateralParameters()
    newTyre.fyActiveParameters = "simple"
    newTyre.fz0 = -1112
    mfTyres[tyreName] = newTyre
    return

def addMFTyreChannel():
    mfChannel = plotTypes[selections["plot_type"].get()]()
    channels[selections["plot_type"].get()][mfChannel.getName()] = mfChannel
    return

def updateMFTyreName():
    selections["mf_name"].set("P:%s - %s"%(selections["pressure"].get(), selections["tire"].get()))
    return

def resetChannels():
    global channels
    channels = {"FY x SA": {},
                "FY x SA2": {},
                "CorneringStiffness x SA": {},
                "PeakSA x Fz": {},
                "CamberStiffness x SA": {},
                "PeakFY x FZ": {}}
    return

plotTypes = {
        "FY x SA": addSlipAngleChannel,
        "FY x SA2": addSlipAngleChannel2,
        "CorneringStiffness x SA": addCorneringStiffnessChannel,
        "PeakSA x Fz": plotPeakSlipAngle,
        "CamberStiffness x SA": addCamberStiffnessChannel,
        "PeakFY x FZ": addPeakFyxFzChannel
    }

def getMFFolders(path):
    pattern = os.path.join(path, f"*{".mfp"}")
    return [os.path.basename(file) for file in glob.glob(pattern)]

def updateMFTyres():
    global mfTyreOptions
    global mfTyres

    for tireFolder in rdm.getImmediateSubfolders("./data_files/tires"):
        print("update mf")
        mfPath = "%s/%s/MF"%("./data_files/tires",tireFolder)
        for f in rdm.getImmediateSubfolders(mfPath):
            name = f
            print(name)
            tempTyre = mf.MFTire(name, fyParameters = {}, path = "%s/%s"%(mfPath, name), fz0=-1112)
            tempTyre.loadLateralParameters(path = "%s/%s"%(mfPath, name))
            mfTyres[name] = tempTyre
            mfTyreSelectionBox['menu'].add_command(label=name, command=tk._setit(selections["mf_selection"], name))
            selections["mf_selection"].set(name)
    return

def initializeUserInterface(root):

    global selections
    global channels
    global blockSelectionBox
    global mfTyreSelectionBox
    global mfTyreOptions
    global mfTyres

    root.geometry("%dx%d"%(WIDTH, HEIGHT))
    root.configure(bg= WINDOW_BACKGORUND_COLOR)

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)

    titleFont = font.Font(family = "Times New Roman", size = 24, slant="italic")
    labelFont = font.Font(family = "Times New Roman", size = 12)

    buttonStyle = ttk.Style()
    buttonStyle.configure('TButton', font = labelFont)
    buttonStyle.configure('TMenubutton', font = labelFont)

    generalFrame = tk.Frame(root, bg = FRAME_BACKGROUND_COLOR)
    generalFrame.grid(row=0, column=0, sticky= "nsew", padx= PADX, pady= PADY)
    generalFrame.rowconfigure(0, weight=1)
    generalFrame.rowconfigure(1, weight=1)
    generalFrame.rowconfigure(2, weight=1)
    generalFrame.columnconfigure(0, weight=1)
    generalFrame.columnconfigure(1, weight=1)
    generalFrame.columnconfigure(2, weight=1)
    generalFrame.columnconfigure(3, weight=1)

    mfTyreFrame = tk.Frame(root, bg = FRAME_BACKGROUND_COLOR)
    mfTyreFrame.grid(row=1, column=0, sticky= "nsew", padx= PADX)
    mfTyreFrame.rowconfigure(0, weight=1)
    mfTyreFrame.rowconfigure(1, weight=1)
    mfTyreFrame.rowconfigure(2, weight=1)
    mfTyreFrame.columnconfigure(0, weight=1)
    mfTyreFrame.columnconfigure(1, weight=1)
    mfTyreFrame.columnconfigure(2, weight=1)
    mfTyreFrame.columnconfigure(3, weight=1)
    mfTyreFrame.columnconfigure(4, weight=1)

    rawDataFrame = tk.Frame(root, bg = FRAME_BACKGROUND_COLOR)
    rawDataFrame.grid(row=2, column=0, sticky= "nsew", padx= PADX, pady= PADY)
    rawDataFrame.rowconfigure(0, weight=1)
    rawDataFrame.rowconfigure(1, weight=1)
    rawDataFrame.rowconfigure(2, weight=1)
    rawDataFrame.columnconfigure(0, weight=1)
    rawDataFrame.columnconfigure(1, weight=1)
    rawDataFrame.columnconfigure(2, weight=1)
    rawDataFrame.columnconfigure(3, weight=1)
    
    
#----------------------TK variables--------------------------#
    tireSelection = tk.StringVar(value = "")
    pressureSelection = tk.StringVar(value = "")
    fzSelection = tk.StringVar(value = "")
    inclinationAngleSelection = tk.StringVar(value = "")
    blockSelection = tk.StringVar(value = "")
    mfFzSelection = tk.StringVar(value = "222")
    mfInclinationAngleSelection = tk.StringVar(value = "0")
    mfTyreSelection = tk.StringVar(value = "")
    mfTyreName = tk.StringVar(value = "")
    plotTypeSelection = tk.StringVar(value = "SA x FY")

    resetChannels()

    selections = {
        "tire": tireSelection,
        "pressure": pressureSelection,
        "fz": fzSelection,
        "inclinationAngle": inclinationAngleSelection,
        "block": blockSelection,
        "mf_fz": mfFzSelection,
        "mf_inclinationAngle": mfInclinationAngleSelection,
        "mf_name": mfTyreName,
        "mf_selection": mfTyreSelection,
        "plot_type": plotTypeSelection
    }

    tireOptions = rdm.getImmediateSubfolders("./data_files/tires")
    pressureOptions= ["8 (psi)", "10 (psi)", "12 (psi)", "14 (psi)"]
    fzOptions= ["222 (N)", "445 (N)", "667 (N)", "890 (N)", "1112 (N)"]
    inclinationAngleOptions= ["0 (deg)", "2 (deg)", "4 (deg)"]
    mfTyreOptions = []
    mfTyres = {}
    plotTypeOptions = ["FY x SA", "FY x SA2", "CorneringStiffness x SA", "PeakSA x FZ", "CamberStiffness x SA", "PeakFY x FZ"]


    generalFrameLabel = ttk.Label(generalFrame, text = "General Settings", font=titleFont, background=FRAME_BACKGROUND_COLOR, foreground= FRAME_TITLE_FONT_COLOR)
    generalFrameLabel.grid(row= 0, column= 0, padx=PADX, pady=PADY, sticky="nw")
    mfTyreFrameLabel = ttk.Label(mfTyreFrame, text = "Magic Formula Channel Settings", font=titleFont, background=FRAME_BACKGROUND_COLOR, foreground= FRAME_TITLE_FONT_COLOR)
    mfTyreFrameLabel.grid(row= 0, column= 0, padx=PADX, pady=PADY, sticky="nw")
    rawDataFrameLabel = ttk.Label(rawDataFrame, text = "Raw Data Channel Settings", font=titleFont, background=FRAME_BACKGROUND_COLOR, foreground= FRAME_TITLE_FONT_COLOR)
    rawDataFrameLabel.grid(row= 0, column= 0, padx=PADX, pady=PADY, sticky="nw")
    
    



#--------------------------Selection Boxes----------------------------------#

    tireSelectionLabel = ttk.Label(generalFrame, text = "Tire Selection", background = "lightblue", font = labelFont)
    #tireSelectionLabel.place(x=int(7*WIDTH/16), y=int(4*HEIGHT/18))
    tireSelectionLabel.grid(row = 1, column = 0, padx = PADX, pady= PADY, sticky= "nsew")
    tireSelectionBox = ttk.OptionMenu(generalFrame, tireSelection, tireOptions[0], *tireOptions, command=updateBlockOptions)
    tireSelectionBox.grid(row = 2, column = 0, padx = PADX, pady= PADY, sticky= "nsew")

    pressureSelectionLabel = ttk.Label(generalFrame, text = "Pressure", background = "lightblue", font = labelFont)
    pressureSelectionLabel.grid(row = 1, column = 1, padx = PADX, pady= PADY, sticky= "nsew")
    pressureSelectionBox = ttk.OptionMenu(generalFrame, pressureSelection, pressureOptions[0], *pressureOptions, command=updateBlockOptions)
    pressureSelectionBox.grid(row = 2, column = 1, padx = PADX, pady= PADY, sticky= "nsew")

    fzSelectionLabel = ttk.Label(rawDataFrame, text = "FZ", background = "lightblue", font = labelFont)
    fzSelectionLabel.grid(row = 1, column = 1, padx = PADX, pady= PADY, sticky= "nsew")
    fzSelectionBox = ttk.OptionMenu(rawDataFrame, fzSelection, fzOptions[0], *fzOptions, command=updateBlockOptions)
    fzSelectionBox.grid(row = 2, column = 1, padx = PADX, pady= PADY, sticky= "nsew")

    inclinationAngleSelectionLabel = ttk.Label(rawDataFrame, text = "InclinationAngle", background = "lightblue", font = labelFont)
    inclinationAngleSelectionLabel.grid(row = 1, column = 2, padx = PADX, pady= PADY, sticky= "nsew")
    inclinationAngleSelectionBox = ttk.OptionMenu(rawDataFrame, inclinationAngleSelection, inclinationAngleOptions[0], *inclinationAngleOptions, command=updateBlockOptions)
    inclinationAngleSelectionBox.grid(row = 2, column = 2, padx = PADX, pady= PADY, sticky= "nsew")

    mfFzSelectionLabel = ttk.Label(mfTyreFrame, text = "MF Fz", background = "lightblue", font = labelFont)
    mfFzSelectionLabel.grid(row = 1, column = 1, padx = PADX, pady= PADY, sticky= "nsew")
    mfFzSelectionBox = ttk.Spinbox(mfTyreFrame, textvariable= mfFzSelection, increment= 100, command=updateMFTyreName, from_=0, to=3000)
    mfFzSelectionBox.grid(row = 2, column = 1, padx = PADX, pady= PADY, sticky= "nsew")

    mfInclinationAngleSelectionLabel = ttk.Label(mfTyreFrame, text = "MF IA", background = "lightblue", font = labelFont)
    mfInclinationAngleSelectionLabel.grid(row = 1, column = 2, padx = PADX, pady= PADY, sticky= "nsew")
    mfInclinationAngleSelectionBox = ttk.Spinbox(mfTyreFrame, textvariable = mfInclinationAngleSelection, increment=0.2, command=updateMFTyreName, from_=-5, to=5)
    mfInclinationAngleSelectionBox.grid(row = 2, column = 2, padx = PADX, pady= PADY, sticky= "nsew")

    blockSelectionLabel = ttk.Label(rawDataFrame, text = "Block", background = "lightblue", font = labelFont)
    blockSelectionLabel.grid(row = 1, column = 0, padx = PADX, pady= PADY, sticky= "nsew")
    blockSelectionBox = ttk.OptionMenu(rawDataFrame, blockSelection)
    updateBlockOptions("NA")
    blockSelectionBox.grid(row = 2, column = 0, padx = PADX, pady= PADY, sticky= "nsew")

    mfTyreNameLabel = ttk.Label(mfTyreFrame, text = "New MF Tyre Name", background = "lightblue", font = labelFont)
    mfTyreNameLabel.grid(row = 1, column = 3, padx = PADX, pady= PADY, sticky= "nsew")
    mfTyreNameBox = ttk.Entry(mfTyreFrame, textvariable=mfTyreName)
    mfTyreNameBox.grid(row = 2, column = 3, padx = PADX, pady= PADY, sticky= "nsew")

    mfTyreLabel = ttk.Label(mfTyreFrame, text = "MF Tyres", background = "lightblue", font = labelFont)
    mfTyreLabel.grid(row = 1, column = 0, padx = PADX, pady= PADY, sticky= "nsew")
    mfTyreSelectionBox = ttk.OptionMenu(mfTyreFrame, mfTyreSelection)
    mfTyreSelectionBox.grid(row = 2, column = 0, padx = PADX, pady= PADY, sticky= "nsew")

    plotTypeLabel = ttk.Label(generalFrame, text= "Plot Type", background = "lightblue", font = labelFont)
    plotTypeLabel.grid(row = 1, column = 2, padx = PADX, pady= PADY, sticky= "nsew")
    plotTypeSelectionBox = ttk.OptionMenu(generalFrame, plotTypeSelection, plotTypeOptions[0], *plotTypeOptions)
    plotTypeSelectionBox.grid(row = 2, column = 2, padx = PADX, pady= PADY, sticky= "nsew")


    updateMFTyres()
    #------------------------Buttons-----------------------------#

    plotSlipAngleButton = ttk.Button(generalFrame, text="Plot Graph", command = plotChannels)
    plotSlipAngleButton.grid(row = 1, column = 3, padx = PADX, pady= PADY, sticky= "nsew")
    #plotSlipAngleButton.grid(row = 5, column= 0, padx=PADX, pady = PADY)

    addChannelButton = ttk.Button(rawDataFrame, text="Add Channel", command = addChannel)
    #addChannelButton.place(x=int(2*WIDTH/16), y=int(5*HEIGHT/18))
    addChannelButton.grid(row = 2, column = 3, padx = PADX, pady= PADY, sticky= "nsew")
    #addChannelButton.pack(padx= PADX, pady= PADY)

    resetChannelButton = ttk.Button(generalFrame, text="Reset Channels", command = resetChannels)
    resetChannelButton.grid(row = 2, column = 3, padx = PADX, pady= PADY, sticky= "nsew")

    addMFTyreButton = ttk.Button(mfTyreFrame, text="Add MF Tyre", command = addMFTyre)
    addMFTyreButton.grid(row = 1, column = 4, padx = PADX, pady= PADY, sticky= "nsew")

    addMFTyreButton = ttk.Button(mfTyreFrame, text="Add MF Tyre Channel", command = addMFTyreChannel)
    addMFTyreButton.grid(row = 2, column = 4, padx = PADX, pady= PADY, sticky= "nsew")

    print(mfTyres)

