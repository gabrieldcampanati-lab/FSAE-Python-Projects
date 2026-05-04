"""
Ver: 0.2

"""

import tkinter as tk
from tkinter import ttk
from tkinter import font
import raw_data_manager as rdm
import tyre_ploter as tp

WIDTH = 240
HEIGHT = 320
PADX, PADY = 5,5

global root

def launchRawDataManager():
    global root
    rdmRoot = tk.Toplevel(root)
    rdmRoot.title("TyreView (Nome Sujeito a Mudanças) - RawDataManager")
    rdm.initializeUserInterface(rdmRoot)

def launchTyrePloter():
    global root
    tpRoot = tk.Toplevel(root)
    tpRoot.title("TyreView (Nome Sujeito a Mudanças) - Tyre Ploter")
    tp.initializeUserInterface(tpRoot)


def initializeUserInterface(root):
    root.geometry("%dx%d"%(WIDTH, HEIGHT))
    root.title("TyreViewer (Nome Sujeito a Mudanças)")
    root.configure(bg="lightblue")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    frame = tk.Frame(root)
    frame.grid()
    titleFont = font.Font(family = "Times New Roman", size = 20, slant="italic")
    titleLable = tk.Label(root, text = "TyreViewer", font = titleFont, background = "lightblue", justify=tk.CENTER)
    titleLable.grid(row = 0, column = 0, padx=PADX, pady=PADY, sticky="nsew")
    rawDataManagerButton = ttk.Button(root, text="RawDataManager", command = launchRawDataManager)
    rawDataManagerButton.grid(row = 1, column = 0, padx=PADX, pady=PADY, sticky="nsew")
    tyrePloterButton = ttk.Button(root, text="TyrePloter", command = launchTyrePloter)
    tyrePloterButton.grid(row = 2, column = 0, padx=PADX, pady=PADY, sticky="nsew")



root = tk.Tk()
initializeUserInterface(root)
root.mainloop()

