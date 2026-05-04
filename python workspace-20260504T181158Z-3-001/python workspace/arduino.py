import serial
import time
import csv
import os

#---configuration---#
PORT = 'COM3'
BAUD_RATE = 115200
LOG_FILE_PATH = "log.csv"
HEADERS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]
#-------------------#

def connectionInit(arduino_port, baud_rate):
    try:
        ser = serial.Serial(arduino_port, baud_rate, timeout = 1)
        print(f"Connected to Arduino on {arduino_port} at {baud_rate} baud.")
    except serial.SerialException as e:
        print(f"Error connecting to serial port: {e}")
        exit()
    return ser

def logLoop(ser, file):
    c = 0
    while True:
        line_bytes = ser.readline()
        if not line_bytes: continue

        try:
            line_str = line_bytes.decode('utf-8')
        except:
            continue
        if not line_str: continue

        file.write(line_str)
        file.flush()
        if c > 1000:
            print(line_str)
            c = 0

    return

def writeLogFile(ser, path, headers):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        logLoop(ser, file)

    return

ser = connectionInit(PORT, BAUD_RATE)
writeLogFile(ser, LOG_FILE_PATH, HEADERS)
