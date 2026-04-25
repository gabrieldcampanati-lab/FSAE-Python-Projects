import numpy as np
import cv2
import time
import serial
import time

WINDOW_SIZE = (640, 480)


def normalize(x, vec):
    a = 10
    b = 40
    return (x-a)/(b-a)


# Configure the serial port
# Replace 'COM3' with your actual port name (e.g., /dev/ttyUSB0 on Linux, COMx on Windows)
# Ensure the baud rate matches the ESP32's configuration (e.g., 115200)
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2) # Wait for the connection to establish
cv2.namedWindow("Real-Time Matrix Image")
firstline = False

try:
    while True:
        
        # Read data from the ESP32
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # Convert comma-separated string to list of integers
            if line and firstline:
                vector = [float(x) for x in line.split(',')]
                vector = np.array(vector)
                #vector = normalize(vector, vector)
                vector *=-1
                matrix = vector.reshape(24,32)
                frame = matrix
                print("Received Vector:", frame)
                break
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                frame = cv2.resize(frame, WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                cv2.imshow("Real-Time Matrix Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            firstline = True
        time.sleep(0.250)

except KeyboardInterrupt:
    print("Serial communication stopped.")
    ser.close()