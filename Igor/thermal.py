import cv2
import numpy as np
import serial
import matplotlib.pyplot as plt

# Configuration
WIDTH, HEIGHT = 32, 24 
# 16-bit float = 2 bytes per value
PAYLOAD_SIZE = WIDTH * HEIGHT * 2 
PORT = 'COM6' 
BAUD = 921600

ser = serial.Serial(PORT, BAUD, timeout=0.1)

while True:
    if ser.in_waiting >= PAYLOAD_SIZE:
        # 1. Read the raw 2-byte-per-pixel data
        raw_data = ser.read(PAYLOAD_SIZE)
        
        # 2. Convert using 'f2' (16-bit float)
        # Use '<f2' for little-endian, which is standard for ESP32

        # 3. Clean up the data
        # float16 has lower precision; check for NaNs or Inf if the sensor is noisy

        # 4. Normalize to 0-255 (uint8) for OpenCV
        # Using MINMAX allows the thermal "contrast" to stay high 
        # even if the absolute temperatures change.
        # 1. Convert bytes using 'f2' (float16)
        raw_vector = np.frombuffer(raw_data, dtype='<u2').reshape((HEIGHT, WIDTH))
        # 2. CAST TO FLOAT32 (The crucial fix for OpenCV)
        vector = raw_vector.astype(np.float32) / 100.0
        v_min, v_max = 22.0, 40.0
        normalized_float = (np.clip(vector, v_min, v_max) - v_min) / (v_max - v_min)
        cmap = plt.get_cmap('jet')
        color_high_res = cmap(normalized_float) # This is now (HEIGHT, WIDTH, 4) in float64

        # 4. Convert to 16-bit BGR for OpenCV (0.0-1.0 -> 0-65535)
        # We drop the alpha channel [:,:,:3] and swap RGB to BGR [:,:,::-1]
        bgr_16bit = (color_high_res[:, :, :3][:, :, ::-1] * 65535).astype(np.uint16)

        # 5. Display
        display = cv2.resize(bgr_16bit, (640, 480), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("16-bit High-Res Heatmap", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cv2.destroyAllWindows()