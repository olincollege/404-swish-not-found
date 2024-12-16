import sys
import serial
from ball_scanner import BallScanner
import time

# import serial

arduinoComPort = "/dev/ttyACM0"
baudRate = 115200

# open the serial port
arduino = serial.Serial(arduinoComPort, baudRate, timeout=1)
time.sleep(3)

# Intialize ball scanner
scanner = BallScanner(save_history=False, calibrate=False, reset=True)

land_history_send = 4
# for i in range(200):
#     print(f"Scan: {i}")
while True:
    scanner.get_scan()
    if len(scanner.ball_history) > 1:
        position = scanner.get_landing_position()
        if len(scanner.land_history_hoop_coords) == land_history_send:
            coords = f"{((scanner.land_history_hoop_coords[land_history_send - 1][0]*39.3701)+1):.2f},{((scanner.land_history_hoop_coords[land_history_send - 1][1]*39.3701)-2.5):.2f}\n"
            arduino.write(bytes(coords, "utf-8"))
            print(f"\nCOORDS SENT: {coords}")
            print("Reset")
            # scanner.reset()
            time.sleep(BallScanner.reset_pause_time)


scanner.close()
sys.exit(0)
