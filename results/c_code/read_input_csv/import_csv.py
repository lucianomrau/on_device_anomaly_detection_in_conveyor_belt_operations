import csv
import serial
import time
import struct

# Open serial connection
serial_port = '/dev/ttyUSB0' # '/dev/ttyACM1'
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate)  
time.sleep(10)  # Give the connection a second to settle

# Open and read the CSV file
csv_file = 'input_data_MCU_demo.csv' # 'input_data_MCU.csv'



# Read CSV file and send data
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if len(row) == 3:
            # Convert to float and pack as bytes
            float_values = [float(val) for val in row]
            packed_data = struct.pack('<fff', *float_values)
            ser.write(packed_data)

ser.close()
print("transmission completed and finished")
