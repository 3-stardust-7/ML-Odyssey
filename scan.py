import numpy as np
import matplotlib.pyplot as plt

def scan_disk_scheduling(requests, head, disk_size=200):
    requests.sort()
    seek_sequence = []
    total_seek_time = 0

    # Find index where head should start moving up
    i = 0
    while i < len(requests) and requests[i] < head:
        i += 1

    # Move upwards first
    for j in range(i, len(requests)):
        seek_sequence.append(requests[j])
        total_seek_time += abs(head - requests[j])
        head = requests[j]

    # Move downwards after reaching the highest request
    for j in range(i-1, -1, -1):
        seek_sequence.append(requests[j])
        total_seek_time += abs(head - requests[j])
        head = requests[j]

    return seek_sequence, total_seek_time

# Example requests and initial head position
requests = [10,20,30,40,50]
head = 25
disk_size = 200  # Defining disk size

# Get SCAN movement data
seek_sequence, total_seek_time = scan_disk_scheduling(requests, head, disk_size)

# Create X values for plotting (step count)
x_values = np.arange(len(seek_sequence))
y_values = seek_sequence

# Plotting the SCAN movement
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label="SCAN Movement")
plt.xlabel("Request Processing Order")
plt.ylabel("Track Number")
plt.title("SCAN Disk Scheduling Algorithm")
plt.grid(True)
plt.legend()
plt.show()
