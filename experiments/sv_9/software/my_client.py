import struct
import cv2
import numpy as np
import socket
import sys

HOST, PORT = "localhost", 9998

# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    data = "Hello world"
    header = ('00', len(data))
    packer_header = struct.Struct('2s I')
    packed_header = packer_header.pack(*header)
    sock.sendall(packed_header)
    sock.sendall(data)

    # Receive data from the server and shut down
    received = sock.recv(1024)
finally:
    sock.close()

print "Sent:     {}".format(header)
print "Received: {}".format(received)
