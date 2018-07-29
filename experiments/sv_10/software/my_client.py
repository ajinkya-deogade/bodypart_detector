import struct
import cv2
import numpy as np
import socket
import sys

packer_header = struct.Struct('= 2s I')
packer_image_header = struct.Struct('= 2s I I')

if __name__ == "__main__":
    HOST, PORT = "localhost", 9998

    image = cv2.imread(sys.argv[1])
#    image = image[900:1100,900:1100,0]
    image = image[:,:,0]
    image_info = np.shape(image)
    print image_info
    image_header = ('00', image_info[0], image_info[1])
    packed_image_header = packer_image_header.pack(*image_header)

    image_data = np.getbuffer(np.ascontiguousarray(image))
    print len(image_data)

    for i in range(0, 1):
        # Create a socket (SOCK_STREAM means a TCP socket)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
        try:
            # Connect to server and send data
            sock.connect((HOST, PORT))

            print i
            header = ('00', len(packed_image_header) + len(image_data))
            packed_header = packer_header.pack(*header)
            sock.sendall(packed_header)
            sock.sendall(packed_image_header)
            sock.sendall(image_data)
            
            # Receive data from the server and shut down
            received = sock.recv(1024)
            print "Sent:     {}".format(header)
            print "Received: {}".format(received)

        finally:
            sock.close()

