import struct
import cv2
import numpy as np
import socket
import sys
import copy

packer_header = struct.Struct('= 2s I')
packer_list_header = struct.Struct('= 2s I')
packer_image_header = struct.Struct('= 2s I I I I')

if __name__ == "__main__":
    HOST, PORT = "localhost", 9998

    images = []
    packed_image_headers = []
    images_data = []
    for i in range(1,len(sys.argv)):
        image = cv2.imread(sys.argv[i])
        crop_x = 0; crop_y = 0
        # crop_x = 900; crop_y = 900; image = image[crop_y:crop_y+200,crop_x:crop_x+200,0]
        image = image[:,:,0]
        images.append(copy.deepcopy(image))

        image_info = np.shape(image)
        print image_info
        image_header = ('01', image_info[0], image_info[1], crop_x, crop_y)
        packed_image_header = packer_image_header.pack(*image_header)
        packed_image_headers.append(copy.deepcopy(packed_image_header))

        image_data = np.getbuffer(np.ascontiguousarray(image))
        print len(image_data)
        images_data.append(image_data)

    list_header = ('00', len(images_data))
    packed_list_header = packer_list_header.pack(*list_header)

    blob_size = len(packed_list_header)
    for j in range(0, len(images_data)):
        blob_size += len(packed_image_headers[j]) + len(images_data[j])
            
    header = ('01', blob_size)
    packed_header = packer_header.pack(*header)

    for i in range(0, 100):
        # Create a socket (SOCK_STREAM means a TCP socket)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
        try:
            # Connect to server and send data
            sock.connect((HOST, PORT))

            print i
            
            sock.sendall(packed_header)

            sock.sendall(packed_list_header)

            for j in range(0, len(images_data)):
                sock.sendall(packed_image_headers[j])
                sock.sendall(images_data[j])
            
            # Receive data from the server and shut down
            received = sock.recv(1024)
            print "Sent:     {}".format(header)
            print "Received: {}".format(received)

        finally:
            sock.close()

