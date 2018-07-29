import struct
import SocketServer
import numpy as np
import cv2

unpacker_header = struct.Struct('= 2s I')
unpacker_image_header = struct.Struct('= 2s I I')

class MyTCPHandler(SocketServer.StreamRequestHandler):

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        self.request.setblocking(1)

        self.data = self.request.recv(unpacker_header.size)
        self.header = unpacker_header.unpack(self.data)
        print "{} wrote:".format(self.client_address[0])
        packet = {}
        packet["version"] = self.header[0]
        packet["size"] = self.header[1]
        print "packet version:" , packet["version"]
        print "packet size:" , packet["size"]

        self.data = self.request.recv(unpacker_image_header.size)
        self.image_header = unpacker_image_header.unpack(self.data)
        image_header = {}
        image_header["version"] = self.image_header[0]
        image_header["rows"] = self.image_header[1]
        image_header["cols"] = self.image_header[2]
        print "image header version:", image_header["version"]
        print "image header num rows:", image_header["rows"]
        print "image header num cols:", image_header["cols"]
        
        image_buffer_size = packet["size"] - unpacker_image_header.size
        self.data = bytearray(image_buffer_size)
        view = memoryview(self.data)
        toread = image_buffer_size
        while toread:
            nbytes = self.request.recv_into(view, toread)
            view = view[nbytes:]
            toread -= nbytes
        print len(self.data)
        self.image_buffer = np.frombuffer(self.data, dtype = 'uint8')
        
        self.image = self.image_buffer.reshape((image_header["rows"], image_header["cols"]))
        self.image = cv2.resize(self.image, (0,0), fx=0.25, fy=0.25)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.image)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

        # Likewise, self.wfile is a file-like object used to write back
        # to the client
        self.wfile.write("RECEIVED" + str(image_header["rows"]) + "x" + str(image_header["cols"]))


if __name__ == "__main__":
    HOST, PORT = "localhost", 9998

    # Create the server, binding to localhost on port 9999
    SocketServer.TCPServer.allow_reuse_address = True
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
