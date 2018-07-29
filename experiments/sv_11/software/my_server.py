import struct
import SocketServer
import numpy as np
import cv2

unpacker_header = struct.Struct('= 2s I')
unpacker_list_header = struct.Struct('= 2s I')
unpacker_image_header = struct.Struct('= 2s I I I I')

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
        assert packet["version"] == "01"
        packet["size"] = self.header[1]
        print "packet version:" , packet["version"]
        print "packet size:" , packet["size"]

        self.data = self.request.recv(unpacker_list_header.size)
        self.list_header = unpacker_list_header.unpack(self.data)
        packet["list"] = {}
        packet["list"]["version"] = self.list_header[0]
        assert packet["list"]["version"] == "00"
        packet["list"]["length"] = self.list_header[1]
        print "list version:" , packet["list"]["version"]
        print "list length:" , packet["list"]["length"]
        
        ack_message="RECEIVED"
        for i in range(0, packet["list"]["length"]):
            self.data = self.request.recv(unpacker_image_header.size)
            self.image_header = unpacker_image_header.unpack(self.data)
            image_header = {}
            image_header["version"] = self.image_header[0]
            assert image_header["version"] == "01"
            image_header["rows"] = self.image_header[1]
            image_header["cols"] = self.image_header[2]
            image_header["crop_x"] = self.image_header[3]
            image_header["crop_y"] = self.image_header[4]

            print "image header version:", image_header["version"]
            print "image header num rows:", image_header["rows"]
            print "image header num cols:", image_header["cols"]
            print "image header origin-x-coord:", image_header["crop_x"]
            print "image header origin-y-coord:", image_header["crop_y"]
        
            image_buffer_size = image_header["rows"]*image_header["cols"]
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
            cv2.namedWindow('image' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('image' + str(i), self.image)

            ack_message += " " + str(image_header["rows"]) + "x" + str(image_header["cols"])

<<<<<<< HEAD
        cv2.waitKey(800)
=======
        cv2.waitKey(100)
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7
        cv2.destroyAllWindows()

        # Likewise, self.wfile is a file-like object used to write back
        # to the client
        self.wfile.write(ack_message)


if __name__ == "__main__":
    HOST, PORT = "localhost", 9998

    # Create the server, binding to localhost on port 9999
    SocketServer.TCPServer.allow_reuse_address = True
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
