import struct
import SocketServer

unpacker_header = struct.Struct('2s I')

class MyTCPHandler(SocketServer.StreamRequestHandler):

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        self.request.setblocking(1)
        self.data = self.request.recv(unpacker_header.size)
        self.header = unpacker_header.unpack(self.data)
        print "{} wrote:".format(self.client_address[0])
        print self.header[0]
        print self.header[1]
        self.data = self.request.recv(self.header[1])
        # Likewise, self.wfile is a file-like object used to write back
        # to the client
        self.wfile.write(self.data.upper())


if __name__ == "__main__":
    HOST, PORT = "localhost", 9998

    # Create the server, binding to localhost on port 9999
    SocketServer.TCPServer.allow_reuse_address = True
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
