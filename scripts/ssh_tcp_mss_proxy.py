"""SSH ProxyCommand relay with a per-connection MSS for a small-MTU tunnel.

Usage: ssh -o 'ProxyCommand=python3 ssh_tcp_mss_proxy.py %h %p' ...
SSH still performs its normal authentication and host-key verification.
"""
import os
import select
import socket
import sys


def relay(host, port, mss=1024):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, mss)
        stream.settimeout(20)
        stream.connect((host, int(port)))
        stream.settimeout(None)
        inputs = [stream, 0]
        while inputs:
            readable, _, _ = select.select(inputs, [], [])
            if stream in readable:
                block = stream.recv(65536)
                if not block:
                    return
                while block:
                    block = block[os.write(1, block):]
            if 0 in readable:
                block = os.read(0, 65536)
                if block:
                    stream.sendall(block)
                else:
                    stream.shutdown(socket.SHUT_WR)
                    inputs.remove(0)


if __name__ == '__main__':
    try:
        relay(sys.argv[1], sys.argv[2])
    except (BrokenPipeError, ConnectionResetError):
        pass
