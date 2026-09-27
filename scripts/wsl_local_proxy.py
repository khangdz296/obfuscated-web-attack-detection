"""Loopback-only TCP proxy used to expose the WSL-hosted DVWA to Windows.

The listener binds only to 127.0.0.1 so the intentionally vulnerable DVWA
instance is not reachable from the LAN.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        with socket.create_connection(
            (self.server.target_host, self.server.target_port), timeout=10
        ) as upstream:
            sockets = [self.request, upstream]
            while True:
                readable, _, _ = select.select(sockets, [], [], 30)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, listen, handler, target_host: str, target_port: int):
        super().__init__(listen, handler)
        self.target_host = target_host
        self.target_port = target_port


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, default=4281)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, default=4280)
    args = parser.parse_args()

    with ProxyServer(
        ("127.0.0.1", args.listen_port),
        ProxyHandler,
        args.target_host,
        args.target_port,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
