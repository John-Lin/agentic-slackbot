import socket

import pytest


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.fixture(autouse=True)
def block_outbound_network(monkeypatch):
    allowed_hosts = {"127.0.0.1", "::1", "localhost"}
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and address and address[0] in allowed_hosts:
            return original_connect(sock, address)
        raise RuntimeError("Network access blocked during tests")

    def guarded_connect_ex(sock, address):
        if isinstance(address, tuple) and address and address[0] in allowed_hosts:
            return original_connect_ex(sock, address)
        raise RuntimeError("Network access blocked during tests")

    def guarded_create_connection(address, *args, **kwargs):
        if isinstance(address, tuple) and address and address[0] in allowed_hosts:
            return original_create_connection(address, *args, **kwargs)
        raise RuntimeError("Network access blocked during tests")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
