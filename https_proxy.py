#!/usr/bin/env python3
"""HTTPS reverse proxy: terminates TLS, forwards to memoria (HTTP) on :8081."""
from __future__ import annotations

import asyncio
import os
import ssl
import sys

import httpx

MEMORIA_UPSTREAM = "http://127.0.0.1:8081"
PROXY_PORT = 8443
CERT_FILE = os.path.join(os.path.dirname(__file__), "server.crt")
KEY_FILE = os.path.join(os.path.dirname(__file__), "server.key")
MAX_BODY = 64 * 1024 * 1024

client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=15)
        if not request_line:
            writer.close()
            return
        try:
            method, target, _version = request_line.decode("latin-1").strip().split(" ", 2)
        except ValueError:
            writer.close()
            return

        content_length = 0
        headers: list[tuple[str, str]] = []
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=15)
            if line in (b"\r\n", b"\n", b""):
                break
            try:
                text = line.decode("latin-1").strip()
                key, _, value = text.partition(":")
                if value.strip() == "":
                    continue
                headers.append((key.strip(), value.strip()))
                if key.lower() == "content-length":
                    content_length = int(value.strip())
            except (ValueError, UnicodeDecodeError):
                continue

        body = b""
        if content_length > 0:
            content_length = min(content_length, MAX_BODY)
            while len(body) < content_length:
                chunk = await reader.read(content_length - len(body))
                if not chunk:
                    break
                body += chunk

        url = MEMORIA_UPSTREAM + target
        response = await client.request(method, url, headers=headers, content=body or None)

        reason = response.reason_phrase or ""
        writer.write(f"HTTP/1.1 {response.status_code} {reason}\r\n".encode("latin-1"))
        hop_by_hop = {
            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
            "te", "trailers", "transfer-encoding", "upgrade", "content-length",
        }
        for key, value in response.headers.items():
            if key.lower() not in hop_by_hop:
                writer.write(f"{key}: {value}\r\n".encode("latin-1"))
        writer.write(f"Content-Length: {len(response.content)}\r\n".encode("latin-1"))
        writer.write(b"\r\n")
        writer.write(response.content)
        await writer.drain()
    except asyncio.TimeoutError:
        pass
    except Exception as exc:
        print(f"[https-proxy] error for {peer}: {exc!r}", file=sys.stderr)
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def main() -> None:
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(CERT_FILE, KEY_FILE)

    server = await asyncio.start_server(handle, "127.0.0.1", PROXY_PORT, ssl=ssl_ctx)
    addr = server.sockets[0].getsockname()
    print(f"[https-proxy] listening on {addr[0]}:{addr[1]} (TLS)")
    print(f"[https-proxy] upstream: {MEMORIA_UPSTREAM}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        asyncio.run(client.aclose())