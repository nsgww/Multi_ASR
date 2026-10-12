#!/usr/bin/env python3
"""Board-side diagnostic: join multicast and print received UDP sizes.

Run on the ASR board while hardware (or send_multicast_wav.py) is streaming:

  python scripts/recv_multicast.py --group 224.2.1.209 --port 6020
  python scripts/recv_multicast.py --group 224.2.1.209 --port 6020 --local-ip 192.168.1.178

If this also gets zero packets while tcpdump sees traffic, check:
  sudo sysctl net.ipv4.conf.all.rp_filter
  sudo sysctl -w net.ipv4.conf.all.rp_filter=2
  sudo sysctl -w net.ipv4.conf.wlP4p1s0.rp_filter=2
"""

from __future__ import annotations

import argparse
import socket
import struct
import subprocess
import time


def list_ifaces() -> list[tuple[str, str, int]]:
    index_by_name = {name: idx for idx, name in socket.if_nameindex()}
    out = subprocess.check_output(
        ["ip", "-4", "-o", "addr", "show", "scope", "global"], text=True
    )
    result: list[tuple[str, str, int]] = []
    for line in out.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        ifname = parts[1].rstrip(":")
        ip = parts[parts.index("inet") + 1].split("/", 1)[0]
        result.append((ifname, ip, index_by_name.get(ifname, 0)))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="224.2.1.209")
    p.add_argument("--port", type=int, default=6020)
    p.add_argument("--local-ip", default="0.0.0.0", help="force join on this NIC IP")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--heartbeat", action="store_true", help="send CMD_B every 5s")
    args = p.parse_args()

    ifaces = list_ifaces()
    print("NICs:", ifaces)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.group, args.port))
        print(f"bound {args.group}:{args.port}")
    except OSError as exc:
        sock.bind(("0.0.0.0", args.port))
        print(f"bound 0.0.0.0:{args.port} (group bind failed: {exc})")

    targets = ifaces
    if args.local_ip not in ("", "0.0.0.0"):
        targets = [t for t in ifaces if t[1] == args.local_ip] or [
            ("?", args.local_ip, 0)
        ]

    for ifname, ip, ifindex in targets:
        try:
            if ifindex > 0:
                mreq = struct.pack(
                    "4s4si",
                    socket.inet_aton(args.group),
                    socket.inet_aton(ip if ip != "0.0.0.0" else "0.0.0.0"),
                    ifindex,
                )
            else:
                mreq = struct.pack(
                    "4s4s", socket.inet_aton(args.group), socket.inet_aton(ip)
                )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            print(f"joined {args.group} on {ifname}/{ip}/if{ifindex}")
        except OSError as exc:
            print(f"join failed {ifname}/{ip}: {exc}")

    sock.settimeout(1.0)
    deadline = time.time() + args.seconds
    last_hb = 0.0
    count = 0
    while time.time() < deadline:
        if args.heartbeat and time.time() - last_hb >= 5.0:
            sock.sendto(b"CMD_B", (args.group, args.port))
            last_hb = time.time()
            print("sent CMD_B")
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        count += 1
        print(f"#{count} from {addr} len={len(data)} head={data[:8]!r}")
    print(f"done, packets={count}")
    sock.close()


if __name__ == "__main__":
    main()
