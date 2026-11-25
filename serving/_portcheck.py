"""Who listens on a TCP port? Linux /proc only, no ss/lsof needed (neither is installed in this container).

usage: python3 -I _portcheck.py <port> [pid]
  without pid: exit 0 if something listens on the port, 1 if nothing does
  with pid:    exit 0 if pid or one of its descendants owns the listening socket,
               1 if another process owns it, 2 if nothing listens (yet) or it cannot be told
"""
import os
import sys

LISTEN = "0A"


def listening_inodes(port: int) -> set[str]:
    inodes = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(name) as f:
                next(f)
                for line in f:
                    cols = line.split()
                    local, state, inode = cols[1], cols[3], cols[9]
                    if state == LISTEN and int(local.rsplit(":", 1)[1], 16) == port:
                        inodes.add(inode)
        except OSError:
            pass
    return inodes


def process_tree(root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    tree, todo = set(), [root]
    while todo:
        p = todo.pop()
        if p not in tree:
            tree.add(p)
            todo.extend(children.get(p, []))
    return tree


def socket_inodes(pid: int) -> set[str]:
    out = set()
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith("socket:["):
                out.add(target[8:-1])
    except OSError:
        pass
    return out


def main() -> int:
    port = int(sys.argv[1])
    inodes = listening_inodes(port)
    if len(sys.argv) < 3:
        return 0 if inodes else 1
    if not inodes:
        return 2
    for p in process_tree(int(sys.argv[2])):
        if socket_inodes(p) & inodes:
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
