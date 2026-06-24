"""Ensure forked child processes die with the parent.

Prevents orphaned DataLoader / ProcessPool workers from lingering after
main2.py exits — including when the parent is killed with SIGKILL, which
cannot be caught.

Uses ``PR_SET_PDEATHSIG`` via ``os.register_at_fork``: every forked child
asks the kernel to deliver SIGKILL when its parent dies. Enforced by the
kernel, so it survives parent SIGKILL.

Deliberately does NOT call ``setpgrp`` / install custom signal handlers:
doing so detaches the process from the terminal's foreground group and
breaks Ctrl+C. Default SIGINT/SIGTERM behavior (terminate parent) is
sufficient — PR_SET_PDEATHSIG handles the children.

Linux-only (prctl). No-op on other platforms.
"""
from __future__ import annotations

import os
import signal
import sys

_PR_SET_PDEATHSIG = 1
_installed = False


def _install_pdeathsig_in_child():
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


def install():
    """Install process-guard hooks. Idempotent. Call once, very early."""
    global _installed
    if _installed:
        return
    if not sys.platform.startswith("linux"):
        _installed = True
        return

    try:
        _install_pdeathsig_in_child()  # protect this process (main2.py) from its parent
        os.register_at_fork(after_in_child=_install_pdeathsig_in_child)
    except Exception:
        pass

    _installed = True
