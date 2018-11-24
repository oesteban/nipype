# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utility functions
"""
from __future__ import (print_function, unicode_literals, division,
                        absolute_import)
import os
import sys
import gc
import errno
import select
import locale
import datetime
from subprocess import Popen, STDOUT, PIPE
from builtins import range, object

from .filemanip import canonicalize_env, read_stream

from .. import logging


iflogger = logging.getLogger('nipype.interface')


class Stream(object):
    """Function to capture stdout and stderr streams with timestamps

    stackoverflow.com/questions/4984549/merge-and-sync-stdout-and-stderr/5188359
    """

    def __init__(self, name, impl):
        self._name = name
        self._impl = impl
        self._buf = ''
        self._rows = []
        self._lastidx = 0
        self.default_encoding = locale.getdefaultlocale()[1] or 'UTF-8'

    def fileno(self):
        "Pass-through for file descriptor."
        return self._impl.fileno()

    def read(self, drain=0):
        "Read from the file descriptor. If 'drain' set, read until EOF."
        while self._read(drain) is not None:
            if not drain:
                break

    def _read(self, drain):
        "Read from the file descriptor"
        fd = self.fileno()
        buf = os.read(fd, 4096).decode(self.default_encoding)
        if not buf and not self._buf:
            return None
        if '\n' not in buf:
            if not drain:
                self._buf += buf
                return []

        # prepend any data previously read, then split into lines and format
        buf = self._buf + buf
        if '\n' in buf:
            tmp, rest = buf.rsplit('\n', 1)
        else:
            tmp = buf
            rest = None
        self._buf = rest
        now = datetime.datetime.now().isoformat()
        rows = tmp.split('\n')
        self._rows += [(now, '%s %s:%s' % (self._name, now, r), r)
                       for r in rows]
        for idx in range(self._lastidx, len(self._rows)):
            iflogger.info(self._rows[idx][1])
        self._lastidx = len(self._rows)


def run_command(runtime, output=None):
    """Run a command, read stdout and stderr, prefix with timestamp.

    The returned runtime contains a merged stdout+stderr log with timestamps
    """

    # Init variables
    cmdline = runtime.cmdline
    env = canonicalize_env(runtime.environ)

    # Open files to redirect output to
    outfile = os.path.join(runtime.cwd, 'nipype.out')
    stdout = open(outfile, 'wb')

    errfile = os.path.join(runtime.cwd, 'nipype.err')
    stderr = open(errfile, 'wb')

    # Fork new process
    proc = Popen(
        cmdline,
        stdout=stdout.fileno(),
        stderr=stderr.fileno(),
        shell=True,
        cwd=runtime.cwd,
        env=env,
        close_fds=(not sys.platform.startswith('win')),
    )

    # if output == 'stream':
    # Start a thread that runs os.stat on logfiles and
    # prints to log if something new is found.

    proc.wait()
    runtime.returncode = proc.returncode
    try:
        proc.terminate()  # Ensure we are done
    except OSError as error:
        # Python 2 raises when the process is already gone
        if error.errno != errno.ESRCH:
            raise

    # Close files
    stdout.close()
    stderr.close()

    # Dereference & force GC for a cleanup
    del proc
    del stdout
    del stderr
    gc.collect()

    runtime.stderr = errfile
    runtime.stdout = outfile
    return runtime
