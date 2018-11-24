# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous file manipulation functions
"""
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)


class NipypeInterfaceError(Exception):
    """Custom error for interfaces"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '{}'.format(self.value)


class NipypeRuntimeError(Exception):
    """ Customized error for interfaces that failed in runtime """

    def __init__(self, runtime, message=None):

        if message is not None:
            message = '\n%s\n' % message
        else:
            message = ''

        allerrors = """\
Nipype interface failed with exit code: {returncode}
{message}
Command
-------

{cmdline}

Standard output
---------------

{stdout}


Standard error
--------------

{stderr}

""".format(**runtime.dictcopy(), message=message)

        # Call the base class constructor with the parameters it needs
        super(NipypeRuntimeError, self).__init__(allerrors)
