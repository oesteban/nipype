# coding: utf-8
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""The :py:mod:`nipype.interfaces.elastix` provides the interface to
the elastix registration software.

.. note:: http://elastix.isi.uu.nl/


"""

from ..base import (CommandLine, CommandLineInputSpec, isdefined,
                    TraitedSpec, File, Directory, traits, InputMultiPath)
from ... import logging
logger = logging.getLogger('interface')

from multiprocessing import cpu_count


class ElastixBaseInputSpec(CommandLineInputSpec):
    output_path = Directory('./', exists=True, mandatory=True, usedefault=True,
                            argstr='-out %s', desc='output directory')
    num_threads = traits.Int(cpu_count(), argstr='-threads %01d', usedefault=True,
                             desc='set the maximum number of threads of elastix')


class ElastixBaseInterface(CommandLine):
    _singleworker = True

    def _format_arg(self, name, trait_spec, value):
        if name == 'num_threads' and value > 1:
            self._singleworker = False

        return super(ElastixBaseInterface, self)._format_arg(name, trait_spec, value)