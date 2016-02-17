# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The fsl module provides classes for interfacing with the `FSL
<http://www.fmrib.ox.ac.uk/fsl/index.html>`_ command line tools.  This
was written to work with FSL version 4.1.4.

These are the base tools for working with FSL.
Preprocessing tools are found in fsl/preprocess.py
Model tools are found in fsl/model.py
DTI tools are found in fsl/dti.py

XXX Make this doc current!

Currently these tools are supported:

* BET v2.1: brain extraction
* FAST v4.1: segmentation and bias correction
* FLIRT v5.5: linear registration
* MCFLIRT: motion correction
* FNIRT v1.0: non-linear warp

Examples
--------
See the docstrings of the individual classes for examples.

"""

import os
from glob import glob
from builtins import object
from ..base import traits, CommandLine, CommandLineInputSpec
from ... import logging

IFLOGGER = logging.getLogger('interface')

FSLDIR = os.getenv('FSLDIR')
if FSLDIR is None:
    IFLOGGER.warn('FSLDIR environment variable is not set')

FSLOUTPUTTYPE = os.getenv('FSLOUTPUTTYPE')
if FSLOUTPUTTYPE is None:
    IFLOGGER.warn('FSLOUTPUTTYPE environment variable is not set, using NIFTI')
    FSLOUTPUTTYPE = 'NIFTI'

class Info(object):
    """Handle fsl output type and version information.

    version refers to the version of fsl on the system

    output type refers to the type of file fsl defaults to writing
    eg, NIFTI, NIFTI_GZ

    """

    ftypes = {'NIFTI': '.nii',
              'NIFTI_PAIR': '.img',
              'NIFTI_GZ': '.nii.gz',
              'NIFTI_PAIR_GZ': '.img.gz'}

    @staticmethod
    def version():
        """Check for fsl version on system

        Parameters
        ----------
        None

        Returns
        -------
        version : str
           Version number as string or None if FSL not found

        """
        # find which fsl being used....and get version from
        # /path/to/fsl/etc/fslversion
        out = None
        if FSLDIR is not None:
            with open('%s/etc/fslversion' % FSLDIR, 'r') as vfile:
                out = vfile.read().strip('\n')
        return out

    @classmethod
    def output_type_to_ext(cls, output_type):
        """Get the file extension for the given output type.

        Parameters
        ----------
        output_type : {'NIFTI', 'NIFTI_GZ', 'NIFTI_PAIR', 'NIFTI_PAIR_GZ'}
            String specifying the output type.

        Returns
        -------
        extension : str
            The file extension for the output type.
        """

        try:
            return cls.ftypes[output_type]
        except KeyError:
            msg = 'Invalid FSLOUTPUTTYPE: ', output_type
            raise KeyError(msg)

    @classmethod
    def output_type(cls):
        """Get the global FSL output file type FSLOUTPUTTYPE.

        This returns the value of the environment variable
        FSLOUTPUTTYPE.  An exception is raised if it is not defined.

        Returns
        -------
        fsl_ftype : string
            Represents the current environment setting of FSLOUTPUTTYPE
        """
        return FSLOUTPUTTYPE

    @staticmethod
    def standard_image(img_name=None):
        """Grab an image from the standard location.

        Returns a list of standard images if called without arguments.

        Could be made more fancy to allow for more relocatability"""
        try:
            fsldir = os.environ['FSLDIR']
        except KeyError:
            raise Exception('FSL environment variables not set')
        stdpath = os.path.join(fsldir, 'data', 'standard')
        if img_name is None:
            return [filename.replace(stdpath + '/', '')
                    for filename in glob(os.path.join(stdpath, '*nii*'))]
        return os.path.join(stdpath, img_name)


class FSLCommandInputSpec(CommandLineInputSpec):
    """
    Base Input Specification for all FSL Commands

    All command support specifying FSLOUTPUTTYPE dynamically
    via output_type.

    Example
    -------
    fsl.ExtractRoi(tmin=42, tsize=1, output_type='NIFTI')
    """
    output_type = traits.Enum(FSLOUTPUTTYPE, list(Info.ftypes.keys()), usedefault=True,
                              desc='FSL output type')

    def _overload_extension(self, value, name=None):
        return value + Info.output_type_to_ext(self.output_type)


class FSLCommand(CommandLine):  # pylint: disable=W0223
    """Base support for FSL commands."""

    input_spec = FSLCommandInputSpec

    def __init__(self, **inputs):
        super(FSLCommand, self).__init__(**inputs)
        self.inputs.on_trait_change(self._output_update, 'output_type')
        self._output_type = FSLOUTPUTTYPE
        self.inputs.environ.update({'FSLOUTPUTTYPE': FSLOUTPUTTYPE})

    def _output_update(self):
        self._output_type = self.inputs.output_type
        self.inputs.environ.update({'FSLOUTPUTTYPE': self.inputs.output_type})

    def _get_ext(self):
        return Info.output_type_to_ext(self.input_spec.output_type)

    def _gen_fname(self, basename, out_path=None, suffix=''):
        if out_path is None:
            out_path = os.getcwd()
        return os.path.join(out_path, basename + suffix + self._get_ext())

    @classmethod
    def set_default_output_type(cls, output_type):
        """Set the default output type for FSL classes.

        This method is used to set the default output type for all fSL
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.output_type.
        """

        if output_type in Info.ftypes:
            cls._output_type = output_type
        else:
            raise AttributeError('Invalid FSL output_type: %s' % output_type)

    @property
    def version(self):
        return Info.version()


def no_fsl():
    """Checks if FSL is NOT installed
    used with skipif to skip tests that will
    fail if FSL is not installed"""
    return Info.version() is None

def check_fsl():
    """Same as the previous. One of these should disappear """
    return Info.version() is not None

def no_fsl_course_data():
    """check if fsl_course data is present"""
    if os.getenv('FSL_COURSE_DATA') is None:
        return False
    return os.path.isdir(os.path.abspath(os.getenv('FSL_COURSE_DATA')))
