# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
   Change directory to provide relative paths for doctests
   >>> import os
   >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
   >>> datadir = os.path.realpath(os.path.join(filepath, '../testing/data'))
   >>> os.chdir(datadir)

Nipype interface for PETPVC.

PETPVC is a software from the Nuclear Medicine Department
of the UCL University Hospital, London, UK.

Its source code is here: https://github.com/UCL/PETPVC

The methods that it implement are explained here:
K. Erlandsson, I. Buvat, P. H. Pretorius, B. A. Thomas, and B. F. Hutton,
"A review of partial volume correction techniques for emission tomography
and their applications in neurology, cardiology and oncology," Phys. Med.
Biol., vol. 57, no. 21, p. R119, 2012.

There is a publication waiting to be accepted for this software tool.


Its command line help shows this:

   -i --input < filename >
      = PET image file
   -o --output < filename >
      = Output file
   [ -m --mask < filename > ]
      = Mask image file
   -p --pvc < keyword >
      = Desired PVC method
   -x < X >
      = The full-width at half maximum in mm along x-axis
   -y < Y >
      = The full-width at half maximum in mm along y-axis
   -z < Z >
      = The full-width at half maximum in mm along z-axis
   [ -d --debug ]
      = Prints debug information
   [ -n --iter [ Val ] ]
      = Number of iterations
        With: Val (Default = 10)
   [ -k [ Val ] ]
      = Number of deconvolution iterations
        With: Val (Default = 10)
   [ -a --alpha [ aval ] ]
      = Alpha value
        With: aval (Default = 1.5)
   [ -s --stop [ stopval ] ]
      = Stopping criterion
        With: stopval (Default = 0.01)

----------------------------------------------
Technique - keyword

Geometric transfer matrix - "GTM"
Labbe approach - "LABBE"
Richardson-Lucy - "RL"
Van-Cittert - "VC"
Region-based voxel-wise correction - "RBV"
RBV with Labbe - "LABBE+RBV"
RBV with Van-Cittert - "RBV+VC"
RBV with Richardson-Lucy - "RBV+RL"
RBV with Labbe and Van-Cittert - "LABBE+RBV+VC"
RBV with Labbe and Richardson-Lucy- "LABBE+RBV+RL"
Multi-target correction - "MTC"
MTC with Labbe - "LABBE+MTC"
MTC with Van-Cittert - "MTC+VC"
MTC with Richardson-Lucy - "MTC+RL"
MTC with Labbe and Van-Cittert - "LABBE+MTC+VC"
MTC with Labbe and Richardson-Lucy- "LABBE+MTC+RL"
Iterative Yang - "IY"
Iterative Yang with Van-Cittert - "IY+VC"
Iterative Yang with Richardson-Lucy - "IY+RL"
Muller Gartner - "MG"
Muller Gartner with Van-Cittert - "MG+VC"
Muller Gartner with Richardson-Lucy - "MG+RL"

"""
from __future__ import print_function
from __future__ import division

import os
import warnings

from nipype.interfaces.base import (
    TraitedSpec,
    CommandLineInputSpec,
    CommandLine,
    File,
    isdefined,
    traits,
)

warn = warnings.warn

pvc_methods = ['GTM',
               'IY',
               'IY+RL',
               'IY+VC',
               'LABBE',
               'LABBE+MTC',
               'LABBE+MTC+RL',
               'LABBE+MTC+VC',
               'LABBE+RBV',
               'LABBE+RBV+RL',
               'LABBE+RBV+VC',
               'MG',
               'MG+RL',
               'MG+VC',
               'MTC',
               'MTC+RL',
               'MTC+VC',
               'RBV',
               'RBV+RL',
               'RBV+VC',
               'RL',
               'VC']


class PETPVCInputSpec(CommandLineInputSpec):
    in_file   = File(desc="PET image file", exists=True, mandatory=True, argstr="-i %s")
    out_file  = File(desc="Output file", genfile=True, hash_files=False, argstr="-o %s")
    mask_file = File(desc="Mask image file", exists=True, mandatory=True, argstr="-m %s")
    pvc       = traits.Enum(pvc_methods, desc="Desired PVC method", mandatory=True, argstr="-p %s")
    fwhm_x    = traits.Float(desc="The full-width at half maximum in mm along x-axis", mandatory=True, argstr="-x %.4f")
    fwhm_y    = traits.Float(desc="The full-width at half maximum in mm along y-axis", mandatory=True, argstr="-y %.4f")
    fwhm_z    = traits.Float(desc="The full-width at half maximum in mm along z-axis", mandatory=True, argstr="-z %.4f")
    debug     = traits.Bool (desc="Prints debug information", usedefault=True, default_value=False, argstr="-d")
    n_iter    = traits.Int  (desc="Number of iterations", default_value=10, argstr="-n %d")
    n_deconv  = traits.Int  (desc="Number of deconvolution iterations", default_value=10, argstr="-k %d")
    alpha     = traits.Float(desc="Alpha value", default_value=1.5, argstr="-a %.4f")
    stop_crit = traits.Float(desc="Stopping criterion", default_value=0.01, argstr="-a %.4f")


class PETPVCOutputSpec(TraitedSpec):
    out_file = File(desc = "Output file")


class PETPVC(CommandLine):
    """ Use PETPVC for partial volume correction of PET images.

    Examples
    --------
    >>> from ..testing import example_data
    >>> #TODO get data for PETPVC
    >>> pvc = PETPVC()
    >>> pvc.inputs.in_file   = 'pet.nii.gz'
    >>> pvc.inputs.mask_file = 'tissues.nii.gz'
    >>> pvc.inputs.out_file  = 'pet_pvc_rbv.nii.gz'
    >>> pvc.inputs.pvc = 'RBV'
    >>> pvc.inputs.fwhm_x = 2.0
    >>> pvc.inputs.fwhm_y = 2.0
    >>> pvc.inputs.fwhm_z = 2.0
    >>> outs = pvc.run() #doctest: +SKIP
    """
    input_spec = PETPVCInputSpec
    output_spec = PETPVCOutputSpec
    _cmd = 'petpvc'

    def _post_run(self):
        
        self.outputs.out_file = self.inputs.out_file
        if not isdefined(self.outputs.out_file):
            method_name = self.inputs.pvc.lower()
            self.outputs.out_file = self._gen_fname(self.inputs.in_file,
                                                  suffix='_{}_pvc'.format(method_name))

        self.outputs.out_file = os.path.abspath(self.outputs.out_file)
        
    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True,
                   ext='.nii.gz'):
        """Generate a filename based on the given parameters.

        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extentions specified in
        <instance>intputs.output_type.

        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        change_ext : bool
            Flag to change the filename extension to the given `ext`.
            (Default is False)

        Returns
        -------
        fname : str
            New filename based on given parameters.

        """
        from nipype.utils.filemanip import fname_presuffix

        if basename == '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if change_ext:
            if suffix:
                suffix = ''.join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ''
        fname = fname_presuffix(basename, suffix=suffix,
                                use_ext=False, newpath=cwd)
        return fname

    def _gen_filename(self, name):
        if name == 'out_file':
            return self.outputs.out_file
        return None