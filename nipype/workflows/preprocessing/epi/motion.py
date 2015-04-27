# coding: utf-8
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Head-motion correction (HMC) workflows are pipelines to correct
for head motion artifacts in dMRI sequences.
They take a series of diffusion weighted images and rigidly co-register
them to one reference image. Finally, the `b`-matrix is rotated accordingly
[Leemans09]_ making use of the rotation matrix obtained by the registration
algorithm.


A list of rigid transformation matrices is provided, so that transforms
can be chained. This is useful to correct for artifacts with only
one interpolation process (as previously discussed `here
<https://github.com/nipy/nipype/pull/530#issuecomment-14505042>`_),
and also to compute nuisance regressors as proposed by [Yendiki13]_.


.. admonition:: References

  .. [Leemans09] Leemans A, and Jones DK, `The B-matrix must be rotated
    when correcting for subject motion in DTI data
    <http://dx.doi.org/10.1002/mrm.21890>`_,
    Magn Reson Med. 61(6):1336-49. 2009. doi: 10.1002/mrm.21890.

  .. [Yendiki13] Yendiki A et al., `Spurious group differences due to head
    motion in a diffusion MRI study
    <http://dx.doi.org/10.1016/j.neuroimage.2013.11.027>`_.
    Neuroimage. 21(88C):79-90. 2013. doi: 10.1016/j.neuroimage.2013.11.027

"""


def hmc_ants(name='motion_correct'):
    """
    HMC using ANTs registration.


    .. warning:: This workflow rotates the `b`-vectors, so please be advised
      that not all the dicom converters ensure the consistency between the
      resulting nifti orientation and the gradients table (e.g. dcm2nii
      checks it).

    Example
    -------

    >>> from nipype.workflows.dmri.fsl.artifacts import hmc_pipeline
    >>> hmc = hmc_pipeline()
    >>> hmc.inputs.inputnode.in_file = 'diffusion.nii'
    >>> hmc.inputs.inputnode.in_bvec = 'diffusion.bvec'
    >>> hmc.inputs.inputnode.in_bval = 'diffusion.bval'
    >>> hmc.inputs.inputnode.in_mask = 'mask.nii'
    >>> hmc.run() # doctest: +SKIP

    Inputs::

        inputnode.in_file - input dwi file
        inputnode.in_mask - weights mask of reference image (a file with data \
range in [0.0, 1.0], indicating the weight of each voxel when computing the \
metric.
        inputnode.in_bvec - gradients file (b-vectors)
        inputnode.ref_num (optional, default=0) index of the b0 volume that \
should be taken as reference

    Outputs::

        outputnode.out_file - corrected dwi file
        outputnode.out_bvec - rotated gradient vectors table
        outputnode.out_xfms - list of transformation matrices

    """
    import nipype.pipeline.engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces import ants
    from nipype.interfaces import fsl
    from .utils import (insert_mat, rotate_bvecs)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'ref_num', 'in_bvec', 'in_bval', 'in_mask']),
        name='inputnode')

    split = pe.Node(fsl.Split(dimension='t'), name='Split4D')
    selref = pe.Node(niu.Select(), name='ExtractReference')
    selmov = pe.Node(niu.Function(
        function=_extract, output_names=['out'],
        input_names=['in_files', 'ref_num']), name='ExtractMoving')

    reg = _ants_4d()
    insmat = pe.Node(niu.Function(input_names=['inlist', 'volid'],
                                  output_names=['out'], function=insert_mat),
                     name='InsertRefmat')
    rot_bvec = pe.Node(niu.Function(
        function=rotate_bvecs, input_names=['in_bvec', 'in_matrix'],
        output_names=['out_file']), name='Rotate_Bvec')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_bvec', 'out_xfms']), name='outputnode')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  split,      [('in_file', 'in_file')]),
        (split,      selref,     [('out_files', 'inlist')]),
        (inputnode,  selref,      [('ref_num', 'index')]),
        (split,      selmov,     [('out_files', 'in_files')]),
        (inputnode,  selmov,      [('ref_num', 'ref_num')]),
        (inputnode,  reg,        [('in_mask', 'inputnode.ref_mask')]),
        (selref,     reg,        [('out', 'inputnode.reference')]),
        (selmov,     reg,        [('out', 'inputnode.moving')]),
        # (reg,        insmat,     [('outputnode.out_xfms', 'inlist')]),
        # (inputnode,  rot_bvec,   [('in_bvec', 'in_bvec')]),
        # (insmat,     rot_bvec,   [('out', 'in_matrix')]),
        # (rot_bvec,   outputnode, [('out_file', 'out_bvec')]),
        (reg,        outputnode, [('outputnode.out_file', 'out_file')]),
        # (insmat,     outputnode, [('out', 'out_xfms')])
    ])
    return wf


def hmc_flirt(name='motion_correct'):
    """
    HMC using FLIRT from FSL. Search angles have been limited to 4 degrees,
    based on the results of [Yendiki13]_.

    .. warning:: This workflow rotates the `b`-vectors, so please be advised
      that not all the dicom converters ensure the consistency between the
      resulting nifti orientation and the gradients table (e.g. dcm2nii
      checks it).


    Example
    -------

    >>> from nipype.workflows.dmri.fsl.artifacts import hmc_pipeline
    >>> hmc = hmc_pipeline()
    >>> hmc.inputs.inputnode.in_file = 'diffusion.nii'
    >>> hmc.inputs.inputnode.in_bvec = 'diffusion.bvec'
    >>> hmc.inputs.inputnode.in_bval = 'diffusion.bval'
    >>> hmc.inputs.inputnode.in_mask = 'mask.nii'
    >>> hmc.run() # doctest: +SKIP

    Inputs::

        inputnode.in_file - input dwi file
        inputnode.in_mask - weights mask of reference image (a file with data \
range in [0.0, 1.0], indicating the weight of each voxel when computing the \
metric.
        inputnode.in_bvec - gradients file (b-vectors)
        inputnode.ref_num (optional, default=0) index of the b0 volume that \
should be taken as reference

    Outputs::

        outputnode.out_file - corrected dwi file
        outputnode.out_bvec - rotated gradient vectors table
        outputnode.out_xfms - list of transformation matrices

    """
    import nipype.pipeline.engine as pe
    from nipype.interfaces import utility as niu
    from .utils import (flirt_4d, insert_mat, rotate_bvecs)
    from nipype.workflows.data import get_flirt_schedule

    params = dict(dof=6, bgvalue=0, save_log=True, no_search=True,
                  # cost='mutualinfo', cost_func='mutualinfo', bins=64,
                  schedule=get_flirt_schedule('hmc'))

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'ref_num', 'in_bvec', 'in_bval', 'in_mask']),
        name='inputnode')
    split = pe.Node(niu.Function(
        function=hmc_split, output_names=['out_ref', 'out_mov'],
        input_names=['in_file', 'ref_num']), name='SplitDWI')
    flirt = flirt_4d(flirt_param=params)
    insmat = pe.Node(niu.Function(input_names=['inlist', 'volid'],
                                  output_names=['out'], function=insert_mat),
                     name='InsertRefmat')
    rot_bvec = pe.Node(niu.Function(
        function=rotate_bvecs, input_names=['in_bvec', 'in_matrix'],
        output_names=['out_file']), name='Rotate_Bvec')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_bvec', 'out_xfms']), name='outputnode')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,     split,   [('in_file', 'in_file'),
                                  ('ref_num', 'ref_num')]),
        (inputnode,  flirt,      [('in_mask', 'inputnode.ref_mask')]),
        (split,      flirt,      [('out_ref', 'inputnode.reference'),
                                  ('out_mov', 'inputnode.in_file')]),
        (flirt,      insmat,     [('outputnode.out_xfms', 'inlist')]),
        (inputnode,  rot_bvec,   [('in_bvec', 'in_bvec')]),
        (insmat,     rot_bvec,   [('out', 'in_matrix')]),
        (rot_bvec,   outputnode, [('out_file', 'out_bvec')]),
        (flirt,      outputnode, [('outputnode.out_file', 'out_file')]),
        (insmat,     outputnode, [('out', 'out_xfms')])
    ])
    return wf


def hmc_split(in_file, ref_num=0, lowbval=5.0):
    """
    Selects the reference and moving volumes from a dwi dataset
    for the purpose of HMC.
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    from nipype.interfaces.base import isdefined

    im = nb.load(in_file)
    data = im.get_data()
    hdr = im.get_header().copy()

    volid = int(ref_num)

    refdata = np.squeeze(data[..., volid])

    if volid == 0:
        movdata = data[..., 1:]
    elif volid == (data.shape[-1] - 1):
        movdata = data[..., :-1]
    else:
        movdata = np.concatenate((data[..., :volid], data[..., (volid + 1):]),
                                 axis=3)

    out_ref = op.abspath('hmc_ref.nii.gz')
    out_mov = op.abspath('hmc_mov.nii.gz')

    hdr.set_data_shape(refdata.shape)
    nb.Nifti1Image(refdata, im.get_affine(), hdr).to_filename(out_ref)

    hdr.set_data_shape(movdata.shape)
    nb.Nifti1Image(movdata, im.get_affine(), hdr).to_filename(out_mov)

    return (out_ref, out_mov)


def _ants_4d(name='DWICoregistration'):
    """
    Generates a workflow for rigid registration of dwi volumes
    using ants
    """
    import nipype.pipeline.engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces import freesurfer as fs
    from nipype.interfaces import ants
    from nipype.interfaces import fsl
    from .utils import enhance

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['reference', 'moving', 'ref_mask']),
        name='inputnode')

    dilate = pe.Node(fsl.maths.MathsCommand(
        nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='Bias')
    refth = pe.Node(fsl.Threshold(thresh=0.0), name='ReferenceThres')

    reg = pe.MapNode(
        ants.Registration(output_warped_image=True),
        iterfield=['moving_image'], name="dwi_to_b0")

    reg.inputs.transforms = ['Rigid']
    reg.inputs.transform_parameters = [(0.05,)]
    reg.inputs.number_of_iterations = [[500]]
    reg.inputs.dimension = 3
    reg.inputs.metric = ['Mattes']
    reg.inputs.metric_weight = [1.0]
    reg.inputs.radius_or_number_of_bins = [64]
    reg.inputs.sampling_strategy = ['Random']
    reg.inputs.sampling_percentage = [0.2]
    reg.inputs.convergence_threshold = [1.e-7]
    reg.inputs.convergence_window_size = [20]
    reg.inputs.smoothing_sigmas = [[2.0]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.shrink_factors = [[1]]
    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [True]
    reg.inputs.initial_moving_transform_com = 0
    reg.inputs.winsorize_lower_quantile = .04
    reg.inputs.winsorize_upper_quantile = .85

    thres = pe.MapNode(fsl.Threshold(thresh=0.0), iterfield=['in_file'],
                       name='RemoveNegative')
    merge = pe.Node(fsl.Merge(dimension='t'), name='MergeDWIs')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_xfms']), name='outputnode')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  dilate,     [('ref_mask', 'in_file')]),
        (inputnode,  n4,         [('reference', 'input_image'),
                                  ('ref_mask', 'mask_image')]),
        (dilate,     reg,        [('out_file', 'fixed_image_mask'),
                                  ('out_file', 'moving_image_mask')]),
        (n4,         refth,      [('output_image', 'in_file')]),
        (refth,      reg,        [('out_file', 'fixed_image')]),
        (inputnode,  reg,        [('moving', 'moving_image')]),
        (reg,        thres,      [('warped_image', 'in_file')]),
        (thres,      merge,      [('out_file', 'in_files')]),
        (merge,      outputnode, [('merged_file', 'out_file')]),
        (reg,        outputnode, [('forward_transforms', 'out_xfms')])
    ])
    return wf


def _extract(in_files, ref_num=0):
    import numpy as np
    out = list(in_files)
    del out[ref_num]
    return out
