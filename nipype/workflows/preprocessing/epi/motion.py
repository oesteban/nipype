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
    reg = _ants_4d()
    rot_bvec = _ants_rotate_bvecs()

    merge = pe.Node(fsl.Merge(dimension='t'), name='MergeDWI')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_bvec', 'out_tran']), name='outputnode')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  split,      [('in_file', 'in_file')]),
        (inputnode,  reg,        [('ref_num', 'inputnode.ref_num'),
                                  ('in_mask', 'inputnode.ref_mask')]),
        (split,      reg,        [('out_files', 'inputnode.in_files')]),
        (reg,        merge,      [('outputnode.out_files', 'in_files')]),
        (inputnode,  rot_bvec,   [('in_bvec', 'inputnode.in_bvec'),
                                  ('ref_num', 'inputnode.ref_num')]),
        (reg,        rot_bvec,   [
            ('outputnode.out_xfms', 'inputnode.in_xfms')]),
        (rot_bvec,   outputnode, [('outputnode.out_bvec', 'out_bvec'),
                                  ('outputnode.out_tran', 'out_tran')]),
        (merge,      outputnode, [('merged_file', 'out_file')])
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


def _ants_4d(name='DWICoregistration', denoise=False):
    """
    Generates a workflow for rigid registration of dwi volumes
    using ants
    """
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.interfaces import freesurfer as fs
    from nipype.interfaces import ants
    from nipype.interfaces import fsl
    from nipype.interfaces.dipy import Denoise
    from nipype.interfaces.io import JSONFileGrabber
    from nipype.workflows.data import get_ants_hmc

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_files', 'ref_mask', 'ref_num']), name='inputnode')

    settings = pe.Node(
        JSONFileGrabber(in_file=get_ants_hmc()), name='Settings')

    enh = pe.MapNode(niu.Function(
        function=_enhance, input_names=['in_file'], output_names=['out_file']),
        name='Enhance', iterfield=['in_file'])

    dilate = pe.Node(fsl.maths.MathsCommand(
        nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')

    selref0 = pe.Node(niu.Select(), name='SelectRef0')
    sch0 = pe.Node(niu.Function(
        function=_ants_schedule, input_names=['in_files', 'ref_num'],
        output_names=['fixed_image', 'moving_image']), name='Schedule0')

    reg = pe.MapNode(
        ants.Registration(output_warped_image=True, dimension=3,
                          collapse_output_transforms=True),
        iterfield=['fixed_image', 'moving_image'], name="Registration4D")

    sortxfm0 = pe.Node(niu.Function(
        function=_ants_concatxfm, input_names=['in_files', 'ref_num'],
        output_names=['out_files']), name='SortXFMsFW')

    selref1 = pe.Node(niu.Select(), name='SelectRef1')
    sch1 = pe.Node(niu.Function(
        function=_ants_schedule, input_names=['in_files', 'ref_num'],
        output_names=['fixed_image', 'moving_image']), name='Schedule1')

    xfm = pe.MapNode(
        ants.ApplyTransforms(dimension=3),
        iterfield=['transforms', 'input_image'], name="ApplyTransforms")

    insref = pe.Node(niu.Function(
        function=_insertref, input_names=['reference', 'in_files', 'pos'],
        output_names=['out_files']), name='InsertRef')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_files', 'out_xfms']),
        name='outputnode')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  dilate,     [('ref_mask', 'in_file')]),
        (inputnode,  enh,        [('in_files', 'in_file')]),
        (inputnode,  selref0,    [('ref_num', 'index')]),
        (enh,        selref0,    [('out_file', 'inlist')]),
        (inputnode,  sch0,       [('ref_num', 'ref_num')]),
        (enh,        sch0,       [('out_file', 'in_files')]),
        (inputnode,  sortxfm0,   [('ref_num', 'ref_num')]),
        (dilate,     reg,        [('out_file', 'fixed_image_mask')]),
        (sch0,       reg,        [('fixed_image', 'fixed_image'),
                                  ('moving_image', 'moving_image')]),
        (reg,        sortxfm0,   [('forward_transforms', 'in_files')]),

        (sortxfm0,   outputnode, [('out_files', 'out_xfms')]),
        (inputnode,  selref1,    [('in_files', 'inlist'),
                                  ('ref_num', 'index')]),
        (inputnode,  sch1,       [('in_files', 'in_files'),
                                  ('ref_num', 'ref_num')]),
        (selref1,    xfm,        [('out', 'reference_image')]),
        (sch1,       xfm,        [('moving_image', 'input_image')]),
        (sortxfm0,   xfm,        [('out_files', 'transforms')]),
        (xfm,        insref,     [('output_image', 'in_files')]),
        (inputnode,  insref,     [('ref_num', 'pos')]),
        (selref1,    insref,     [('out', 'reference')]),
        (insref,     outputnode, [('out_files', 'out_files')])
    ])

    # connect settings to registration
    wf.connect([
        (settings,   reg,        [
            ('transforms', 'transforms'),
            ('transform_parameters', 'transform_parameters'),
            ('number_of_iterations', 'number_of_iterations'),
            ('metric', 'metric'),
            ('metric_weight', 'metric_weight'),
            ('sampling_strategy', 'sampling_strategy'),
            ('sampling_percentage', 'sampling_percentage'),
            ('convergence_threshold', 'convergence_threshold'),
            ('convergence_window_size', 'convergence_window_size'),
            ('smoothing_sigmas', 'smoothing_sigmas'),
            ('sigma_units', 'sigma_units'),
            ('shrink_factors', 'shrink_factors'),
            ('use_estimate_learning_rate_once',
             'use_estimate_learning_rate_once'),
            ('use_histogram_matching', 'use_histogram_matching'),
            ('initial_moving_transform_com', 'initial_moving_transform_com'),
            ('radius_or_number_of_bins', 'radius_or_number_of_bins'),
            ('num_threads', 'num_threads')
            # ('winsorize_upper_quantile', 'winsorize_upper_quantile')
        ])
    ])
    return wf


def _enhance(in_file):
    import nibabel as nb
    import numpy as np
    import os.path as op

    im = nb.load(in_file)
    data = im.get_data()
    data[data < 0] = 0
    thres = np.percentile(data[data > 0.0], 99.98)
    data[data > thres] = thres

    out_file = op.abspath('enhanced.nii.gz')
    nb.Nifti1Image(data.astype(np.float32), im.get_affine(),
                   im.get_header()).to_filename(out_file)
    return out_file


def _ants_schedule(in_files, ref_num=0):
    fixed_image = list(in_files)
    moving_image = list(in_files)

    if ref_num == 0:
        fixed_image = fixed_image[:-1]
        moving_image = moving_image[1:]
    elif ref_num == len(in_files) - 1:
        fixed_image = list(reversed(fixed_image[:-1]))
        moving_image = list(reversed(moving_image[1:]))
    else:
        fixed_image.insert(ref_num, fixed_image[ref_num])
        fixed_image = fixed_image[1:-1]
        moving_image = moving_image[:ref_num] + \
            moving_image[ref_num + 1:]
    return fixed_image, moving_image


def _ants_concatxfm(in_files, ref_num=0):
    import numpy as np
    out_files = []

    if ref_num == 0:
        for i in range(len(in_files)):
            out_files.append(
                np.ravel(list(reversed(in_files[:i + 1]))).tolist())
    elif ref_num == len(in_files) - 1:
        for i in range(len(in_files)):
            f = list(reversed(in_files))
            out_files = _ants_concatxfm(f)
    return out_files


def _ants_rotate_bvecs(name='BmatrixRotation'):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from nipype.algorithms import misc as nam
    from nipype.interfaces import ants

    def _reverse(inlist):
        return [list(reversed(sub)) for sub in inlist]

    def _gen_flags(inlist):
        return [[True] * len(el) for el in inlist]

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_bvec', 'in_xfms', 'ref_num']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_bvec', 'out_tran']), name='outputnode')

    tocsv = pe.Node(niu.Function(
        function=_bvecs2csv, input_names=['in_bvec'],
        output_names=['out_files']), name='Bvecs2CSV')

    xfm = pe.MapNode(ants.ApplyTransformsToPoints(
        dimension=3), iterfield=['input_file', 'transforms',
                                 'invert_transform_flags'],
        name='RotateBvec')
    tobvec = pe.Node(niu.Function(
        function=_csv2bvecs, input_names=['inlist', 'ref_num'],
        output_names=['out_bvec', 'out_tran']), name='GenBmatrix')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  tocsv,      [('in_bvec', 'in_bvec')]),
        (tocsv,      xfm,        [('out_files', 'input_file')]),
        (inputnode,  xfm,        [(('in_xfms', _reverse), 'transforms'),
                                  (('in_xfms', _gen_flags),
                                   'invert_transform_flags')]),
        (xfm,        tobvec,     [('output_file', 'inlist')]),
        (inputnode,  tobvec,     [('ref_num', 'ref_num')]),
        (tobvec,     outputnode, [('out_bvec', 'out_bvec'),
                                  ('out_tran', 'out_tran')])
    ])

    return wf


def _bvecs2csv(in_bvec):
    import numpy as np
    import os.path as op
    bvec = np.loadtxt(in_bvec).T
    out_files = []
    for i in range(1, len(bvec)):
        out_file = op.abspath('bvec%04d.csv' % i)
        bv = np.array([[0, 0, 0], bvec[i]])
        np.savetxt(
            out_file, bv, delimiter=',', header='x,y,z')
        out_files.append(out_file)
    return out_files


def _csv2bvecs(inlist, ref_num=0):
    import numpy as np
    import os.path as op

    bvecs = []
    translations = []

    for f in inlist:
        d = np.loadtxt(f, delimiter=',')
        t = d[0]
        r = d[1] - t
        bvecs.append(r.tolist())
        translations.append(t.tolist())

    bvecs.insert(ref_num, [0., 0., 0.])
    translations.insert(ref_num, [0., 0., 0.])
    out_bvec = op.abspath('rotated.bvecs')
    out_tran = op.abspath('translations.txt')

    np.savetxt(out_bvec, np.array(bvecs).T)
    np.savetxt(out_tran, translations)
    return out_bvec, out_tran


def _extract(in_files, ref_num=0):
    import numpy as np
    out = list(in_files)
    del out[ref_num]
    return out


def _insertref(reference, in_files, pos=0):
    import numpy as np
    out_files = np.atleast_1d(in_files).tolist()
    out_files.insert(pos, reference)
    return out_files
