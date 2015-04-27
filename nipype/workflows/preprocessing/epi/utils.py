# coding: utf-8
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os

import nipype.pipeline.engine as pe
from nipype.interfaces.io import JSONFileGrabber
from nipype.interfaces import utility as niu
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import ants
from nipype.interfaces import fsl


def _eff_t_echo(echospacing, acc_factor):
    eff_echo = echospacing / (1.0 * acc_factor)
    return eff_echo


def _fix_enc_dir(enc_dir):
    enc_dir = enc_dir.lower()
    if enc_dir == 'lr':
        return 'x-'
    if enc_dir == 'rl':
        return 'x'
    if enc_dir == 'ap':
        return 'y-'
    if enc_dir == 'pa':
        return 'y'
    return enc_dir


def _nonb0(in_bval):
    import numpy as np
    bvals = np.loadtxt(in_bval)
    return np.where(bvals != 0)[0].tolist()


def _xfm_jacobian(in_xfm):
    import numpy as np
    from math import fabs
    return [fabs(np.linalg.det(np.loadtxt(xfm))) for xfm in in_xfm]


def _get_zoom(in_file, enc_dir):
    import nibabel as nb

    zooms = nb.load(in_file).get_header().get_zooms()

    if 'y' in enc_dir:
        return zooms[1]
    elif 'x' in enc_dir:
        return zooms[0]
    elif 'z' in enc_dir:
        return zooms[2]
    else:
        raise ValueError('Wrong encoding direction string')


def flirt_4d(name='DWICoregistration', excl_nodiff=False,
             flirt_param={}):
    """
    Generates a workflow for linear registration of dwi volumes
    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['reference',
                        'in_file', 'ref_mask', 'in_xfms', 'in_bval']),
                        name='inputnode')

    initmat = pe.Node(niu.Function(input_names=['in_bval', 'in_xfms',
                      'excl_nodiff'], output_names=['init_xfms'],
                                   function=_checkinitxfm), name='InitXforms')
    initmat.inputs.excl_nodiff = excl_nodiff
    dilate = pe.Node(fsl.maths.MathsCommand(nan2zeros=True,
                     args='-kernel sphere 5 -dilM'), name='MskDilate')
    split = pe.Node(fsl.Split(dimension='t'), name='SplitDWIs')
    pick_ref = pe.Node(niu.Select(), name='Pick_b0')
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='Bias')
    enhb0 = pe.Node(niu.Function(input_names=['in_file', 'in_mask',
                    'clip_limit'], output_names=['out_file'],
                                 function=enhance), name='B0Equalize')
    enhb0.inputs.clip_limit = 0.015
    enhdw = pe.MapNode(niu.Function(input_names=['in_file', 'in_mask'],
                       output_names=['out_file'], function=enhance),
                       name='DWEqualize', iterfield=['in_file'])
    flirt = pe.MapNode(fsl.FLIRT(**flirt_param), name='CoRegistration',
                       iterfield=['in_file', 'in_matrix_file'])
    thres = pe.MapNode(fsl.Threshold(thresh=0.0), iterfield=['in_file'],
                       name='RemoveNegative')
    merge = pe.Node(fsl.Merge(dimension='t'), name='MergeDWIs')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file',
                         'out_xfms']), name='outputnode')
    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,  split,      [('in_file', 'in_file')]),
        (inputnode,  dilate,     [('ref_mask', 'in_file')]),
        (inputnode,  enhb0,      [('ref_mask', 'in_mask')]),
        (inputnode,  initmat,    [('in_xfms', 'in_xfms'),
                                  ('in_bval', 'in_bval')]),
        (inputnode,  n4,         [('reference', 'input_image'),
                                  ('ref_mask', 'mask_image')]),
        (dilate,     flirt,      [('out_file', 'ref_weight'),
                                  ('out_file', 'in_weight')]),
        (n4,         enhb0,      [('output_image', 'in_file')]),
        (split,      enhdw,      [('out_files', 'in_file')]),
        (dilate,     enhdw,      [('out_file', 'in_mask')]),
        (enhb0,      flirt,      [('out_file', 'reference')]),
        (enhdw,      flirt,      [('out_file', 'in_file')]),
        (initmat,    flirt,      [('init_xfms', 'in_matrix_file')]),
        (flirt,      thres,      [('out_file', 'in_file')]),
        (thres,      merge,      [('out_file', 'in_files')]),
        (merge,      outputnode, [('merged_file', 'out_file')]),
        (flirt,      outputnode, [('out_matrix_file', 'out_xfms')])
    ])
    return wf


def apply_all_corrections(name='UnwarpArtifacts'):
    """
    Combines two lists of linear transforms with the deformation field
    map obtained typically after the SDC process.
    Additionally, computes the corresponding bspline coefficients and
    the map of determinants of the jacobian.
    """

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_sdc',
                        'in_hmc', 'in_ecc', 'in_dwi']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_warp',
                         'out_coeff', 'out_jacobian']), name='outputnode')
    warps = pe.MapNode(fsl.ConvertWarp(relwarp=True),
                       iterfield=['premat', 'postmat'],
                       name='ConvertWarp')

    selref = pe.Node(niu.Select(index=[0]), name='Reference')

    split = pe.Node(fsl.Split(dimension='t'), name='SplitDWIs')
    unwarp = pe.MapNode(fsl.ApplyWarp(), iterfield=['in_file', 'field_file'],
                        name='UnwarpDWIs')

    coeffs = pe.MapNode(fsl.WarpUtils(out_format='spline'),
                        iterfield=['in_file'], name='CoeffComp')
    jacobian = pe.MapNode(fsl.WarpUtils(write_jacobian=True),
                          iterfield=['in_file'], name='JacobianComp')
    jacmult = pe.MapNode(fsl.MultiImageMaths(op_string='-mul %s'),
                         iterfield=['in_file', 'operand_files'],
                         name='ModulateDWIs')

    thres = pe.MapNode(fsl.Threshold(thresh=0.0), iterfield=['in_file'],
                       name='RemoveNegative')
    merge = pe.Node(fsl.Merge(dimension='t'), name='MergeDWIs')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode,   warps,      [('in_sdc', 'warp1'),
                                   ('in_hmc', 'premat'),
                                   ('in_ecc', 'postmat'),
                                   ('in_dwi', 'reference')]),
        (inputnode,   split,      [('in_dwi', 'in_file')]),
        (split,       selref,     [('out_files', 'inlist')]),
        (warps,       unwarp,     [('out_file', 'field_file')]),
        (split,       unwarp,     [('out_files', 'in_file')]),
        (selref,      unwarp,     [('out', 'ref_file')]),
        (selref,      coeffs,     [('out', 'reference')]),
        (warps,       coeffs,     [('out_file', 'in_file')]),
        (selref,      jacobian,   [('out', 'reference')]),
        (coeffs,      jacobian,   [('out_file', 'in_file')]),
        (unwarp,      jacmult,    [('out_file', 'in_file')]),
        (jacobian,    jacmult,    [('out_jacobian', 'operand_files')]),
        (jacmult,     thres,      [('out_file', 'in_file')]),
        (thres,       merge,      [('out_file', 'in_files')]),
        (warps,       outputnode, [('out_file', 'out_warp')]),
        (coeffs,      outputnode, [('out_file', 'out_coeff')]),
        (jacobian,    outputnode, [('out_jacobian', 'out_jacobian')]),
        (merge,       outputnode, [('merged_file', 'out_file')])
    ])
    return wf


def extract_bval(in_dwi, in_bval, b=0, out_file=None):
    """
    Writes an image containing only the volumes with b-value specified at
    input
    """
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_dwi))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("%s_tsoi%s" % (fname, ext))

    im = nb.load(in_dwi)
    dwidata = im.get_data()
    bvals = np.loadtxt(in_bval)

    if b == 'diff':
        selection = np.where(bvals != 0)
    elif b == 'nodiff':
        selection = np.where(bvals == 0)
    else:
        selection = np.where(bvals == b)

    extdata = np.squeeze(dwidata.take(selection, axis=3))
    hdr = im.get_header().copy()
    hdr.set_data_shape(extdata.shape)
    nb.Nifti1Image(extdata, im.get_affine(),
                   hdr).to_filename(out_file)
    return out_file


def remove_comp(in_file, in_bval, volid=0, out_file=None):
    """
    Removes the volume ``volid`` from the 4D nifti file
    """
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("%s_extract%s" % (fname, ext))

    im = nb.load(in_file)
    data = im.get_data()
    hdr = im.get_header().copy()
    bval = np.loadtxt(in_bval)

    if volid == 0:
        data = data[..., 1:]
        bval = bval[1:]
    elif volid == (data.shape[-1] - 1):
        data = data[..., :-1]
        bval = bval[:-1]
    else:
        data = np.concatenate((data[..., :volid], data[..., (volid + 1):]),
                              axis=3)
        bval = np.hstack((bval[:volid], bval[(volid + 1):]))
    hdr.set_data_shape(data.shape)
    nb.Nifti1Image(data, im.get_affine(), hdr).to_filename(out_file)

    out_bval = op.abspath('bval_extract.txt')
    np.savetxt(out_bval, bval)
    return out_file, out_bval


def insert_mat(inlist, volid=0):
    import numpy as np
    import os.path as op
    idfname = op.abspath('identity.mat')
    out = inlist
    np.savetxt(idfname, np.eye(4))
    out.insert(volid, idfname)
    return out


def recompose_dwi(in_dwi, in_bval, in_corrected, out_file=None):
    """
    Recompose back the dMRI data accordingly the b-values table after EC
    correction
    """
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_dwi))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("%s_eccorrect%s" % (fname, ext))

    im = nb.load(in_dwi)
    dwidata = im.get_data()
    bvals = np.loadtxt(in_bval)
    dwis = np.where(bvals != 0)[0].tolist()

    if len(dwis) != len(in_corrected):
        raise RuntimeError(('Length of DWIs in b-values table and after'
                            'correction should match'))

    for bindex, dwi in zip(dwis, in_corrected):
        dwidata[..., bindex] = nb.load(dwi).get_data()

    nb.Nifti1Image(dwidata, im.get_affine(),
                   im.get_header()).to_filename(out_file)
    return out_file


def recompose_xfm(in_bval, in_xfms):
    """
    Insert identity transformation matrices in b0 volumes to build up a list
    """
    import numpy as np
    import os.path as op

    bvals = np.loadtxt(in_bval)
    out_matrix = np.array([np.eye(4)] * len(bvals))
    xfms = iter([np.loadtxt(xfm) for xfm in in_xfms])
    out_files = []

    for i, b in enumerate(bvals):
        if b == 0.0:
            mat = np.eye(4)
        else:
            mat = xfms.next()

        out_name = op.abspath('eccor_%04d.mat' % i)
        out_files.append(out_name)
        np.savetxt(out_name, mat)

    return out_files


def time_avg(in_file, index=[0], out_file=None):
    """
    Average the input time-series, selecting the indices given in index

    .. warning:: time steps should be already registered (corrected for
      head motion artifacts).

    """
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("%s_baseline%s" % (fname, ext))

    index = np.atleast_1d(index).tolist()

    imgs = np.array(nb.four_to_three(nb.load(in_file)))[index]
    if len(index) == 1:
        data = imgs[0].get_data().astype(np.float32)
    else:
        data = np.average(np.array([im.get_data().astype(np.float32) for im in imgs]),
                         axis=0)

    hdr = imgs[0].get_header().copy()
    hdr.set_data_shape(data.shape)
    hdr.set_xyzt_units('mm')
    hdr.set_data_dtype(np.float32)
    nb.Nifti1Image(data, imgs[0].get_affine(), hdr).to_filename(out_file)
    return out_file


def b0_indices(in_bval, max_b=10.0):
    """
    Extract the indices of slices in a b-values file with a low b value
    """
    import numpy as np
    bval = np.loadtxt(in_bval)
    return np.argwhere(bval <= max_b).flatten().tolist()


def b0_average(in_dwi, in_bval, max_b=10.0, out_file=None):
    """
    A function that averages the *b0* volumes from a DWI dataset.
    As current dMRI data are being acquired with all b-values > 0.0,
    the *lowb* volumes are selected by specifying the parameter max_b.

    .. warning:: *b0* should be already registered (head motion artifact should
      be corrected).

    """
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_dwi))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("%s_avg_b0%s" % (fname, ext))

    imgs = np.array(nb.four_to_three(nb.load(in_dwi)))
    index = b0_indices(in_bval, max_b=max_b)
    b0s = [im.get_data().astype(np.float32)
           for im in imgs[index]]
    b0 = np.average(np.array(b0s), axis=0)

    hdr = imgs[0].get_header().copy()
    hdr.set_data_shape(b0.shape)
    hdr.set_xyzt_units('mm')
    hdr.set_data_dtype(np.float32)
    nb.Nifti1Image(b0, imgs[0].get_affine(), hdr).to_filename(out_file)
    return out_file


def rotate_bvecs(in_bvec, in_matrix):
    """
    Rotates the input bvec file accordingly with a list of matrices.

    .. note:: the input affine matrix transforms points in the destination
      image to their corresponding coordinates in the original image.
      Therefore, this matrix should be inverted first, as we want to know
      the target position of :math:`\\vec{r}`.

    """
    import os
    import numpy as np

    name, fext = os.path.splitext(os.path.basename(in_bvec))
    if fext == '.gz':
        name, _ = os.path.splitext(name)
    out_file = os.path.abspath('%s_rotated.bvec' % name)
    bvecs = np.loadtxt(in_bvec).T
    new_bvecs = []

    if len(bvecs) != len(in_matrix):
        raise RuntimeError(('Number of b-vectors (%d) and rotation '
                           'matrices (%d) should match.') % (len(bvecs),
                           len(in_matrix)))

    for bvec, mat in zip(bvecs, in_matrix):
        if np.all(bvec == 0.0):
            new_bvecs.append(bvec)
        else:
            invrot = np.linalg.inv(np.loadtxt(mat))[:3, :3]
            newbvec = invrot.dot(bvec)
            new_bvecs.append((newbvec/np.linalg.norm(newbvec)))

    np.savetxt(out_file, np.array(new_bvecs).T, fmt='%0.15f')
    return out_file


def eddy_rotate_bvecs(in_bvec, eddy_params):
    """
    Rotates the input bvec file accordingly with a list of parameters sourced
    from ``eddy``, as explained `here
    <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/EDDY/Faq#Will_eddy_rotate_my_bevcs_for_me.3F>`_.
    """
    import os
    import numpy as np
    from math import sin, cos

    name, fext = os.path.splitext(os.path.basename(in_bvec))
    if fext == '.gz':
        name, _ = os.path.splitext(name)
    out_file = os.path.abspath('%s_rotated.bvec' % name)
    bvecs = np.loadtxt(in_bvec).T
    new_bvecs = []

    params = np.loadtxt(eddy_params)

    if len(bvecs) != len(params):
        raise RuntimeError(('Number of b-vectors and rotation '
                           'matrices should match.'))

    for bvec, row in zip(bvecs, params):
        if np.all(bvec == 0.0):
            new_bvecs.append(bvec)
        else:
            ax = row[3]
            ay = row[4]
            az = row[5]

            Rx = np.array([[1.0, 0.0, 0.0],
                          [0.0, cos(ax), -sin(ax)],
                          [0.0, sin(ax), cos(ax)]])
            Ry = np.array([[cos(ay), 0.0, sin(ay)],
                          [0.0, 1.0, 0.0],
                          [-sin(ay), 0.0, cos(ay)]])
            Rz = np.array([[cos(az), -sin(az), 0.0],
                          [sin(az), cos(az), 0.0],
                          [0.0, 0.0, 1.0]])
            R = Rx.dot(Ry).dot(Rz)

            invrot = np.linalg.inv(R)
            newbvec = invrot.dot(bvec)
            new_bvecs.append((newbvec/np.linalg.norm(newbvec)))

    np.savetxt(out_file, np.array(new_bvecs).T, fmt='%0.15f')
    return out_file


def demean_image(in_file, in_mask=None, out_file=None):
    """
    Demean image data inside mask
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    import math

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_demean.nii.gz' % fname)

    im = nb.load(in_file)
    data = im.get_data().astype(np.float32)
    msk = np.ones_like(data)

    if in_mask is not None:
        msk = nb.load(in_mask).get_data().astype(np.float32)
        msk[msk > 0] = 1.0
        msk[msk < 1] = 0.0

    mean = np.median(data[msk == 1].reshape(-1))
    data[msk == 1] = data[msk == 1] - mean
    nb.Nifti1Image(data, im.get_affine(),
                   im.get_header()).to_filename(out_file)
    return out_file


def add_empty_vol(in_file, out_file=None):
    """
    Adds an empty vol to the phase difference image
    """
    import nibabel as nb
    import os.path as op
    import numpy as np
    import math

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_4D.nii.gz' % fname)

    im = nb.load(in_file)
    zim = nb.Nifti1Image(np.zeros_like(im.get_data()), im.get_affine(),
                         im.get_header())
    nb.funcs.concat_images([im, zim]).to_filename(out_file)
    return out_file


def reorient_bvecs(in_dwi, old_dwi, in_bvec):
    """
    Checks reorientations of ``in_dwi`` w.r.t. ``old_dwi`` and
    reorients the in_bvec table accordingly.
    """
    import os
    import numpy as np
    import nibabel as nb

    name, fext = os.path.splitext(os.path.basename(in_bvec))
    if fext == '.gz':
        name, _ = os.path.splitext(name)
    out_file = os.path.abspath('%s_reorient.bvec' % name)
    bvecs = np.loadtxt(in_bvec).T
    new_bvecs = []

    N = nb.load(in_dwi).get_affine()
    O = nb.load(old_dwi).get_affine()
    RS = N.dot(np.linalg.inv(O))[:3, :3]
    sc_idx = np.where((np.abs(RS) != 1) & (RS != 0))
    S = np.ones_like(RS)
    S[sc_idx] = RS[sc_idx]
    R = RS/S

    new_bvecs = [R.dot(b) for b in bvecs]
    np.savetxt(out_file, np.array(new_bvecs).T, fmt='%0.15f')
    return out_file


def copy_hdr(in_file, in_file_hdr, out_file=None):
    import numpy as np
    import nibabel as nb
    import os.path as op

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_fixhdr.nii.gz' % fname)

    imref = nb.load(in_file_hdr)
    hdr = imref.get_header().copy()
    hdr.set_data_dtype(np.float32)
    vsm = nb.load(in_file).get_data().astype(np.float32)
    hdr.set_data_shape(vsm.shape)
    hdr.set_xyzt_units('mm')
    nii = nb.Nifti1Image(vsm, imref.get_affine(), hdr)
    nii.to_filename(out_file)
    return out_file


def enhance(in_file, clip_limit=0.010, in_mask=None, out_file=None):
    import numpy as np
    import nibabel as nb
    import os.path as op
    from skimage import exposure, img_as_int

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_enh.nii.gz' % fname)

    im = nb.load(in_file)
    imdata = im.get_data()
    imshape = im.get_shape()

    if in_mask is not None:
        msk = nb.load(in_mask).get_data()
        msk[msk > 0] = 1
        msk[msk < 1] = 0
        imdata = imdata * msk

    immin = imdata.min()
    imdata = (imdata - immin).astype(np.uint16)

    adapted = exposure.equalize_adapthist(imdata.reshape(imshape[0], -1),
                                          clip_limit=clip_limit)

    nb.Nifti1Image(adapted.reshape(imshape), im.get_affine(),
                   im.get_header()).to_filename(out_file)

    return out_file


def _checkinitxfm(in_bval, excl_nodiff, in_xfms=None):
    from nipype.interfaces.base import isdefined
    import numpy as np
    import os.path as op
    bvals = np.loadtxt(in_bval)

    gen_id = ((in_xfms is None) or
              (not isdefined(in_xfms)) or
              (len(in_xfms) != len(bvals)))

    init_xfms = []
    if excl_nodiff:
        dws = np.where(bvals != 0)[0].tolist()
    else:
        dws = range(len(bvals))

    if gen_id:
        for i in dws:
            xfm_file = op.abspath('init_%04d.mat' % i)
            np.savetxt(xfm_file, np.eye(4))
            init_xfms.append(xfm_file)
    else:
        init_xfms = [in_xfms[i] for i in dws]

    return init_xfms
