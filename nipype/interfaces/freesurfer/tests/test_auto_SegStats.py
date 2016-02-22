# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..model import SegStats


def test_SegStats_inputs():
    input_map = dict(annot=dict(argstr='--annot %s %s %s',
    mandatory=True,
    xor=('segmentation_file', 'annot', 'surf_label'),
    ),
    args=dict(argstr='%s',
    ),
    avgwf_file=dict(argstr='--avgwfvol %s',
    ),
    avgwf_txt_file=dict(argstr='--avgwf %s',
    ),
    brain_vol=dict(),
    calc_power=dict(argstr='--%s',
    ),
    calc_snr=dict(argstr='--snr',
    ),
    color_table_file=dict(argstr='--ctab %s',
    xor=('color_table_file', 'default_color_table', 'gca_color_table'),
    ),
    cortex_vol_from_surf=dict(argstr='--surf-ctx-vol',
    ),
    default_color_table=dict(argstr='--ctab-default',
    xor=('color_table_file', 'default_color_table', 'gca_color_table'),
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    etiv=dict(argstr='--etiv',
    ),
    etiv_only=dict(),
    exclude_ctx_gm_wm=dict(argstr='--excl-ctxgmwm',
    ),
    exclude_id=dict(argstr='--excludeid %d',
    ),
    frame=dict(argstr='--frame %d',
    ),
    gca_color_table=dict(argstr='--ctab-gca %s',
    xor=('color_table_file', 'default_color_table', 'gca_color_table'),
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='--i %s',
    ),
    mask_erode=dict(argstr='--maskerode %d',
    ),
    mask_file=dict(argstr='--mask %s',
    ),
    mask_frame=dict(requires=['mask_file'],
    ),
    mask_invert=dict(argstr='--maskinvert',
    ),
    mask_sign=dict(),
    mask_thresh=dict(argstr='--maskthresh %f',
    ),
    multiply=dict(argstr='--mul %f',
    ),
    non_empty_only=dict(argstr='--nonempty',
    ),
    partial_volume_file=dict(argstr='--pv %f',
    ),
    segment_id=dict(argstr='--id %s...',
    ),
    segmentation_file=dict(argstr='--seg %s',
    mandatory=True,
    xor=('segmentation_file', 'annot', 'surf_label'),
    ),
    sf_avg_file=dict(argstr='--sfavg %s',
    ),
    subjects_dir=dict(),
    summary_file=dict(argstr='--sum %s',
    genfile=True,
    ),
    surf_label=dict(argstr='--slabel %s %s %s',
    mandatory=True,
    xor=('segmentation_file', 'annot', 'surf_label'),
    ),
    terminal_output=dict(nohash=True,
    ),
    vox=dict(argstr='--vox %s',
    ),
    wm_vol_from_surf=dict(argstr='--surf-wm-vol',
    ),
    )
    inputs = SegStats._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_SegStats_outputs():
    output_map = dict(avgwf_file=dict(),
    avgwf_txt_file=dict(),
    sf_avg_file=dict(),
    summary_file=dict(),
    )
    outputs = SegStats._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
