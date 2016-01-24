# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..odf import ODFTracker


def test_ODFTracker_inputs():
    input_map = dict(ODF=dict(mandatory=True,
    ),
    angle_threshold=dict(argstr='-at %f',
    ),
    args=dict(argstr='%s',
    ),
    disc=dict(argstr='-disc',
    ),
    dsi=dict(argstr='-dsi',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    image_orientation_vectors=dict(argstr='-iop %f',
    ),
    input_data_prefix=dict(argstr='%s',
    position=0,
    usedefault=True,
    ),
    input_output_type=dict(argstr='-it %s',
    usedefault=True,
    ),
    invert_x=dict(argstr='-ix',
    ),
    invert_y=dict(argstr='-iy',
    ),
    invert_z=dict(argstr='-iz',
    ),
    limit=dict(argstr='-limit %d',
    ),
    mask1_file=dict(argstr='-m %s',
    mandatory=True,
    position=2,
    ),
    mask1_threshold=dict(position=3,
    ),
    mask2_file=dict(argstr='-m2 %s',
    position=4,
    ),
    mask2_threshold=dict(position=5,
    ),
    max=dict(mandatory=True,
    ),
    out_file=dict(argstr='%s',
    position=1,
    usedefault=True,
    ),
    random_seed=dict(argstr='-rseed %s',
    ),
    runge_kutta2=dict(argstr='-rk2',
    ),
    slice_order=dict(argstr='-sorder %d',
    ),
    step_length=dict(argstr='-l %f',
    ),
    swap_xy=dict(argstr='-sxy',
    ),
    swap_yz=dict(argstr='-syz',
    ),
    swap_zx=dict(argstr='-szx',
    ),
    terminal_output=dict(nohash=True,
    ),
    voxel_order=dict(argstr='-vorder %s',
    ),
    )
    inputs = ODFTracker.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ODFTracker_outputs():
    output_map = dict(track_file=dict(),
    )
    outputs = ODFTracker.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
