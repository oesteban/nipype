# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..utils import Overlay


def test_Overlay_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    auto_thresh_bg=dict(argstr='-a',
    mandatory=True,
    position=5,
    xor=('auto_thresh_bg', 'full_bg_range', 'bg_thresh'),
    ),
    background_image=dict(argstr='%s',
    mandatory=True,
    position=4,
    ),
    bg_thresh=dict(argstr='%.3f %.3f',
    mandatory=True,
    position=5,
    xor=('auto_thresh_bg', 'full_bg_range', 'bg_thresh'),
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    full_bg_range=dict(argstr='-A',
    mandatory=True,
    position=5,
    xor=('auto_thresh_bg', 'full_bg_range', 'bg_thresh'),
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    out_file=dict(argstr='%s',
    genfile=True,
    hash_files=False,
    position=-1,
    ),
    out_type=dict(argstr='%s',
    position=2,
    usedefault=True,
    ),
    output_type=dict(usedefault=True,
    ),
    show_negative_stats=dict(argstr='%s',
    position=8,
    xor=['stat_image2'],
    ),
    stat_image=dict(argstr='%s',
    mandatory=True,
    position=6,
    ),
    stat_image2=dict(argstr='%s',
    position=9,
    xor=['show_negative_stats'],
    ),
    stat_thresh=dict(argstr='%.2f %.2f',
    mandatory=True,
    position=7,
    ),
    stat_thresh2=dict(argstr='%.2f %.2f',
    position=10,
    ),
    terminal_output=dict(nohash=True,
    ),
    transparency=dict(argstr='%s',
    position=1,
    usedefault=True,
    ),
    use_checkerboard=dict(argstr='-c',
    position=3,
    ),
    )
    inputs = Overlay.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Overlay_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Overlay.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
