# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..utils import Smooth


def test_Smooth_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fwhm=dict(argstr='-kernel gauss %.03f -fmean',
    mandatory=True,
    position=1,
    xor=['sigma'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=0,
    ),
    output_type=dict(usedefault=True,
    ),
    sigma=dict(argstr='-kernel gauss %.03f -fmean',
    mandatory=True,
    position=1,
    xor=['fwhm'],
    ),
    smoothed_file=dict(argstr='%s',
    hash_files=False,
    position=2,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = Smooth._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Smooth_outputs():
    output_map = dict(smoothed_file=dict(),
    )
    outputs = Smooth._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
