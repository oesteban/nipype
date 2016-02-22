# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import MedianFilter3D


def test_MedianFilter3D_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    debug=dict(argstr='-debug',
    position=1,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=-2,
    ),
    out_filename=dict(argstr='%s',
    genfile=True,
    position=-1,
    ),
    quiet=dict(argstr='-quiet',
    position=1,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = MedianFilter3D._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_MedianFilter3D_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = MedianFilter3D._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
