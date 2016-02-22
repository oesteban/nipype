# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..maths import ErodeImage


def test_ErodeImage_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=2,
    ),
    internal_datatype=dict(argstr='-dt %s',
    position=1,
    ),
    kernel_file=dict(argstr='%s',
    position=5,
    xor=['kernel_size'],
    ),
    kernel_shape=dict(argstr='-kernel %s',
    position=4,
    ),
    kernel_size=dict(argstr='%.4f',
    position=5,
    xor=['kernel_file'],
    ),
    minimum_filter=dict(argstr='-eroF',
    position=6,
    usedefault=True,
    ),
    nan2zeros=dict(argstr='-nan',
    position=3,
    usedefault=True,
    ),
    out_file=dict(argstr='%s',
    hash_files=False,
    position=-2,
    ),
    output_datatype=dict(argstr='-odt %s',
    position=-1,
    ),
    output_type=dict(usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = ErodeImage._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ErodeImage_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = ErodeImage._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
