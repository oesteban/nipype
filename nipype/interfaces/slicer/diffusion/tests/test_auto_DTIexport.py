# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..diffusion import DTIexport


def test_DTIexport_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputTensor=dict(argstr='%s',
    position=-2,
    ),
    outputFile=dict(argstr='%s',
    hash_files=False,
    position=-1,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = DTIexport._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_DTIexport_outputs():
    output_map = dict(outputFile=dict(position=-1,
    ),
    )
    outputs = DTIexport._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
