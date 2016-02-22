# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..converters import DWISimpleCompare


def test_DWISimpleCompare_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    checkDWIData=dict(argstr='--checkDWIData ',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputVolume1=dict(argstr='--inputVolume1 %s',
    ),
    inputVolume2=dict(argstr='--inputVolume2 %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = DWISimpleCompare._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_DWISimpleCompare_outputs():
    output_map = dict()
    outputs = DWISimpleCompare._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
