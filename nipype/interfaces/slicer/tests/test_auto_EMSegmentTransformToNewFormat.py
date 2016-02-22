# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..utilities import EMSegmentTransformToNewFormat


def test_EMSegmentTransformToNewFormat_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputMRMLFileName=dict(argstr='--inputMRMLFileName %s',
    ),
    outputMRMLFileName=dict(argstr='--outputMRMLFileName %s',
    hash_files=False,
    ),
    templateFlag=dict(argstr='--templateFlag ',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = EMSegmentTransformToNewFormat._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_EMSegmentTransformToNewFormat_outputs():
    output_map = dict(outputMRMLFileName=dict(),
    )
    outputs = EMSegmentTransformToNewFormat._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
