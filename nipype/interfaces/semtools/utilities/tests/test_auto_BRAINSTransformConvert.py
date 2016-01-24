# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..brains import BRAINSTransformConvert


def test_BRAINSTransformConvert_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    displacementVolume=dict(argstr='--displacementVolume %s',
    hash_files=False,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputTransform=dict(argstr='--inputTransform %s',
    ),
    outputPrecisionType=dict(argstr='--outputPrecisionType %s',
    ),
    outputTransform=dict(argstr='--outputTransform %s',
    hash_files=False,
    ),
    outputTransformType=dict(argstr='--outputTransformType %s',
    ),
    referenceVolume=dict(argstr='--referenceVolume %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = BRAINSTransformConvert.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_BRAINSTransformConvert_outputs():
    output_map = dict(displacementVolume=dict(),
    outputTransform=dict(),
    )
    outputs = BRAINSTransformConvert.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
