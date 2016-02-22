# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..segmentation import OtsuThresholdSegmentation


def test_OtsuThresholdSegmentation_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    brightObjects=dict(argstr='--brightObjects ',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    faceConnected=dict(argstr='--faceConnected ',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputVolume=dict(argstr='%s',
    position=-2,
    ),
    minimumObjectSize=dict(argstr='--minimumObjectSize %d',
    ),
    numberOfBins=dict(argstr='--numberOfBins %d',
    ),
    outputVolume=dict(argstr='%s',
    hash_files=False,
    position=-1,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = OtsuThresholdSegmentation._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_OtsuThresholdSegmentation_outputs():
    output_map = dict(outputVolume=dict(position=-1,
    ),
    )
    outputs = OtsuThresholdSegmentation._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
