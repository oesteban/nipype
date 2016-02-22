# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..gtract import gtractCoregBvalues


def test_gtractCoregBvalues_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    debugLevel=dict(argstr='--debugLevel %d',
    ),
    eddyCurrentCorrection=dict(argstr='--eddyCurrentCorrection ',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fixedVolume=dict(argstr='--fixedVolume %s',
    ),
    fixedVolumeIndex=dict(argstr='--fixedVolumeIndex %d',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    maximumStepSize=dict(argstr='--maximumStepSize %f',
    ),
    minimumStepSize=dict(argstr='--minimumStepSize %f',
    ),
    movingVolume=dict(argstr='--movingVolume %s',
    ),
    numberOfIterations=dict(argstr='--numberOfIterations %d',
    ),
    numberOfSpatialSamples=dict(argstr='--numberOfSpatialSamples %d',
    ),
    numberOfThreads=dict(argstr='--numberOfThreads %d',
    ),
    outputTransform=dict(argstr='--outputTransform %s',
    hash_files=False,
    ),
    outputVolume=dict(argstr='--outputVolume %s',
    hash_files=False,
    ),
    registerB0Only=dict(argstr='--registerB0Only ',
    ),
    relaxationFactor=dict(argstr='--relaxationFactor %f',
    ),
    samplingPercentage=dict(argstr='--samplingPercentage %f',
    ),
    spatialScale=dict(argstr='--spatialScale %f',
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = gtractCoregBvalues._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_gtractCoregBvalues_outputs():
    output_map = dict(outputTransform=dict(),
    outputVolume=dict(),
    )
    outputs = gtractCoregBvalues._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
