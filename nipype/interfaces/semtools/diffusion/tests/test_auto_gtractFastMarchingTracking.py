# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.semtools.diffusion.gtract import gtractFastMarchingTracking

def test_gtractFastMarchingTracking_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    costStepSize=dict(argstr='--costStepSize %f',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputAnisotropyVolume=dict(argstr='--inputAnisotropyVolume %s',
    ),
    inputCostVolume=dict(argstr='--inputCostVolume %s',
    ),
    inputStartingSeedsLabelMapVolume=dict(argstr='--inputStartingSeedsLabelMapVolume %s',
    ),
    inputTensorVolume=dict(argstr='--inputTensorVolume %s',
    ),
    maximumStepSize=dict(argstr='--maximumStepSize %f',
    ),
    minimumStepSize=dict(argstr='--minimumStepSize %f',
    ),
    numberOfIterations=dict(argstr='--numberOfIterations %d',
    ),
    numberOfThreads=dict(argstr='--numberOfThreads %d',
    ),
    outputTract=dict(argstr='--outputTract %s',
    hash_files=False,
    ),
    seedThreshold=dict(argstr='--seedThreshold %f',
    ),
    startingSeedsLabel=dict(argstr='--startingSeedsLabel %d',
    ),
    terminal_output=dict(nohash=True,
    ),
    trackingThreshold=dict(argstr='--trackingThreshold %f',
    ),
    writeXMLPolyDataFile=dict(argstr='--writeXMLPolyDataFile ',
    ),
    )
    inputs = gtractFastMarchingTracking.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_gtractFastMarchingTracking_outputs():
    output_map = dict(outputTract=dict(),
    )
    outputs = gtractFastMarchingTracking.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

