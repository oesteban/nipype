# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..gtract import gtractFiberTracking


def test_gtractFiberTracking_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    branchingAngle=dict(argstr='--branchingAngle %f',
    ),
    branchingThreshold=dict(argstr='--branchingThreshold %f',
    ),
    curvatureThreshold=dict(argstr='--curvatureThreshold %f',
    ),
    endingSeedsLabel=dict(argstr='--endingSeedsLabel %d',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    guidedCurvatureThreshold=dict(argstr='--guidedCurvatureThreshold %f',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputAnisotropyVolume=dict(argstr='--inputAnisotropyVolume %s',
    ),
    inputEndingSeedsLabelMapVolume=dict(argstr='--inputEndingSeedsLabelMapVolume %s',
    ),
    inputStartingSeedsLabelMapVolume=dict(argstr='--inputStartingSeedsLabelMapVolume %s',
    ),
    inputTensorVolume=dict(argstr='--inputTensorVolume %s',
    ),
    inputTract=dict(argstr='--inputTract %s',
    ),
    maximumBranchPoints=dict(argstr='--maximumBranchPoints %d',
    ),
    maximumGuideDistance=dict(argstr='--maximumGuideDistance %f',
    ),
    maximumLength=dict(argstr='--maximumLength %f',
    ),
    minimumLength=dict(argstr='--minimumLength %f',
    ),
    numberOfThreads=dict(argstr='--numberOfThreads %d',
    ),
    outputTract=dict(argstr='--outputTract %s',
    hash_files=False,
    ),
    randomSeed=dict(argstr='--randomSeed %d',
    ),
    seedThreshold=dict(argstr='--seedThreshold %f',
    ),
    startingSeedsLabel=dict(argstr='--startingSeedsLabel %d',
    ),
    stepSize=dict(argstr='--stepSize %f',
    ),
    tendF=dict(argstr='--tendF %f',
    ),
    tendG=dict(argstr='--tendG %f',
    ),
    terminal_output=dict(deprecated='1.0.0',
    nohash=True,
    ),
    trackingMethod=dict(argstr='--trackingMethod %s',
    ),
    trackingThreshold=dict(argstr='--trackingThreshold %f',
    ),
    useLoopDetection=dict(argstr='--useLoopDetection ',
    ),
    useRandomWalk=dict(argstr='--useRandomWalk ',
    ),
    useTend=dict(argstr='--useTend ',
    ),
    writeXMLPolyDataFile=dict(argstr='--writeXMLPolyDataFile ',
    ),
    )
    inputs = gtractFiberTracking.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_gtractFiberTracking_outputs():
    output_map = dict(outputTract=dict(),
    )
    outputs = gtractFiberTracking.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
