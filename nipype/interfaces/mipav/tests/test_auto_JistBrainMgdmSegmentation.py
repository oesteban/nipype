# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..developer import JistBrainMgdmSegmentation


def test_JistBrainMgdmSegmentation_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inAdjust=dict(argstr='--inAdjust %s',
    ),
    inAtlas=dict(argstr='--inAtlas %s',
    ),
    inCompute=dict(argstr='--inCompute %s',
    ),
    inCurvature=dict(argstr='--inCurvature %f',
    ),
    inData=dict(argstr='--inData %f',
    ),
    inFLAIR=dict(argstr='--inFLAIR %s',
    ),
    inMP2RAGE=dict(argstr='--inMP2RAGE %s',
    ),
    inMP2RAGE2=dict(argstr='--inMP2RAGE2 %s',
    ),
    inMPRAGE=dict(argstr='--inMPRAGE %s',
    ),
    inMax=dict(argstr='--inMax %d',
    ),
    inMin=dict(argstr='--inMin %f',
    ),
    inOutput=dict(argstr='--inOutput %s',
    ),
    inPV=dict(argstr='--inPV %s',
    ),
    inPosterior=dict(argstr='--inPosterior %f',
    ),
    inSteps=dict(argstr='--inSteps %d',
    ),
    inTopology=dict(argstr='--inTopology %s',
    ),
    null=dict(argstr='--null %s',
    ),
    outLevelset=dict(argstr='--outLevelset %s',
    hash_files=False,
    ),
    outPosterior2=dict(argstr='--outPosterior2 %s',
    hash_files=False,
    ),
    outPosterior3=dict(argstr='--outPosterior3 %s',
    hash_files=False,
    ),
    outSegmented=dict(argstr='--outSegmented %s',
    hash_files=False,
    ),
    terminal_output=dict(nohash=True,
    ),
    xDefaultMem=dict(argstr='-xDefaultMem %d',
    ),
    xMaxProcess=dict(argstr='-xMaxProcess %d',
    usedefault=True,
    ),
    xPrefExt=dict(argstr='--xPrefExt %s',
    ),
    )
    inputs = JistBrainMgdmSegmentation._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_JistBrainMgdmSegmentation_outputs():
    output_map = dict(outLevelset=dict(),
    outPosterior2=dict(),
    outPosterior3=dict(),
    outSegmented=dict(),
    )
    outputs = JistBrainMgdmSegmentation._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
