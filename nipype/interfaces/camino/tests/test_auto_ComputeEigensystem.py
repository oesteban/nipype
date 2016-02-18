# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..dti import ComputeEigensystem


def test_ComputeEigensystem_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='< %s',
    mandatory=True,
    position=1,
    ),
    inputdatatype=dict(argstr='-inputdatatype %s',
    usedefault=True,
    ),
    inputmodel=dict(argstr='-inputmodel %s',
    ),
    maxcomponents=dict(argstr='-maxcomponents %d',
    ),
    out_file=dict(argstr='> %s',
    position=-1,
    usedefault=True,
    ),
    outputdatatype=dict(argstr='-outputdatatype %s',
    usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = ComputeEigensystem.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ComputeEigensystem_outputs():
    output_map = dict(eigen=dict(),
    )
    outputs = ComputeEigensystem.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
