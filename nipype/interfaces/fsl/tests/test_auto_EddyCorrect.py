# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..epi import EddyCorrect


def test_EddyCorrect_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=0,
    ),
    out_file=dict(argstr='%s',
    name_source=['in_file'],
    name_template='%s_edc',
    output_name='eddy_corrected',
    position=1,
    ),
    output_type=dict(usedefault=True,
    ),
    ref_num=dict(argstr='%d',
    mandatory=True,
    position=2,
    usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = EddyCorrect.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_EddyCorrect_outputs():
    output_map = dict(eddy_corrected=dict(),
    )
    outputs = EddyCorrect.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
