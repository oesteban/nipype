# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import Retroicor


def test_Retroicor_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    card=dict(argstr='-card %s',
    position=-2,
    ),
    cardphase=dict(argstr='-cardphase %s',
    hash_files=False,
    position=-6,
    ),
    in_file=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=-1,
    ),
    order=dict(argstr='-order %s',
    position=-5,
    ),
    out_file=dict(argstr='-prefix %s',
    mandatory=True,
    position=1,
    ),
    output_type=dict(usedefault=True,
    ),
    resp=dict(argstr='-resp %s',
    position=-3,
    ),
    respphase=dict(argstr='-respphase %s',
    hash_files=False,
    position=-7,
    ),
    threshold=dict(argstr='-threshold %d',
    position=-4,
    ),
    )
    inputs = Retroicor._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Retroicor_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Retroicor._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
