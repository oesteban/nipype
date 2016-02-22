# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import Automask


def test_Automask_inputs():
    input_map = dict(apply_prefix=dict(argstr='-apply_prefix %s',
    ),
    args=dict(argstr='%s',
    ),
    brain_file=dict(),
    clfrac=dict(argstr='-clfrac %.2f',
    ),
    dilate=dict(argstr='-dilate %s',
    ),
    erode=dict(argstr='-erode %s',
    ),
    in_file=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=-1,
    ),
    out_file=dict(),
    output_type=dict(usedefault=True,
    ),
    prefix=dict(argstr='-prefix %s',
    ),
    )
    inputs = Automask._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Automask_outputs():
    output_map = dict(brain_file=dict(),
    out_file=dict(),
    )
    outputs = Automask._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
