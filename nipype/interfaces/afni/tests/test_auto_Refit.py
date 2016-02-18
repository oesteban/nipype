# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import Refit


def test_Refit_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    deoblique=dict(argstr='-deoblique',
    usedefault=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    copyfile=True,
    mandatory=True,
    position=-1,
    ),
    space=dict(argstr='-space %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    xdel=dict(argstr='-xdel %f',
    ),
    xorigin=dict(argstr='-xorigin %s',
    ),
    ydel=dict(argstr='-ydel %f',
    ),
    yorigin=dict(argstr='-yorigin %s',
    ),
    zdel=dict(argstr='-zdel %f',
    ),
    zorigin=dict(argstr='-zorigin %s',
    ),
    )
    inputs = Refit.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Refit_outputs():
    output_map = dict(out_file=dict(keep_extension=False,
    name_source='in_file',
    name_template='%s',
    ),
    )
    outputs = Refit.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
