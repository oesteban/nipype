# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..model import FEATModel


def test_FEATModel_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ev_files=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=1,
    ),
    fsf_file=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=0,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    output_type=dict(usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = FEATModel._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_FEATModel_outputs():
    output_map = dict(con_file=dict(),
    design_cov=dict(),
    design_file=dict(),
    design_image=dict(),
    fcon_file=dict(),
    )
    outputs = FEATModel._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
