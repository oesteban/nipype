# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import Warp


def test_Warp_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    deoblique=dict(argstr='-deoblique',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    gridset=dict(argstr='-gridset %s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=-1,
    ),
    interp=dict(argstr='-%s',
    ),
    matparent=dict(argstr='-matparent %s',
    ),
    mni2tta=dict(argstr='-mni2tta',
    ),
    newgrid=dict(argstr='-newgrid %f',
    ),
    out_file=dict(),
    outputtype=dict(usedefault=True,
    ),
    prefix=dict(argstr='-prefix %s',
    ),
    terminal_output=dict(nohash=True,
    ),
    tta2mni=dict(argstr='-tta2mni',
    ),
    zpad=dict(argstr='-zpad %d',
    ),
    )
    inputs = Warp._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Warp_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Warp._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
