# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..registration import AnalyzeWarp


def test_AnalyzeWarp_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    num_threads=dict(argstr='-threads %01d',
    nohash=True,
    ),
    output_path=dict(argstr='-out %s',
    mandatory=True,
    usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    transform_file=dict(argstr='-tp %s',
    mandatory=True,
    ),
    )
    inputs = AnalyzeWarp.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_AnalyzeWarp_outputs():
    output_map = dict(disp_field=dict(),
    jacdet_map=dict(),
    jacmat_map=dict(),
    )
    outputs = AnalyzeWarp.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
