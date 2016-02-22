# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..io import XNATSource


def test_XNATSource_inputs():
    input_map = dict(cache_dir=dict(),
    config=dict(mandatory=True,
    xor=['server'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    pwd=dict(),
    query_template=dict(mandatory=True,
    ),
    query_template_args=dict(usedefault=True,
    ),
    server=dict(mandatory=True,
    requires=['user', 'pwd'],
    xor=['config'],
    ),
    user=dict(),
    )
    inputs = XNATSource._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_XNATSource_outputs():
    output_map = dict()
    outputs = XNATSource._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
