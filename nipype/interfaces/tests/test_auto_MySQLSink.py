# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..io import MySQLSink


def test_MySQLSink_inputs():
    input_map = dict(config=dict(mandatory=True,
    xor=['host'],
    ),
    database_name=dict(mandatory=True,
    ),
    host=dict(mandatory=True,
    requires=['username', 'password'],
    usedefault=True,
    xor=['config'],
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    password=dict(),
    table_name=dict(mandatory=True,
    ),
    username=dict(),
    )
    inputs = MySQLSink._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_MySQLSink_outputs():
    output_map = dict()
    outputs = MySQLSink._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
