# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..io import DataGrabber


def test_DataGrabber_inputs():
    input_map = dict(base_directory=dict(),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    raise_on_empty=dict(usedefault=True,
    ),
    sort_filelist=dict(mandatory=True,
    ),
    template=dict(mandatory=True,
    ),
    template_args=dict(),
    )
    inputs = DataGrabber._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_DataGrabber_outputs():
    output_map = dict()
    outputs = DataGrabber._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
