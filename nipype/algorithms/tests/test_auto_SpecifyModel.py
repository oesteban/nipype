# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..modelgen import SpecifyModel


def test_SpecifyModel_inputs():
    input_map = dict(event_files=dict(mandatory=True,
    xor=['subject_info', 'event_files'],
    ),
    functional_runs=dict(copyfile=False,
    mandatory=True,
    ),
    high_pass_filter_cutoff=dict(mandatory=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    input_units=dict(mandatory=True,
    ),
    outlier_files=dict(copyfile=False,
    ),
    realignment_parameters=dict(copyfile=False,
    ),
    subject_info=dict(mandatory=True,
    xor=['subject_info', 'event_files'],
    ),
    time_repetition=dict(mandatory=True,
    ),
    )
    inputs = SpecifyModel._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_SpecifyModel_outputs():
    output_map = dict(session_info=dict(),
    )
    outputs = SpecifyModel._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
