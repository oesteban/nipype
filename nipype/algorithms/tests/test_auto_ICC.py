# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ...testing import assert_equal
from ..icc import ICC


def test_ICC_inputs():
    input_map = dict(icc_map=dict(),
    mask=dict(mandatory=True,
    ),
    session_F_map=dict(),
    session_var_map=dict(),
    subject_var_map=dict(),
    subjects_sessions=dict(mandatory=True,
    ),
    )
    inputs = ICC._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ICC_outputs():
    output_map = dict(icc_map=dict(),
    session_F_map=dict(),
    session_var_map=dict(),
    subject_var_map=dict(),
    )
    outputs = ICC._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
