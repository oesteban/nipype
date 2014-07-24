# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.utility import MultipleSelectInterface

def test_MultipleSelectInterface_inputs():
    input_map = dict(index=dict(mandatory=True,
    ),
    )
    inputs = MultipleSelectInterface.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_MultipleSelectInterface_outputs():
    output_map = dict()
    outputs = MultipleSelectInterface.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

