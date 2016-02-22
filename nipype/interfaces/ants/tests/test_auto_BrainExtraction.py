# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..segmentation import BrainExtraction


def test_BrainExtraction_inputs():
    input_map = dict(anatomical_image=dict(argstr='-a %s',
    mandatory=True,
    ),
    args=dict(argstr='%s',
    ),
    brain_probability_mask=dict(argstr='-m %s',
    copyfile=False,
    mandatory=True,
    ),
    brain_template=dict(argstr='-e %s',
    mandatory=True,
    ),
    debug=dict(argstr='-z 1',
    ),
    dimension=dict(argstr='-d %d',
    usedefault=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    extraction_registration_mask=dict(argstr='-f %s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    image_suffix=dict(argstr='-s %s',
    usedefault=True,
    ),
    keep_temporary_files=dict(argstr='-k %d',
    ),
    num_threads=dict(nohash=True,
    usedefault=True,
    ),
    out_prefix=dict(argstr='-o %s',
    usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    use_floatingpoint_precision=dict(argstr='-q %d',
    ),
    use_random_seeding=dict(argstr='-u %d',
    ),
    )
    inputs = BrainExtraction._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_BrainExtraction_outputs():
    output_map = dict(BrainExtractionBrain=dict(),
    BrainExtractionMask=dict(),
    )
    outputs = BrainExtraction._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
