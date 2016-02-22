# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..diffusion import ResampleDTIVolume


def test_ResampleDTIVolume_inputs():
    input_map = dict(Inverse_ITK_Transformation=dict(argstr='--Inverse_ITK_Transformation ',
    ),
    Reference=dict(argstr='--Reference %s',
    ),
    args=dict(argstr='%s',
    ),
    centered_transform=dict(argstr='--centered_transform ',
    ),
    correction=dict(argstr='--correction %s',
    ),
    defField=dict(argstr='--defField %s',
    ),
    default_pixel_value=dict(argstr='--default_pixel_value %f',
    ),
    direction_matrix=dict(argstr='--direction_matrix %s',
    sep=',',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    hfieldtype=dict(argstr='--hfieldtype %s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    image_center=dict(argstr='--image_center %s',
    ),
    inputVolume=dict(argstr='%s',
    position=-2,
    ),
    interpolation=dict(argstr='--interpolation %s',
    ),
    notbulk=dict(argstr='--notbulk ',
    ),
    number_of_thread=dict(argstr='--number_of_thread %d',
    ),
    origin=dict(argstr='--origin %s',
    ),
    outputVolume=dict(argstr='%s',
    hash_files=False,
    position=-1,
    ),
    rotation_point=dict(argstr='--rotation_point %s',
    ),
    size=dict(argstr='--size %s',
    sep=',',
    ),
    spaceChange=dict(argstr='--spaceChange ',
    ),
    spacing=dict(argstr='--spacing %s',
    sep=',',
    ),
    spline_order=dict(argstr='--spline_order %d',
    ),
    terminal_output=dict(nohash=True,
    ),
    transform=dict(argstr='--transform %s',
    ),
    transform_matrix=dict(argstr='--transform_matrix %s',
    sep=',',
    ),
    transform_order=dict(argstr='--transform_order %s',
    ),
    transform_tensor_method=dict(argstr='--transform_tensor_method %s',
    ),
    transformationFile=dict(argstr='--transformationFile %s',
    ),
    window_function=dict(argstr='--window_function %s',
    ),
    )
    inputs = ResampleDTIVolume._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_ResampleDTIVolume_outputs():
    output_map = dict(outputVolume=dict(position=-1,
    ),
    )
    outputs = ResampleDTIVolume._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
