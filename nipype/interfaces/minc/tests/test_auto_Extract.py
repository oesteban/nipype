# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..minc import Extract


def test_Extract_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    count=dict(argstr='-count %s',
    sep=',',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    flip_any_direction=dict(argstr='-any_direction',
    xor=('flip_positive_direction', 'flip_negative_direction', 'flip_any_direction'),
    ),
    flip_negative_direction=dict(argstr='-negative_direction',
    xor=('flip_positive_direction', 'flip_negative_direction', 'flip_any_direction'),
    ),
    flip_positive_direction=dict(argstr='-positive_direction',
    xor=('flip_positive_direction', 'flip_negative_direction', 'flip_any_direction'),
    ),
    flip_x_any=dict(argstr='-xanydirection',
    xor=('flip_x_positive', 'flip_x_negative', 'flip_x_any'),
    ),
    flip_x_negative=dict(argstr='-xdirection',
    xor=('flip_x_positive', 'flip_x_negative', 'flip_x_any'),
    ),
    flip_x_positive=dict(argstr='+xdirection',
    xor=('flip_x_positive', 'flip_x_negative', 'flip_x_any'),
    ),
    flip_y_any=dict(argstr='-yanydirection',
    xor=('flip_y_positive', 'flip_y_negative', 'flip_y_any'),
    ),
    flip_y_negative=dict(argstr='-ydirection',
    xor=('flip_y_positive', 'flip_y_negative', 'flip_y_any'),
    ),
    flip_y_positive=dict(argstr='+ydirection',
    xor=('flip_y_positive', 'flip_y_negative', 'flip_y_any'),
    ),
    flip_z_any=dict(argstr='-zanydirection',
    xor=('flip_z_positive', 'flip_z_negative', 'flip_z_any'),
    ),
    flip_z_negative=dict(argstr='-zdirection',
    xor=('flip_z_positive', 'flip_z_negative', 'flip_z_any'),
    ),
    flip_z_positive=dict(argstr='+zdirection',
    xor=('flip_z_positive', 'flip_z_negative', 'flip_z_any'),
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    image_maximum=dict(argstr='-image_maximum %s',
    ),
    image_minimum=dict(argstr='-image_minimum %s',
    ),
    image_range=dict(argstr='-image_range %s %s',
    ),
    input_file=dict(argstr='%s',
    mandatory=True,
    position=-2,
    ),
    nonormalize=dict(argstr='-nonormalize',
    xor=('normalize', 'nonormalize'),
    ),
    normalize=dict(argstr='-normalize',
    xor=('normalize', 'nonormalize'),
    ),
    out_file=dict(argstr='> %s',
    position=-1,
    usedefault=True,
    ),
    output_file=dict(hash_files=False,
    keep_extension=False,
    name_source=['input_file'],
    name_template='%s.raw',
    position=-1,
    ),
    start=dict(argstr='-start %s',
    sep=',',
    ),
    terminal_output=dict(nohash=True,
    ),
    write_ascii=dict(argstr='-ascii',
    xor=('write_ascii', 'write_ascii', 'write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double', 'write_signed', 'write_unsigned'),
    ),
    write_byte=dict(argstr='-byte',
    xor=('write_ascii', 'write_ascii', 'write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double', 'write_signed', 'write_unsigned'),
    ),
    write_double=dict(argstr='-double',
    xor=('write_ascii', 'write_ascii', 'write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double', 'write_signed', 'write_unsigned'),
    ),
    write_float=dict(argstr='-float',
    xor=('write_ascii', 'write_ascii', 'write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double', 'write_signed', 'write_unsigned'),
    ),
    write_int=dict(argstr='-int',
    xor=('write_ascii', 'write_ascii', 'write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double', 'write_signed', 'write_unsigned'),
    ),
    write_long=dict(argstr='-long',
    xor=('write_ascii', 'write_ascii', 'write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double', 'write_signed', 'write_unsigned'),
    ),
    write_range=dict(argstr='-range %s %s',
    ),
    write_short=dict(argstr='-short',
    xor=('write_ascii', 'write_ascii', 'write_byte', 'write_short', 'write_int', 'write_long', 'write_float', 'write_double', 'write_signed', 'write_unsigned'),
    ),
    write_signed=dict(argstr='-signed',
    xor=('write_signed', 'write_unsigned'),
    ),
    write_unsigned=dict(argstr='-unsigned',
    xor=('write_signed', 'write_unsigned'),
    ),
    )
    inputs = Extract._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Extract_outputs():
    output_map = dict(output_file=dict(),
    )
    outputs = Extract._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
