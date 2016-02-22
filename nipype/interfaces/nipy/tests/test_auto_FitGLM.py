# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..model import FitGLM


def test_FitGLM_inputs():
    input_map = dict(TR=dict(mandatory=True,
    ),
    drift_model=dict(usedefault=True,
    ),
    hrf_model=dict(usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    mask=dict(),
    method=dict(usedefault=True,
    ),
    model=dict(usedefault=True,
    ),
    normalize_design_matrix=dict(usedefault=True,
    ),
    plot_design_matrix=dict(usedefault=True,
    ),
    save_residuals=dict(usedefault=True,
    ),
    session_info=dict(mandatory=True,
    ),
    )
    inputs = FitGLM._input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_FitGLM_outputs():
    output_map = dict(a=dict(),
    axis=dict(),
    beta=dict(),
    constants=dict(),
    dof=dict(),
    nvbeta=dict(),
    reg_names=dict(),
    residuals=dict(),
    s2=dict(),
    )
    outputs = FitGLM._output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
