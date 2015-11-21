# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipype.testing import (assert_raises, assert_equal, assert_true, assert_false)
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from copy import deepcopy


def test_cw_removal_cond_unset():
    def _sum(a, b):
        return a + b

    cwf = pe.ConditionalWorkflow(
        'TestConditionalWorkflow', condition_map=[('c', 'outputnode.out')])

    inputnode = pe.Node(niu.IdentityInterface(fields=['a', 'b']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']),
                         name='outputnode')
    sumnode = pe.Node(niu.Function(
        input_names=['a', 'b'], output_names=['sum'],
        function=_sum), name='SumNode')
    cwf.connect([
        (inputnode, sumnode, [('a', 'a'), ('b', 'b')]),
        (sumnode, outputnode, [('sum', 'out')])
    ])

    cwf.inputs.inputnode.a = 2
    cwf.inputs.inputnode.b = 3

    fg = cwf._create_flat_graph()
    cwf._set_needed_outputs(fg)
    eg = pe.generate_expanded_graph(deepcopy(fg))
    # when the condition is not set, the sumnode should remain
    yield assert_equal, len(eg.nodes()), 1


def test_cw_removal_cond_set():
    def _sum(a, b):
        return a + b

    cwf = pe.ConditionalWorkflow(
        'TestConditionalWorkflow', condition_map=[('c', 'outputnode.out')])

    inputnode = pe.Node(niu.IdentityInterface(fields=['a', 'b']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']),
                         name='outputnode')
    sumnode = pe.Node(niu.Function(
        input_names=['a', 'b'], output_names=['sum'],
        function=_sum), name='SumNode')
    cwf.connect([
        (inputnode, sumnode, [('a', 'a'), ('b', 'b')]),
        (sumnode, outputnode, [('sum', 'out')])
    ])

    cwf.inputs.inputnode.a = 2
    cwf.inputs.inputnode.b = 3
    cwf.conditions.c = 0

    fg = cwf._create_flat_graph()
    cwf._set_needed_outputs(fg)
    eg = pe.generate_expanded_graph(deepcopy(fg))
    # when the condition is set, all nodes are removed
    yield assert_equal, len(eg.nodes()), 0


def test_cw_removal_cond_connected_not_set():
    def _sum(a, b):
        return a + b

    cwf = pe.ConditionalWorkflow(
        'TestConditionalWorkflow', condition_map=[('c', 'outputnode.out')])

    inputnode = pe.Node(niu.IdentityInterface(fields=['a', 'b']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']),
                         name='outputnode')
    sumnode = pe.Node(niu.Function(
        input_names=['a', 'b'], output_names=['sum'],
        function=_sum), name='SumNode')
    cwf.connect([
        (inputnode, sumnode, [('a', 'a'), ('b', 'b')]),
        (sumnode, outputnode, [('sum', 'out')])
    ])

    cwf.inputs.inputnode.a = 2
    cwf.inputs.inputnode.b = 3

    outernode = pe.Node(niu.IdentityInterface(fields=['c']), name='outer')
    wf = pe.Workflow('OuterWorkflow')
    wf.connect(outernode, 'c', cwf, 'conditions.c')

    fg = wf._create_flat_graph()
    wf._set_needed_outputs(fg)
    eg = pe.generate_expanded_graph(deepcopy(fg))
    # when the condition is set, all nodes are removed
    yield assert_equal, len(eg.nodes()), 1


def test_cw_removal_cond_connected_and_set():
    def _sum(a, b):
        return a + b

    cwf = pe.ConditionalWorkflow(
        'TestConditionalWorkflow', condition_map=[('c', 'outputnode.out')])

    inputnode = pe.Node(niu.IdentityInterface(fields=['a', 'b']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']),
                         name='outputnode')
    sumnode = pe.Node(niu.Function(
        input_names=['a', 'b'], output_names=['sum'],
        function=_sum), name='SumNode')
    cwf.connect([
        (inputnode, sumnode, [('a', 'a'), ('b', 'b')]),
        (sumnode, outputnode, [('sum', 'out')])
    ])

    cwf.inputs.inputnode.a = 2
    cwf.inputs.inputnode.b = 3

    outernode = pe.Node(niu.IdentityInterface(fields=['c']), name='outer')
    wf = pe.Workflow('OuterWorkflow')
    wf.connect(outernode, 'c', cwf, 'conditions.c')
    outernode.inputs.c = 7

    fg = wf._create_flat_graph()
    wf._set_needed_outputs(fg)
    eg = pe.generate_expanded_graph(deepcopy(fg))
    # when the condition is set, all nodes are removed
    yield assert_equal, len(eg.nodes()), 0