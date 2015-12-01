# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipype.testing import (assert_raises, assert_equal,
                            assert_true, assert_false)
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nipype.pipeline import engine as pe
from copy import deepcopy
import os.path as op
from tempfile import mkdtemp
from shutil import rmtree
import json


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

    # check result
    tmpfile = op.join(mkdtemp(), 'result.json')
    jsonsink = pe.Node(nio.JSONFileSink(input_names=['sum'],
                       out_file=tmpfile), name='sink')
    cwf.connect([(outputnode, jsonsink, [('out', 'sum')])])
    res = cwf.run()

    with open(tmpfile, 'r') as f:
        result = json.dumps(json.load(f))

    rmtree(op.dirname(tmpfile))
    yield assert_equal, result, '{"sum": 5}'


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

    # check result
    tmpfile = op.join(mkdtemp(), 'result.json')
    jsonsink = pe.Node(nio.JSONFileSink(input_names=['sum'],
                       out_file=tmpfile), name='sink')
    cwf.connect([(outputnode, jsonsink, [('out', 'sum')])])
    res = cwf.run()

    with open(tmpfile, 'r') as f:
        result = json.dumps(json.load(f))

    rmtree(op.dirname(tmpfile))
    yield assert_equal, result, '{"sum": 0}'


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

    # check result
    tmpfile = op.join(mkdtemp(), 'result.json')
    jsonsink = pe.Node(nio.JSONFileSink(input_names=['sum'],
                       out_file=tmpfile), name='sink')
    wf.connect([(cwf, jsonsink, [('outputnode.out', 'sum')])])
    res = wf.run()

    with open(tmpfile, 'r') as f:
        result = json.dumps(json.load(f))

    rmtree(op.dirname(tmpfile))
    yield assert_equal, result, '{"sum": 5}'


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

    outernode = pe.Node(niu.IdentityInterface(fields=['a', 'b', 'c']),
                        name='outer')
    wf = pe.Workflow('OuterWorkflow')
    wf.connect([
        (outernode, cwf, [('a', 'inputnode.a'), ('b', 'inputnode.b'),
                          ('c', 'conditions.c')])
    ])
    outernode.inputs.a = 2
    outernode.inputs.b = 3
    outernode.inputs.c = 7

    fg = wf._create_flat_graph()
    wf._set_needed_outputs(fg)
    eg = pe.generate_expanded_graph(deepcopy(fg))

    # when the condition is set, all nodes are removed
    yield assert_equal, len(eg.nodes()), 0

    # check result
    tmpfile = op.join(mkdtemp(), 'result.json')
    jsonsink = pe.Node(nio.JSONFileSink(input_names=['sum'],
                       out_file=tmpfile), name='sink')
    wf.connect([(cwf, jsonsink, [('outputnode.out', 'sum')])])
    res = wf.run()

    with open(tmpfile, 'r') as f:
        result = json.dumps(json.load(f))

    rmtree(op.dirname(tmpfile))
    yield assert_equal, result, '{"sum": 7}'