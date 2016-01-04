#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from copy import deepcopy
import os.path as op
from tempfile import mkdtemp
from shutil import rmtree

from ....testing import (assert_raises, assert_equal,
                         assert_true, assert_false)
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu


ifresult = None


class SetInputSpec(nib.TraitedSpec):
    val = nib.traits.Any(mandatory=True, desc='input')


class SetOutputSpec(nib.TraitedSpec):
    out = nib.traits.Any(desc='ouput')


class SetInterface(nib.BaseInterface):
    input_spec = SetInputSpec
    output_spec = SetOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        global ifresult
        runtime.returncode = 0
        ifresult = self.inputs.val
        return runtime

    def _list_outputs(self):
        global ifresult
        outputs = self._outputs().get()
        outputs['out'] = self.inputs.val
        return outputs


def _base_workflow(name='InterfacedWorkflow'):
    wf = pe.InterfacedWorkflow(
        name=name, input_names=['input0'], output_names=['output0'])
    
    mynode = pe.Node(SetInterface(), name='internalnode')
    wf.connect('in', 'input0', mynode, 'val')
    wf.connect(mynode, 'out', 'out', 'output0')
    return wf


def _sum_workflow(name='InterfacedSumWorkflow', b=0):
    name += '%02d' % b

    def _sum(a, b):
        return a + b + 1

    wf = pe.InterfacedWorkflow(
        name=name, input_names=['input0'],
        output_names=['output0'])
    sum0 = pe.Node(niu.Function(
        input_names=['a', 'b'], output_names=['out'], function=_sum),
        name='testnode')
    sum0.inputs.b = b

    # test connections
    wf.connect('in', 'input0', sum0, 'a')
    wf.connect(sum0, 'out', 'out', 'output0')
    return wf


def test_interfaced_workflow():
    global ifresult

    x = lambda: pe.InterfacedWorkflow(name='ShouldRaise')
    yield assert_raises, ValueError, x
    x = lambda: pe.InterfacedWorkflow(name='ShouldRaise',
                                      input_names=['input0'])
    yield assert_raises, ValueError, x
    x = lambda: pe.InterfacedWorkflow(name='ShouldRaise',
                                      output_names=['output0'])
    yield assert_raises, ValueError, x

    wf = pe.InterfacedWorkflow(
        name='InterfacedWorkflow', input_names=['input0'],
        output_names=['output0'])

    # Check it doesn't expose inputs/outputs of internal nodes
    inputs = wf.inputs.get()
    yield assert_equal, inputs, {'input0': nib.Undefined}

    outputs = wf.outputs.get()
    yield assert_equal, outputs, {'output0': None}

    # test connections
    mynode = pe.Node(SetInterface(), name='internalnode')
    wf.connect('in', 'input0', mynode, 'val')
    wf.connect(mynode, 'out', 'out', 'output0')

    # test setting input
    wf.inputs.input0 = 5
    yield assert_equal, wf.inputs.get(), {'input0': 5}

    wf.run()
    yield assert_equal, ifresult, 5

    # Try to create an outbound connection from an inner node
    wf = _base_workflow()
    outerwf = pe.Workflow('OuterWorkflow')
    outernode = pe.Node(niu.IdentityInterface(fields=['val']),
                        name='outernode')
    x = lambda: outerwf.connect(wf, 'internalnode.out', outernode, 'val')
    yield assert_raises, Exception, x

    # Try to create an inbound connection from an outer node
    wf = _base_workflow()
    outerwf = pe.Workflow('OuterWorkflow')
    outernode = pe.Node(niu.IdentityInterface(fields=['val']),
                        name='outernode')
    x = lambda: outerwf.connect(outernode, 'val', wf, 'internalnode.val')
    yield assert_raises, Exception, x

    # Try to insert a sub-workflow with an outbound connection
    outerwf = pe.Workflow('OuterWorkflow')
    outernode = pe.Node(niu.IdentityInterface(fields=['val']),
                        name='outernode')

    subwf = pe.Workflow('SubWorkflow')
    inputnode = pe.Node(niu.IdentityInterface(fields=['in']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']),
                         name='outputnode')
    subnode = pe.Node(SetInterface(), name='internalnode')
    subwf.connect([
        (inputnode, subnode, [('in', 'val')]),
        (subnode, outputnode, [('out', 'out')]),
    ])

    outerwf.connect(subwf, 'internalnode.out', outernode, 'val')

    wf = pe.InterfacedWorkflow(
        name='InterfacedWorkflow', input_names=['input0'],
        output_names=['output0'])
    x = lambda: wf.connect('in', 'input0', subwf, 'inputnode.in')
    yield assert_raises, Exception, x

    # Try to insert a sub-workflow with an inbound connection
    outerwf = pe.Workflow('OuterWorkflow')
    outernode = pe.Node(niu.IdentityInterface(fields=['val']),
                        name='outernode')

    subwf = pe.Workflow('SubWorkflow')
    inputnode = pe.Node(niu.IdentityInterface(fields=['in']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out']),
                         name='outputnode')
    subnode = pe.Node(SetInterface(), name='internalnode')
    subwf.connect([
        (subnode, outputnode, [('out', 'out')]),
    ])

    outerwf.connect(outernode, 'val', subwf, 'internalnode.val')

    wf = pe.InterfacedWorkflow(
        name='InterfacedWorkflow', input_names=['input0'],
        output_names=['output0'])
    x = lambda: wf.connect('in', 'input0', subwf, 'inputnode.in')
    yield assert_raises, Exception, x

def test_graft_workflow():
    global ifresult
    wf1 = _sum_workflow()
    wf = pe.GraftWorkflow(
        name='GraftWorkflow', fields_from=wf1)
    wf.insert(wf1)
    wf.insert(_sum_workflow(b=2))

    outer = pe.Workflow('OuterWorkflow')
    mynode = pe.Node(SetInterface(), name='internalnode')

    outer.connect([
        (wf, mynode, [('outputnode.output0', 'val')])
    ])

    wf.inputs.input0 = 3

    ifresult = None
    outer.run()
    yield assert_equal, ifresult, [4, 6]
