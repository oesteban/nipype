#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Defines functionality for pipelined execution of interfaces

The `Workflow` class provides core functionality for batch processing.

   Change directory to provide relative paths for doctests
   >>> import os
   >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
   >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
   >>> os.chdir(datadir)

"""

from future import standard_library
standard_library.install_aliases()

from datetime import datetime
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from copy import deepcopy
import pickle
import os
import os.path as op
import shutil
import sys
from warnings import warn
import numpy as np
import networkx as nx

from nipype.utils.misc import (getsource, create_function_from_source,
                               package_check, str2bool)
from nipype.external.six import string_types
from nipype import config, logging

from nipype.interfaces.base import (traits, TraitedSpec, TraitDictObject,
                                    TraitListObject)
from nipype.interfaces.utility import IdentityInterface

from .utils import (make_output_dir, get_print_name, merge_dict)
from .graph import (generate_expanded_graph, export_graph, write_workflow_prov,
                    format_dot, topological_sort)
from .base import NodeBase
from .nodes import Node, MapNode

logger = logging.getLogger('workflow')
package_check('networkx', '1.3')


class Workflow(NodeBase):
    """Controls the setup and execution of a pipeline of processes."""

    def __init__(self, name, base_dir=None):
        """Create a workflow object.

        Parameters
        ----------
        name : alphanumeric string
            unique identifier for the workflow
        base_dir : string, optional
            path to workflow storage

        """
        super(Workflow, self).__init__(name, base_dir)
        self._graph = nx.DiGraph()
        self.config = deepcopy(config._sections)
        self._signalnode = Node(IdentityInterface(
            fields=self.signals.copyable_trait_names()), 'signalnode')
        self.add_nodes([self._signalnode])

        # Automatically initialize signal
        for s in self.signals.copyable_trait_names():
            setattr(self._signalnode.inputs, s, getattr(self.signals, s))

    def _update_disable(self):
        logger.debug('Signal disable is now %s for workflow %s' %
                     (self.signals.disable, self.fullname))
        self._signalnode.inputs.disable = self.signals.disable

    # PUBLIC API
    def clone(self, name):
        """Clone a workflow

        .. note::

          Will reset attributes used for executing workflow. See
          _init_runtime_fields.

        Parameters
        ----------

        name: alphanumeric name
            unique name for the workflow

        """
        clone = super(Workflow, self).clone(name)
        clone._reset_hierarchy()
        return clone

    # Graph creation functions
    def connect(self, *args, **kwargs):
        """Connect nodes in the pipeline.

        This routine also checks if inputs and outputs are actually provided by
        the nodes that are being connected.

        Creates edges in the directed graph using the nodes and edges specified
        in the `connection_list`.  Uses the NetworkX method
        DiGraph.add_edges_from.

        Parameters
        ----------

        args : list or a set of four positional arguments

            Four positional arguments of the form::

              connect(source, sourceoutput, dest, destinput)

            source : nodewrapper node
            sourceoutput : string (must be in source.outputs)
            dest : nodewrapper node
            destinput : string (must be in dest.inputs)

            A list of 3-tuples of the following form::

             [(source, target,
                 [('sourceoutput/attribute', 'targetinput'),
                 ...]),
             ...]

            Or::

             [(source, target, [(('sourceoutput1', func, arg2, ...),
                                         'targetinput'), ...]),
             ...]
             sourceoutput1 will always be the first argument to func
             and func will be evaluated and the results sent ot targetinput

             currently func needs to define all its needed imports within the
             function as we use the inspect module to get at the source code
             and execute it remotely
        """
        if len(args) == 1:
            connection_list = args[0]
        elif len(args) == 4:
            connection_list = [(args[0], args[2], [(args[1], args[3])])]
        else:
            raise TypeError('connect() takes either 4 arguments, or 1 list of'
                            ' connection tuples (%d args given)' % len(args))
        if not kwargs:
            kwargs = {}

        disconnect = kwargs.get('disconnect', False)

        if disconnect:
            self.disconnect(connection_list)
            return

        conn_type = kwargs.get('conn_type', 'data')
        logger.debug('connect(disconnect=%s, conn_type=%s): %s' %
                     (disconnect, conn_type, connection_list))

        # Check if nodes are already in the graph
        newnodes = []
        for srcnode, dstnode, _ in connection_list:
            if self in [srcnode, dstnode]:
                msg = ('Workflow connect cannot contain itself as node:'
                       ' src[%s] dest[%s] workflow[%s]') % (srcnode,
                                                            dstnode,
                                                            self.name)

                raise IOError(msg)
            if (srcnode not in newnodes) and not self._has_node(srcnode):
                newnodes.append(srcnode)
            if (dstnode not in newnodes) and not self._has_node(dstnode):
                newnodes.append(dstnode)
        if newnodes:
            logger.debug('New nodes: %s' % newnodes)
            self._check_nodes(newnodes)
            for node in newnodes:
                if node._hierarchy is None:
                    node._hierarchy = self.name

        # check correctness of required connections
        not_found = []
        connected_ports = {}
        for srcnode, dstnode, connects in connection_list:
            logger.debug('Checking connection %s to %s' % (srcnode, dstnode))

            if dstnode not in connected_ports:
                connected_ports[dstnode] = []
            # check to see which ports of dstnode are already
            # connected.
            if dstnode in self._graph.nodes():
                for edge in self._graph.in_edges_iter(dstnode):
                    data = self._graph.get_edge_data(*edge)
                    for sourceinfo, destname, _ in data['connect']:
                        if destname not in connected_ports[dstnode]:
                            connected_ports[dstnode] += [destname]
            for source, dest in connects:
                # Currently datasource/sink/grabber.io modules
                # determine their inputs/outputs depending on
                # connection settings.  Skip these modules in the check
                if dest in connected_ports[dstnode]:
                    raise Exception('Already connected (%s.%s -> %s.%s' % (
                        srcnode, source, dstnode, dest))
                if not (hasattr(dstnode, '_interface') and
                        '.io' in str(dstnode._interface.__class__)):
                    if not dstnode._check_inputs(dest):
                        not_found.append(['in', '%s' % dstnode, dest])
                if not (hasattr(srcnode, '_interface') and
                        '.io' in str(srcnode._interface.__class__)):
                    if isinstance(source, tuple):
                        # handles the case that source is specified
                        # with a function
                        sourcename = source[0]
                    elif isinstance(source, string_types):
                        sourcename = source
                    else:
                        raise Exception(('Unknown source specification in '
                                         'connection from output of %s') %
                                        srcnode.name)
                    if sourcename and not srcnode._check_outputs(sourcename):
                        not_found.append(['out', '%s' % srcnode, sourcename])
                connected_ports[dstnode] += [dest]
        infostr = []
        for info in not_found:
            infostr += ["Module %s has no %sput called %s\n" % (info[1],
                                                                info[0],
                                                                info[2])]
        if not_found:
            infostr.insert(
                0, 'Some connections were not found connecting %s.%s to '
                '%s.%s' % (srcnode, source, dstnode, dest))
            raise Exception('\n'.join(infostr))

        # turn functions into strings
        for srcnode, dstnode, connects in connection_list:
            for idx, (src, dest) in enumerate(connects):
                if isinstance(src, tuple) and not isinstance(src[1], string_types):
                    function_source = getsource(src[1])
                    connects[idx] = ((src[0], function_source, src[2:]), dest)

        # add connections
        for srcnode, dstnode, connects in connection_list:
            edge_data = self._graph.get_edge_data(
                srcnode, dstnode, {'connect': []})

            msg = 'No existing connections' if not edge_data['connect'] else \
                'Previous connections exist'
            msg += ' from %s to %s %s' % (srcnode.fullname, dstnode.fullname,
                                          connects)
            logger.debug(msg)

            edge_data['connect'] += [(c[0], c[1], conn_type)
                                     for c in connects]
            logger.debug('(%s, %s): new edge data: %s' %
                         (srcnode, dstnode, str(edge_data)))

            self._graph.add_edges_from([(srcnode, dstnode, edge_data)])
            # edge_data = self._graph.get_edge_data(srcnode, dstnode)

    def disconnect(self, *args):
        """Disconnect nodes
        See the docstring for connect for format.
        """
        if len(args) == 1:
            connection_list = args[0]
        elif len(args) == 4:
            connection_list = [(args[0], args[2], [(args[1], args[3])])]
        else:
            raise TypeError('disconnect() takes either 4 arguments, or 1 list '
                            'of connection tuples (%d args given)' % len(args))

        for srcnode, dstnode, conn in connection_list:
            logger.debug('disconnect(): %s->%s %s' % (srcnode, dstnode, conn))
            if self in [srcnode, dstnode]:
                raise IOError(
                    'Workflow connect cannot contain itself as node: src[%s] '
                    'dest[%s] workflow[%s]') % (srcnode, dstnode, self.name)

            # If node is not in the graph, not connected
            if not self._has_node(srcnode) or not self._has_node(dstnode):
                continue

            edge_data = self._graph.get_edge_data(
                srcnode, dstnode, {'connect': []})
            ed_conns = [(c[0], c[1]) for c in edge_data['connect']]
            ed_meta = [c[2] for c in edge_data['connect']]

            remove = []
            for edge in conn:
                if edge in ed_conns:
                    idx = ed_conns.index(edge)
                    remove.append((edge[0], edge[1], ed_meta[idx]))

            logger.debug('disconnect(): remove list %s' % remove)
            for el in remove:
                edge_data['connect'].remove(el)
                logger.debug('disconnect(): removed connection %s' % str(el))

            if not edge_data['connect']:
                self._graph.remove_edge(srcnode, dstnode)
            else:
                self._graph.add_edges_from(
                    [(srcnode, dstnode, edge_data)])

    def add_nodes(self, nodes):
        """ Add nodes to a workflow

        Parameters
        ----------
        nodes : list
            A list of NodeBase-based objects
        """
        newnodes = []
        all_nodes = self._get_all_nodes()
        for node in nodes:
            if self._has_node(node):
                raise IOError('Node %s already exists in the workflow' % node)
            if isinstance(node, Workflow):
                for subnode in node._get_all_nodes():
                    if subnode in all_nodes:
                        raise IOError(('Subnode %s of node %s already exists '
                                       'in the workflow') % (subnode, node))
            newnodes.append(node)
        if not newnodes:
            logger.debug('no new nodes to add')
            return
        for node in newnodes:
            if not issubclass(node.__class__, NodeBase):
                raise Exception('Node %s must be a subclass of NodeBase' %
                                str(node))
        self._check_nodes(newnodes)
        for node in newnodes:
            if node._hierarchy is None:
                node._hierarchy = self.name
        self._graph.add_nodes_from(newnodes)

    def remove_nodes(self, nodes):
        """ Remove nodes from a workflow

        Parameters
        ----------
        nodes : list
            A list of NodeBase-based objects
        """
        self._graph.remove_nodes_from(nodes)

    # Input-Output access
    @property
    def inputs(self):
        return self._get_inputs()

    @property
    def outputs(self):
        return self._get_outputs()

    def get_node(self, name):
        """Return an internal node by name
        """
        nodenames = name.split('.')
        nodename = nodenames[0]
        outnode = [node for node in self._graph.nodes() if
                   str(node).endswith('.' + nodename)]
        if outnode:
            outnode = outnode[0]
            if nodenames[1:] and issubclass(outnode.__class__, Workflow):
                outnode = outnode.get_node('.'.join(nodenames[1:]))
        else:
            outnode = None
        return outnode

    def list_node_names(self):
        """List names of all nodes in a workflow
        """
        outlist = []
        for node in nx.topological_sort(self._graph):
            if isinstance(node, Workflow):
                outlist.extend(['.'.join((node.name, nodename)) for nodename in
                                node.list_node_names()])
            else:
                outlist.append(node.name)
        return sorted(outlist)

    def write_graph(self, dotfilename='graph.dot', graph2use='hierarchical',
                    format="png", simple_form=True):
        """Generates a graphviz dot file and a png file

        Parameters
        ----------

        graph2use: 'orig', 'hierarchical' (default), 'flat', 'exec', 'colored'
            orig - creates a top level graph without expanding internal
            workflow nodes;
            flat - expands workflow nodes recursively;
            hierarchical - expands workflow nodes recursively with a
            notion on hierarchy;
            colored - expands workflow nodes recursively with a
            notion on hierarchy in color;
            exec - expands workflows to depict iterables

        format: 'png', 'svg'

        simple_form: boolean (default: True)
            Determines if the node name used in the graph should be of the form
            'nodename (package)' when True or 'nodename.Class.package' when
            False.

        """
        self._connect_signals()

        graphtypes = ['orig', 'flat', 'hierarchical', 'exec', 'colored']
        if graph2use not in graphtypes:
            raise ValueError('Unknown graph2use keyword. Must be one of: ' +
                             str(graphtypes))
        base_dir, dotfilename = op.split(dotfilename)
        if base_dir == '':
            if self.base_dir:
                base_dir = self.base_dir
                if self.name:
                    base_dir = op.join(base_dir, self.name)
            else:
                base_dir = os.getcwd()
        base_dir = make_output_dir(base_dir)
        if graph2use in ['hierarchical', 'colored']:
            dotfilename = op.join(base_dir, dotfilename)
            self.write_hierarchical_dotfile(dotfilename=dotfilename,
                                            colored=graph2use == "colored",
                                            simple_form=simple_form)
            format_dot(dotfilename, format=format)
        else:
            graph = self._graph
            if graph2use in ['flat', 'exec']:
                graph = self._create_flat_graph()
            if graph2use == 'exec':
                graph = generate_expanded_graph(deepcopy(graph))
            export_graph(graph, base_dir, dotfilename=dotfilename,
                         format=format, simple_form=simple_form)

    def write_hierarchical_dotfile(self, dotfilename=None, colored=False,
                                   simple_form=True):
        dotlist = ['digraph %s{' % self.name]
        dotlist.append(self._get_dot(prefix='  ', colored=colored,
                                     simple_form=simple_form))
        dotlist.append('}')
        dotstr = '\n'.join(dotlist)
        if dotfilename:
            fp = open(dotfilename, 'wt')
            fp.writelines(dotstr)
            fp.close()
        else:
            logger.info(dotstr)

    def export(self, filename=None, prefix="output", format="python",
               include_config=False):
        """Export object into a different format

        Parameters
        ----------
        filename: string
           file to save the code to; overrides prefix
        prefix: string
           prefix to use for output file
        format: string
           one of "python"
        include_config: boolean
           whether to include node and workflow config values

        """
        from utils import format_node

        formats = ["python"]
        if format not in formats:
            raise ValueError('format must be one of: %s' % '|'.join(formats))
        flatgraph = self._create_flat_graph()
        nodes = nx.topological_sort(flatgraph)

        lines = ['# Workflow']
        importlines = ['from nipype.pipeline.engine import Workflow, '
                       'Node, MapNode']
        functions = {}
        if format == "python":
            connect_template = '%s.connect(%%s, %%s, %%s, "%%s")' % self.name
            connect_template2 = '%s.connect(%%s, "%%s", %%s, "%%s")' \
                                % self.name
            wfdef = '%s = Workflow("%s")' % (self.name, self.name)
            lines.append(wfdef)
            if include_config:
                lines.append('%s.config = %s' % (self.name, self.config))
            for idx, node in enumerate(nodes):
                nodename = node.fullname.replace('.', '_')
                # write nodes
                nodelines = format_node(node, format='python',
                                        include_config=include_config)
                for line in nodelines:
                    if line.startswith('from'):
                        if line not in importlines:
                            importlines.append(line)
                    else:
                        lines.append(line)
                # write connections
                for u, _, d in flatgraph.in_edges_iter(nbunch=node,
                                                       data=True):
                    for cd in d['connect']:
                        if isinstance(cd[0], tuple):
                            args = list(cd[0])
                            if args[1] in functions:
                                funcname = functions[args[1]]
                            else:
                                func = create_function_from_source(args[1])
                                funcname = [name for name in func.__globals__
                                            if name != '__builtins__'][0]
                                functions[args[1]] = funcname
                            args[1] = funcname
                            args = tuple([arg for arg in args if arg])
                            line_args = (u.fullname.replace('.', '_'),
                                         args, nodename, cd[1])
                            line = connect_template % line_args
                            line = line.replace("'%s'" % funcname, funcname)
                            lines.append(line)
                        else:
                            line_args = (u.fullname.replace('.', '_'),
                                         cd[0], nodename, cd[1])
                            lines.append(connect_template2 % line_args)
            functionlines = ['# Functions']
            for function in functions:
                functionlines.append(pickle.loads(function).rstrip())
            all_lines = importlines + functionlines + lines

            if not filename:
                filename = '%s%s.py' % (prefix, self.name)
            with open(filename, 'wt') as fp:
                fp.writelines('\n'.join(all_lines))
        return all_lines

    def run(self, plugin=None, plugin_args=None, updatehash=False):
        """ Execute the workflow

        Parameters
        ----------

        plugin: plugin name or object
            Plugin to use for execution. You can create your own plugins for
            execution.
        plugin_args : dictionary containing arguments to be sent to plugin
            constructor. see individual plugin doc strings for details.
        """
        if plugin is None:
            plugin = config.get('execution', 'plugin')
        if not isinstance(plugin, string_types):
            runner = plugin
        else:
            name = 'nipype.pipeline.plugins'
            try:
                __import__(name)
            except ImportError:
                msg = 'Could not import plugin module: %s' % name
                logger.error(msg)
                raise ImportError(msg)
            else:
                plugin_mod = getattr(sys.modules[name], '%sPlugin' % plugin)
                runner = plugin_mod(plugin_args=plugin_args)
        flatgraph = self._create_flat_graph()
        self.config = merge_dict(deepcopy(config._sections), self.config)
        if 'crashdump_dir' in self.config:
            warn(("Deprecated: workflow.config['crashdump_dir']\n"
                  "Please use config['execution']['crashdump_dir']"))
            crash_dir = self.config['crashdump_dir']
            self.config['execution']['crashdump_dir'] = crash_dir
            del self.config['crashdump_dir']
        logger.info(str(sorted(self.config)))
        self._set_needed_outputs(flatgraph)
        execgraph = generate_expanded_graph(deepcopy(flatgraph))
        for index, node in enumerate(execgraph.nodes()):
            node.config = merge_dict(deepcopy(self.config), node.config)
            node.base_dir = self.base_dir
            node.index = index
            if isinstance(node, MapNode):
                node.use_plugin = (plugin, plugin_args)
        self._configure_exec_nodes(execgraph)
        if str2bool(self.config['execution']['create_report']):
            self._write_report_info(self.base_dir, self.name, execgraph)
        runner.run(execgraph, updatehash=updatehash, config=self.config)
        datestr = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        if str2bool(self.config['execution']['write_provenance']):
            prov_base = op.join(self.base_dir,
                                'workflow_provenance_%s' % datestr)
            logger.info('Provenance file prefix: %s' % prov_base)
            write_workflow_prov(execgraph, prov_base, format='all')
        return execgraph

    # PRIVATE API AND FUNCTIONS

    def _write_report_info(self, workingdir, name, graph):
        from nipype.utils.filemanip import save_json
        if workingdir is None:
            workingdir = os.getcwd()
        report_dir = op.join(workingdir, name)
        if not op.exists(report_dir):
            os.makedirs(report_dir)
        shutil.copyfile(op.join(op.dirname(__file__),
                                'report_template.html'),
                        op.join(report_dir, 'index.html'))
        shutil.copyfile(op.join(op.dirname(__file__),
                                '..', '..', 'external', 'd3.js'),
                        op.join(report_dir, 'd3.js'))
        nodes, groups = topological_sort(graph, depth_first=True)
        graph_file = op.join(report_dir, 'graph1.json')
        json_dict = {'nodes': [], 'links': [], 'groups': [], 'maxN': 0}
        for i, node in enumerate(nodes):
            report_file = "%s/_report/report.rst" % \
                          node.output_dir().replace(report_dir, '')
            result_file = "%s/result_%s.pklz" % \
                          (node.output_dir().replace(report_dir, ''),
                           node.name)
            json_dict['nodes'].append(dict(name='%d_%s' % (i, node.name),
                                           report=report_file,
                                           result=result_file,
                                           group=groups[i]))
        maxN = 0
        for gid in np.unique(groups):
            procs = [i for i, val in enumerate(groups) if val == gid]
            N = len(procs)
            if N > maxN:
                maxN = N
            json_dict['groups'].append(dict(procs=procs,
                                            total=N,
                                            name='Group_%05d' % gid))
        json_dict['maxN'] = maxN
        for u, v in graph.in_edges_iter():
            json_dict['links'].append(dict(source=nodes.index(u),
                                           target=nodes.index(v),
                                           value=1))
        save_json(graph_file, json_dict)
        graph_file = op.join(report_dir, 'graph.json')
        template = '%%0%dd_' % np.ceil(np.log10(len(nodes))).astype(int)

        def getname(u, i):
            name_parts = u.fullname.split('.')
            # return '.'.join(name_parts[:-1] + [template % i + name_parts[-1]])
            return template % i + name_parts[-1]
        json_dict = []
        for i, node in enumerate(nodes):
            imports = []
            for u, v in graph.in_edges_iter(nbunch=node):
                imports.append(getname(u, nodes.index(u)))
            json_dict.append(dict(name=getname(node, i),
                                  size=1,
                                  group=groups[i],
                                  imports=imports))
        save_json(graph_file, json_dict)

    def _set_needed_outputs(self, graph):
        """Initialize node with list of which outputs are needed."""
        rm_outputs = self.config['execution']['remove_unnecessary_outputs']
        if not str2bool(rm_outputs):
            return
        for node in graph.nodes():
            node.needed_outputs = []
            for edge in graph.out_edges_iter(node):
                data = graph.get_edge_data(*edge)
                sourceinfo = [v1[0] if isinstance(v1, tuple) else v1
                              for v1, v2, _ in data['connect']]
                node.needed_outputs += [v for v in sourceinfo
                                        if v not in node.needed_outputs]
            if node.needed_outputs:
                node.needed_outputs = sorted(node.needed_outputs)

    def _configure_exec_nodes(self, graph):
        """Ensure that each node knows where to get inputs from
        """
        for node in graph.nodes():
            node.input_source = {}
            for edge in graph.in_edges_iter(node):
                data = graph.get_edge_data(*edge)
                for conn in sorted(data['connect']):
                    sourceinfo, field = conn[0], conn[1]
                    node.input_source[field] = \
                        (op.join(edge[0].output_dir(),
                                 'result_%s.pklz' % edge[0].name),
                         sourceinfo)

    def _check_nodes(self, nodes):
        """Checks if any of the nodes are already in the graph

        """
        node_names = [node.name for node in self._graph.nodes()]
        node_lineage = [node._hierarchy for node in self._graph.nodes()]
        for node in nodes:
            if node.name in node_names:
                idx = node_names.index(node.name)
                if node_lineage[idx] in [node._hierarchy, self.name]:
                    raise IOError('Duplicate node %s found.' % node)
            else:
                node_names.append(node.name)

    def _has_attr(self, parameter, subtype='in'):
        """Checks if a parameter is available as an input or output
        """
        if subtype == 'in':
            subobject = self.inputs
        else:
            subobject = self.outputs
        attrlist = parameter.split('.')
        cur_out = subobject
        for attr in attrlist:
            if not hasattr(cur_out, attr):
                return False
            cur_out = getattr(cur_out, attr)
        return True

    def _get_parameter_node(self, parameter, subtype='in'):
        """Returns the underlying node corresponding to an input or
        output parameter
        """
        if subtype == 'in':
            subobject = self.inputs
        else:
            subobject = self.outputs
        attrlist = parameter.split('.')
        cur_out = subobject
        for attr in attrlist[:-1]:
            cur_out = getattr(cur_out, attr)
        return cur_out.traits()[attrlist[-1]].node

    def _check_outputs(self, parameter):
        return self._has_attr(parameter, subtype='out')

    def _check_inputs(self, parameter):
        return self._has_attr(parameter, subtype='in')

    def _get_inputs(self):
        """Returns the inputs of a workflow

        This function does not return any input ports that are already
        connected
        """
        inputdict = TraitedSpec()
        for node in self._graph.nodes():
            inputdict.add_trait(node.name, traits.Instance(TraitedSpec))
            if isinstance(node, Workflow):
                setattr(inputdict, node.name, node.inputs)
            else:
                taken_inputs = []
                for _, _, d in self._graph.in_edges_iter(nbunch=node,
                                                         data=True):
                    for cd in d['connect']:
                        taken_inputs.append(cd[1])
                unconnectedinputs = TraitedSpec()
                for key, trait in list(node.inputs.items()):
                    if key not in taken_inputs:
                        unconnectedinputs.add_trait(key,
                                                    traits.Trait(trait,
                                                                 node=node))
                        value = getattr(node.inputs, key)
                        setattr(unconnectedinputs, key, value)
                setattr(inputdict, node.name, unconnectedinputs)
                getattr(inputdict, node.name).on_trait_change(self._set_input)
        return inputdict

    def _get_outputs(self):
        """Returns all possible output ports that are not already connected
        """
        outputdict = TraitedSpec()
        for node in self._graph.nodes():
            outputdict.add_trait(node.name, traits.Instance(TraitedSpec))
            if isinstance(node, Workflow):
                setattr(outputdict, node.name, node.outputs)
            elif node.outputs:
                outputs = TraitedSpec()
                for key, _ in list(node.outputs.items()):
                    outputs.add_trait(key, traits.Any(node=node))
                    setattr(outputs, key, None)
                setattr(outputdict, node.name, outputs)
        return outputdict

    def _set_input(self, object, name, newvalue):
        """Trait callback function to update a node input
        """
        logger.debug('_set_input(%s, %s) on %s.' % (
            name, newvalue, self.fullname))
        object.traits()[name].node.set_input(name, newvalue)

    def _set_node_input(self, node, param, source, sourceinfo):
        """Set inputs of a node given the edge connection"""
        if isinstance(sourceinfo, string_types):
            val = source.get_output(sourceinfo)
        elif isinstance(sourceinfo, tuple):
            if callable(sourceinfo[1]):
                val = sourceinfo[1](source.get_output(sourceinfo[0]),
                                    *sourceinfo[2:])
        newval = val
        if isinstance(val, TraitDictObject):
            newval = dict(val)
        if isinstance(val, TraitListObject):
            newval = val[:]
        logger.debug('setting node input: %s->%s', param, str(newval))
        node.set_input(param, deepcopy(newval))

    def _get_all_nodes(self):
        allnodes = []
        for node in self._graph.nodes():
            if isinstance(node, Workflow):
                allnodes.extend(node._get_all_nodes())
            else:
                allnodes.append(node)
        return allnodes

    def _has_node(self, wanted_node):
        for node in self._graph.nodes():
            if wanted_node == node:
                return True
            if isinstance(node, Workflow):
                if node._has_node(wanted_node):
                    return True
        return False

    def _connect_signals(self):
        signals = self.signals.copyable_trait_names()

        for node in self._graph.nodes():
            if node == self._signalnode:
                continue

            if node.signals is None:
                continue

            prefix = ''
            if isinstance(node, Workflow):
                node._connect_signals()
                prefix = 'signalnode.'

            for s in signals:
                sdest = prefix + s
                self.connect(self._signalnode, s, node, sdest,
                             conn_type='control')

    def _create_flat_graph(self):
        """Make a simple DAG where no node is a workflow."""
        logger.debug('Creating flat graph for workflow: %s', self.name)
        workflowcopy = deepcopy(self)
        workflowcopy._generate_flatgraph()
        return workflowcopy._graph

    def _reset_hierarchy(self):
        """Reset the hierarchy on a graph
        """
        for node in self._graph.nodes():
            if isinstance(node, Workflow):
                node._reset_hierarchy()
                for innernode in node._graph.nodes():
                    innernode._hierarchy = '.'.join((self.name,
                                                     innernode._hierarchy))
            else:
                node._hierarchy = self.name

    def _generate_flatgraph(self):
        """Generate a graph containing only Nodes or MapNodes
        """
        logger.debug('expanding workflow: %s', self)
        nodes2remove = []
        if not nx.is_directed_acyclic_graph(self._graph):
            raise Exception(('Workflow: %s is not a directed acyclic graph '
                             '(DAG)') % self.name)
        nodes = nx.topological_sort(self._graph)
        for node in nodes:
            logger.debug('processing node: %s' % node)
            if isinstance(node, Workflow):
                nodes2remove.append(node)
                # use in_edges instead of in_edges_iter to allow
                # disconnections to take place properly. otherwise, the
                # edge dict is modified.
                for u, _, d in self._graph.in_edges(nbunch=node, data=True):
                    logger.debug('in: connections-> %s' % str(d['connect']))
                    for cd in deepcopy(d['connect']):
                        logger.debug("in: %s" % str(cd))
                        dstnode = node._get_parameter_node(cd[1], subtype='in')
                        srcnode = u
                        srcout = cd[0]
                        dstin = cd[1].split('.')[-1]
                        logger.debug('in edges: %s %s %s %s' %
                                     (srcnode, srcout, dstnode, dstin))
                        self.disconnect(u, cd[0], node, cd[1])
                        self.connect(srcnode, srcout, dstnode, dstin,
                                     conn_type=cd[2])
                # do not use out_edges_iter for reasons stated in in_edges
                for _, v, d in self._graph.out_edges(nbunch=node, data=True):
                    logger.debug('out: connections-> %s' % str(d['connect']))
                    for cd in deepcopy(d['connect']):
                        logger.debug("out: %s" % str(cd))
                        dstnode = v
                        if isinstance(cd[0], tuple):
                            parameter = cd[0][0]
                        else:
                            parameter = cd[0]
                        srcnode = node._get_parameter_node(parameter,
                                                           subtype='out')
                        if isinstance(cd[0], tuple):
                            srcout = list(cd[0])
                            srcout[0] = parameter.split('.')[-1]
                            srcout = tuple(srcout)
                        else:
                            srcout = parameter.split('.')[-1]
                        dstin = cd[1]
                        logger.debug('out edges: %s %s %s %s' % (srcnode,
                                                                 srcout,
                                                                 dstnode,
                                                                 dstin))
                        self.disconnect(node, cd[0], v, cd[1])
                        self.connect(srcnode, srcout, dstnode, dstin)
                # expand the workflow node
                # logger.debug('expanding workflow: %s', node)
                node._generate_flatgraph()
                for innernode in node._graph.nodes():
                    innernode._hierarchy = '.'.join((self.name,
                                                     innernode._hierarchy))
                self._graph.add_nodes_from(node._graph.nodes())
                self._graph.add_edges_from(node._graph.edges(data=True))
        if nodes2remove:
            self._graph.remove_nodes_from(nodes2remove)
        logger.debug('finished expanding workflow: %s', self)

    def _get_dot(self, prefix=None, hierarchy=None, colored=False,
                 simple_form=True, level=0):
        """Create a dot file with connection info
        """
        if prefix is None:
            prefix = '  '
        if hierarchy is None:
            hierarchy = []
        colorset = ['#FFFFC8', '#0000FF', '#B4B4FF', '#E6E6FF', '#FF0000',
                    '#FFB4B4', '#FFE6E6', '#00A300', '#B4FFB4', '#E6FFE6']

        dotlist = ['%slabel="%s";' % (prefix, self.name)]
        for node in nx.topological_sort(self._graph):
            fullname = '.'.join(hierarchy + [node.fullname])
            nodename = fullname.replace('.', '_')
            if not isinstance(node, Workflow):
                node_class_name = get_print_name(node, simple_form=simple_form)
                if not simple_form:
                    node_class_name = '.'.join(node_class_name.split('.')[1:])
                if hasattr(node, 'iterables') and node.iterables:
                    dotlist.append(('%s[label="%s", shape=box3d,'
                                    'style=filled, color=black, colorscheme'
                                    '=greys7 fillcolor=2];') % (nodename,
                                                                node_class_name))
                else:
                    if colored:
                        dotlist.append(('%s[label="%s", style=filled,'
                                        ' fillcolor="%s"];')
                                       % (nodename, node_class_name,
                                           colorset[level]))
                    else:
                        dotlist.append(('%s[label="%s"];')
                                       % (nodename, node_class_name))

        for node in nx.topological_sort(self._graph):
            if isinstance(node, Workflow):
                fullname = '.'.join(hierarchy + [node.fullname])
                nodename = fullname.replace('.', '_')
                dotlist.append('subgraph cluster_%s {' % nodename)
                if colored:
                    dotlist.append(prefix + prefix + 'edge [color="%s"];' % (colorset[level + 1]))
                    dotlist.append(prefix + prefix + 'style=filled;')
                    dotlist.append(prefix + prefix + 'fillcolor="%s";' % (colorset[level + 2]))
                dotlist.append(node._get_dot(prefix=prefix + prefix,
                                             hierarchy=hierarchy + [self.name],
                                             colored=colored,
                                             simple_form=simple_form, level=level + 3))
                dotlist.append('}')
                if level == 6:
                    level = 2
            else:
                for subnode in self._graph.successors_iter(node):
                    if node._hierarchy != subnode._hierarchy:
                        continue
                    if not isinstance(subnode, Workflow):
                        nodefullname = '.'.join(hierarchy + [node.fullname])
                        subnodefullname = '.'.join(hierarchy +
                                                   [subnode.fullname])
                        nodename = nodefullname.replace('.', '_')
                        subnodename = subnodefullname.replace('.', '_')
                        for _ in self._graph.get_edge_data(node,
                                                           subnode)['connect']:
                            dotlist.append('%s -> %s;' % (nodename,
                                                          subnodename))
                        logger.debug('connection: ' + dotlist[-1])
        # add between workflow connections
        for u, v, d in self._graph.edges_iter(data=True):
            uname = '.'.join(hierarchy + [u.fullname])
            vname = '.'.join(hierarchy + [v.fullname])
            for src, dest, _ in d['connect']:
                uname1 = uname
                vname1 = vname
                if isinstance(src, tuple):
                    srcname = src[0]
                else:
                    srcname = src
                if '.' in srcname:
                    uname1 += '.' + '.'.join(srcname.split('.')[:-1])
                if '.' in dest and '@' not in dest:
                    if not isinstance(v, Workflow):
                        if 'datasink' not in \
                           str(v._interface.__class__).lower():
                            vname1 += '.' + '.'.join(dest.split('.')[:-1])
                    else:
                        vname1 += '.' + '.'.join(dest.split('.')[:-1])
                if uname1.split('.')[:-1] != vname1.split('.')[:-1]:
                    dotlist.append('%s -> %s;' % (uname1.replace('.', '_'),
                                                  vname1.replace('.', '_')))
                    logger.debug('cross connection: ' + dotlist[-1])
        return ('\n' + prefix).join(dotlist)


class CachedWorkflow(Workflow):
    """
    Implements a kind of workflow that can be by-passed if all the fields
    of an input `cachenode` are set.
    """

    def __init__(self, name, base_dir=None, cache_map=[]):
        """Create a workflow object.
        Parameters
        ----------
        name : alphanumeric string
            unique identifier for the workflow
        base_dir : string, optional
            path to workflow storage
        cache_map : list of tuples, non-empty
            each tuple indicates the input port name and the node and output
            port name, for instance ('b', 'outputnode.sum') will map the
            workflow input 'conditions.b' to 'outputnode.sum'.
            'b'
        """

        from nipype.interfaces.utility import CheckInterface, Merge, Select
        super(CachedWorkflow, self).__init__(name, base_dir)

        if cache_map is None or not cache_map:
            raise ValueError('CachedWorkflow cache_map must be a '
                             'non-empty list of tuples')

        if isinstance(cache_map, tuple):
            cache_map = [cache_map]

        cond_in, cond_out = zip(*cache_map)
        self._cache = Node(IdentityInterface(
            fields=list(cond_in)), name='cachenode')
        self._check = Node(CheckInterface(
            fields=list(cond_in)), 'decidenode', control=False)
        self._outputnode = Node(IdentityInterface(
            fields=cond_out), name='outputnode')

        def _switch_idx(val):
            return [int(val)]

        def _fix_undefined(val):
            from nipype.interfaces.base import isdefined
            if isdefined(val):
                return val
            return None

        self._plain_connect(self._check, 'out', self._signalnode, 'disable')
        self._switches = {}
        for ci, co in cache_map:
            m = Node(Merge(2), 'Merge_%s' % co, control=False)
            s = Node(Select(), 'Switch_%s' % co, control=False)
            self._plain_connect([
                (m, s, [('out', 'inlist')]),
                (self._cache, self._check, [(ci, ci)]),
                (self._cache, m, [((ci, _fix_undefined), 'in2')]),
                (self._signalnode, s, [(('disable', _switch_idx), 'index')]),
                (s, self._outputnode, [('out', co)])
            ])
            self._switches[co] = m

    def _plain_connect(self, *args, **kwargs):
        super(CachedWorkflow, self).connect(*args, **kwargs)

    def connect(self, *args, **kwargs):
        """Connect nodes in the pipeline.
        """

        if len(args) == 1:
            flat_conns = args[0]
        elif len(args) == 4:
            flat_conns = [(args[0], args[2], [(args[1], args[3])])]
        else:
            raise Exception('unknown set of parameters to connect function')
        if not kwargs:
            disconnect = False
        else:
            disconnect = kwargs.get('disconnect', False)

        list_conns = []
        for srcnode, dstnode, conns in flat_conns:
            is_output = (isinstance(dstnode, string_types) and
                         dstnode == 'output')
            if not is_output:
                list_conns.append((srcnode, dstnode, conns))
            else:
                for srcport, dstport in conns:
                    mrgnode = self._switches.get(dstport, None)
                    if mrgnode is None:
                        raise RuntimeError('Destination port not found')
                    logger.debug('Mapping %s to %s' % (srcport, dstport))
                    list_conns.append((srcnode, mrgnode, [(srcport, 'in1')]))

        super(CachedWorkflow, self).connect(list_conns, disconnect=disconnect)