# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nipype interfaces core
......................


Defines the ``Interface`` API and the body of the
most basic interfaces.
The I/O specifications corresponding to these base
interfaces are found in the ``specs`` module.

"""
from __future__ import (print_function, division, unicode_literals,
                        absolute_import)

from copy import deepcopy
from datetime import datetime as dt
import os
import platform
import shlex
import sys
from builtins import object, open, str
from dateutil.parser import parse as parseutc
from future import standard_library

from ... import config, logging
from ...utils.errors import NipypeRuntimeError
from ...utils.provenance import write_provenance
from ...utils.subprocess import run_command
from ...utils.misc import str2bool, rgetcwd
from ...utils.filemanip import which  #, get_dependencies

from .traits_extension import isdefined
from .specs import (BaseInterfaceInputSpec, CommandLineInputSpec,
                    StdOutCommandLineInputSpec, MpiCommandLineInputSpec,
                    check_inputs)
from .support import (Bunch, InterfaceResult, format_help)

standard_library.install_aliases()

iflogger = logging.getLogger('nipype.interface')

PY35 = sys.version_info >= (3, 5)
PY3 = sys.version_info[0] > 2
VALID_TERMINAL_OUTPUT = [
    'stream', 'allatonce', 'file', 'file_split', 'file_stdout', 'file_stderr',
    'none'
]
__docformat__ = 'restructuredtext'


class Interface(object):
    """
    This is an abstract definition for Interface objects.

    It provides no functionality.  It defines the necessary attributes
    and methods all Interface objects should have.
    """
    __slots__ = ['inputs', '_outputs', '_references', '_always_run', '_version',
                 '_additional_metadata', '_redirect_x']

    input_spec = None  # A traited input specification
    output_spec = None  # A traited output specification

    def __init__(self, **inputs):
        """ Initialize command with given args and inputs. """
        if not self.input_spec:
            raise NotImplementedError(
                'No input specification found for %s.' % self.__class__.__name__)

        # Set inputs
        self.inputs = self.input_spec(**inputs)  # noqa

        self._outputs = None
        self._references = []  # hook for duecredit refs
        self._always_run = False
        self._version = None
        self._additional_metadata = []
        self._redirect_x = False

    def _set_output(self, out_name, value):
        if not self._outputs:
            raise RuntimeError('No outputs')
        setattr(self._outputs, out_name, value)

    @property
    def always_run(self):
        """ should the interface be always run even if the
            inputs were not changed?
        """
        return self._always_run

    @property
    def version(self):
        """ Interface version """
        if self._version is None and str2bool(config.get(
            'execution', 'stop_on_unknown_version')):
            raise ValueError('Interface %s has no version information' %
                             self.__class__.__name__)
        return self._version

    @classmethod
    def help(cls, returnhelp=False):
        """ Prints class help """
        allhelp = format_help(cls)
        if returnhelp:
            return allhelp
        print(allhelp)
        return None  # R1710

    def run(self):
        """Execute the command."""
        raise NotImplementedError


class BaseInterface(Interface):
    """Implements common interface functionality.

    Implements
    ----------

    * Initializes inputs/outputs from input_spec/output_spec
    * Provides help based on input_spec and output_spec
    * Checks for mandatory inputs before running an interface
    * Runs an interface and returns results
    * Determines which inputs should be copied or linked to cwd


    Relevant Interface attributes
    -----------------------------

    ``input_spec`` points to the traited class for the inputs
    ``output_spec`` points to the traited class for the outputs
    ``_redirect_x`` should be set to ``True`` when the interface requires
      connecting to a ``$DISPLAY`` (default is ``False``).
    ``resource_monitor`` if ``False`` prevents resource-monitoring this
      interface, if ``True`` monitoring will be enabled IFF the general
      Nipype config is set on (``resource_monitor = true``).

    An interface pattern that allows outputs to be set in a dictionary
    called ``_results`` that is automatically interpreted by
    ``_list_outputs()`` to find the outputs.

    When implementing ``_run_interface``, set outputs with::

        self._results[out_name] = out_value

    This can be a way to upgrade a ``Function`` interface to do type checking.

    Examples
    --------

    >>> from nipype.interfaces.base import (
    ...     BaseInterface, BaseInterfaceInputSpec, TraitedSpec)

    >>> def double(x):
    ...    return 2 * x
    ...
    >>> class DoubleInputSpec(BaseInterfaceInputSpec):
    ...     x = traits.Float(mandatory=True)
    ...
    >>> class DoubleOutputSpec(TraitedSpec):
    ...     doubled = traits.Float()
    ...
    >>> class Double(BaseInterface):
    ...     input_spec = DoubleInputSpec
    ...     output_spec = DoubleOutputSpec
    ...
    ...     def _run_interface(self, runtime):
    ...          self._set_output('doubled', double(self.inputs.x))
    ...          return runtime

    >>> dbl = Double()
    >>> dbl.inputs.x = 2
    >>> dbl.run().outputs.doubled
    4.0

    """
    __slots__ = ['_resource_monitor']

    input_spec = BaseInterfaceInputSpec

    def __init__(self, from_file=None, resource_monitor=None, **inputs):
        """ Instantiate a minimal interface """
        super(BaseInterface, self).__init__(**inputs)
        self._resource_monitor = resource_monitor if resource_monitor is not None else True

        # if from_file is not None:
        #     self.load_inputs_from_json(from_file, overwrite=True)

        #     for name, value in list(inputs.items()):
        #         setattr(self.inputs, name, value)

    @property
    def resource_monitor(self):
        """ Enable resource monitoring for this interface """
        return self._resource_monitor

    @resource_monitor.setter
    def resource_monitor(self, value):
        if value is not None:
            self._resource_monitor = value is True

    def _run_interface(self, runtime, correct_return_codes=(0, )):
        """ Core function that executes interface
        """
        raise NotImplementedError

    def run(self, cwd=None, **inputs):
        """Execute this interface.

        This interface will not raise an exception if runtime.returncode is
        non-zero.

        Parameters
        ----------

        cwd : specify a folder where the interface should be run
        inputs : allows the interface settings to be updated

        Returns
        -------
        results :  an InterfaceResult object containing a copy of the instance
        that was executed, provenance information and, if successful, results
        """
        from ...utils.profiler import ResourceMonitor

        # Tear-up: get current and prev directories first
        syscwd = rgetcwd(error=False)  # Recover when wd does not exist
        if cwd is None:
            cwd = syscwd

        self.inputs.trait_set(**inputs)  # Set inputs given at run time
        check_inputs(self.inputs)  # Validate inputs

        if self.output_spec:
            self._outputs = self.output_spec()

        # initialize provenance tracking
        store_provenance = str2bool(
            config.get('execution', 'write_provenance', 'false'))
        env = deepcopy(dict(os.environ))
        if self._redirect_x:
            env['DISPLAY'] = config.get_display()

        runtime = Bunch(
            cwd=cwd,
            prevcwd=syscwd,
            returncode=None,
            duration=None,
            environ=env,
            startTime=dt.isoformat(dt.utcnow()),
            endTime=None,
            platform=platform.platform(),
            hostname=platform.node(),
            version=self.version)
        runtime_attrs = set(runtime.dictcopy())

        mon_sp = None
        if  self.resource_monitor and config.resource_monitor:
            mon_freq = float(
                config.get('execution', 'resource_monitor_frequency', 1))
            proc_pid = os.getpid()
            iflogger.debug(
                'Creating a ResourceMonitor on a %s interface, PID=%d.',
                self.__class__.__name__, proc_pid)
            mon_sp = ResourceMonitor(proc_pid, freq=mon_freq, cwd=cwd)
            mon_sp.start()

        try:
            os.chdir(cwd)  # Change to the interface wd
            runtime = self._pre_run_hook(runtime)
            runtime = self._run_interface(runtime)
            runtime = self._post_run_hook(runtime)
        except Exception as e:
            import traceback
            # Retrieve the maximum info fast
            runtime.traceback = traceback.format_exc()
            # Gather up the exception arguments and append nipype info.
            exc_args = getattr(e, 'args', tuple())
            exc_args += (
                'An exception of type %s occurred while running interface %s.'
                % (type(e).__name__, self.__class__.__name__), )
            if config.get('logging', 'interface_level',
                          'info').lower() == 'debug':
                exc_args += ('Inputs: %s' % str(self.inputs.get_traitsfree()), )

            runtime.traceback_args = ('\n'.join(
                ['%s' % arg for arg in exc_args]), )
        finally:
            os.chdir(syscwd)  # Get back ASAP

            # Make sure runtime profiler is shut down
            if mon_sp is not None:
                import numpy as np
                mon_sp.stop()

                runtime.mem_peak_gb = None
                runtime.cpu_percent = None

                # Read .prof file in and set runtime values
                vals = np.loadtxt(mon_sp.fname, delimiter=',')
                if vals.size:
                    vals = np.atleast_2d(vals)
                    runtime.mem_peak_gb = vals[:, 1].max() / 1024
                    runtime.cpu_percent = vals[:, 2].max()

                    runtime.prof_dict = {
                        'time': vals[:, 0].tolist(),
                        'cpus': vals[:, 1].tolist(),
                        'rss_GiB': (vals[:, 2] / 1024).tolist(),
                        'vms_GiB': (vals[:, 3] / 1024).tolist(),
                    }

            if runtime is None or runtime_attrs - set(runtime.dictcopy()):
                raise RuntimeError("{} interface failed to return valid "
                                   "runtime object".format(
                                       self.__class__.__name__))
            # This needs to be done always
            runtime.endTime = dt.isoformat(dt.utcnow())
            timediff = parseutc(runtime.endTime) - parseutc(runtime.startTime)
            runtime.duration = (timediff.days * 86400 + timediff.seconds +
                                timediff.microseconds / 1e6)
            results = InterfaceResult(
                self.__class__,
                runtime,
                inputs=self.inputs.get_traitsfree(),
                outputs=self._outputs.get_traitsfree(),
                provenance=None)

            # Add provenance (if required)
            if store_provenance:
                # Provenance will only throw a warning if something went wrong
                results.provenance = write_provenance(results)

        return results

    def _pre_run_hook(self, runtime):
        """
        Perform any pre-_run_interface() processing

        Subclasses may override this function to modify ``runtime`` object or
        interface state

        MUST return runtime object
        """
        return runtime

    def _post_run_hook(self, runtime):
        """
        Perform any post-_run_interface() processing

        Subclasses may override this function to modify ``runtime`` object or
        interface state

        MUST return runtime object
        """
        return runtime


class SimpleInterface(BaseInterface):
    """
    """

    def __init__(self, from_file=None, resource_monitor=None, **inputs):
        super(SimpleInterface, self).__init__(
            from_file=from_file, resource_monitor=resource_monitor, **inputs)
        self._results = {}

    def _list_outputs(self):
        return self._results


class CommandLine(BaseInterface):
    """Implements functionality to interact with command line programs
    class must be instantiated with a command argument

    Parameters
    ----------

    command : string
        define base immutable `command` you wish to run

    args : string, optional
        optional arguments passed to base `command`


    Examples
    --------
    >>> import pprint
    >>> from nipype.interfaces.base import CommandLine
    >>> cli = CommandLine(command='ls', environ={'DISPLAY': ':1'})
    >>> cli.inputs.args = '-al'
    >>> cli.cmdline
    'ls -al'

    # Use get_traitsfree() to check all inputs set
    >>> pprint.pprint(cli.inputs.get_traitsfree())  # doctest:
    {'args': '-al',
     'environ': {'DISPLAY': ':1'}}

    >>> cli.inputs.get_hashval()[0][0]
    ('args', '-al')
    >>> cli.inputs.get_hashval()[1]
    '11c37f97649cd61627f4afe5136af8c0'

    """

    __slots__ = ['_cmd', '_cmd_prefix', '_terminal_output', '_environ', '_ldd']
    input_spec = CommandLineInputSpec
    _cmd = None
    _terminal_output = 'stream'

    @classmethod
    def set_default_terminal_output(cls, output_type):
        """Set the default terminal output for CommandLine Interfaces.

        This method is used to set default terminal output for
        CommandLine Interfaces.  However, setting this will not
        update the output type for any existing instances.  For these,
        assign the <instance>.terminal_output.
        """

        if output_type in VALID_TERMINAL_OUTPUT:
            cls._terminal_output = output_type
        else:
            raise AttributeError(
                'Invalid terminal output_type: %s' % output_type)

    @classmethod
    def help(cls, returnhelp=False):
        if cls._cmd is None:
            raise NotImplementedError(
                'CommandLineInterface should wrap an executable, but '
                'none has been set.')
        allhelp = 'Wraps command ``{cmd}``.\n\n{help}'.format(
            cmd=cls._cmd, help=super(CommandLine, cls).help(returnhelp=True))
        if returnhelp:
            return allhelp
        print(allhelp)
        return None  # R1710

    def __init__(self, command=None, terminal_output=None, **inputs):
        super(CommandLine, self).__init__(**inputs)
        # Set command. Input argument takes precedence
        self._cmd = command or getattr(self, '_cmd', None)
        if self._cmd is None:
            raise NotImplementedError("Missing command")

        self._cmd_prefix = ''
        self._environ = None
        self.terminal_output = terminal_output or getattr(
            self, '_terminal_output', 'stream')

        # Store dependencies in runtime object
        self._ldd = str2bool(
            config.get('execution', 'get_linked_libs', 'true'))

    @property
    def cmd(self):
        """Base command, immutable"""
        return self._cmd

    @property
    def cmdline(self):
        """ `command` plus any arguments (args)
        validates arguments and generates command line"""
        check_inputs(self.inputs, raise_exception=False)
        allargs = [self._cmd_prefix + self.cmd] + self._parse_inputs()
        return ' '.join(allargs)

    @property
    def terminal_output(self):
        """ Indicates how the terminal output will be handled """
        return self._terminal_output

    @terminal_output.setter
    def terminal_output(self, value):
        if value not in VALID_TERMINAL_OUTPUT:
            raise RuntimeError(
                'Setting invalid value "%s" for terminal_output. Valid values are '
                '%s.' % (value,
                         ', '.join(['"%s"' % v
                                    for v in VALID_TERMINAL_OUTPUT])))
        self._terminal_output = value

    def _get_environ(self):
        return getattr(self.inputs, 'environ', {})

    def _run_interface(self, runtime, correct_return_codes=(0, )):
        """Execute command via subprocess

        Parameters
        ----------
        runtime : passed by the run function

        Returns
        -------
        runtime : updated runtime information
            adds stdout, stderr, merged, cmdline, dependencies, command_path

        """

        out_environ = self._get_environ()
        # Initialize runtime Bunch
        runtime.stdout = None
        runtime.stderr = None
        runtime.cmdline = self.cmdline
        runtime.environ.update(out_environ)

        # which $cmd
        executable_name = shlex.split(self._cmd_prefix + self.cmd)[0]
        cmd_path = which(executable_name, env=runtime.environ)

        if cmd_path is None:
            raise IOError(
                'No command "%s" found on host %s. Please check that the '
                'corresponding package is installed.' % (executable_name,
                                                         runtime.hostname))

        runtime.command_path = cmd_path

        # TODO memoize
        # runtime.dependencies = (get_dependencies(executable_name,
        #                                          runtime.environ)
        #                         if self._ldd else '<skipped>')
        runtime = run_command(runtime, output=self.terminal_output)
        if runtime.returncode is None or \
                runtime.returncode not in correct_return_codes:
            raise NipypeRuntimeError(runtime)

        return runtime

    def _format_arg(self, name, trait_spec, value):
        """A helper function for _parse_inputs

        Formats a trait containing argstr metadata
        """
        return format_arg(name, trait_spec, value)

    def _list_outputs(self):
        metadata = dict(name_source=lambda t: t is not None)
        traits = self.inputs.traits(**metadata)
        if traits:
            outputs = self.output_spec().trait_get()
            for name, trait_spec in list(traits.items()):
                out_name = name
                if trait_spec.output_name is not None:
                    out_name = trait_spec.output_name
                fname = self._filename_from_source(name)
                if isdefined(fname):
                    outputs[out_name] = os.path.abspath(fname)
            return outputs

    def _parse_inputs(self, skip=None):
        """Parse all inputs using the ``argstr`` format string in the Trait.

        Any inputs that are assigned (not the default_value) are formatted
        to be added to the command line.

        Returns
        -------
        all_args : list
            A list of all inputs formatted for the command line.

        """
        return parse_inputs(self.inputs, skip=None)


class StdOutCommandLine(CommandLine):
    input_spec = StdOutCommandLineInputSpec

    def _gen_filename(self, name):
        return self._gen_outfilename() if name == 'out_file' else None

    def _gen_outfilename(self):
        raise NotImplementedError


class MpiCommandLine(CommandLine):
    """Implements functionality to interact with command line programs
    that can be run with MPI (i.e. using 'mpiexec').

    Examples
    --------
    >>> from nipype.interfaces.base import MpiCommandLine
    >>> mpi_cli = MpiCommandLine(command='my_mpi_prog')
    >>> mpi_cli.inputs.args = '-v'
    >>> mpi_cli.cmdline
    'my_mpi_prog -v'

    >>> mpi_cli.inputs.use_mpi = True
    >>> mpi_cli.inputs.n_procs = 8
    >>> mpi_cli.cmdline
    'mpiexec -n 8 my_mpi_prog -v'
    """
    input_spec = MpiCommandLineInputSpec

    @property
    def cmdline(self):
        """Adds 'mpiexec' to begining of command"""
        result = []
        if self.inputs.use_mpi:
            result.append('mpiexec')
            if self.inputs.n_procs:
                result.append('-n %d' % self.inputs.n_procs)
        result.append(super(MpiCommandLine, self).cmdline)
        return ' '.join(result)


class SEMLikeCommandLine(CommandLine):
    """In SEM derived interface all outputs have corresponding inputs.
    However, some SEM commands create outputs that are not defined in the XML.
    In those cases one has to create a subclass of the autogenerated one and
    overload the _list_outputs method. _outputs_from_inputs should still be
    used but only for the reduced (by excluding those that do not have
    corresponding inputs list of outputs.
    """

    def _list_outputs(self):
        outputs = self.output_spec().trait_get()
        return self._outputs_from_inputs(outputs)

    def _outputs_from_inputs(self, outputs):
        for name in list(outputs.keys()):
            corresponding_input = getattr(self.inputs, name)
            if isdefined(corresponding_input):
                if (isinstance(corresponding_input, bool)
                        and corresponding_input):
                    outputs[name] = \
                        os.path.abspath(self._outputs_filenames[name])
                else:
                    if isinstance(corresponding_input, list):
                        outputs[name] = [
                            os.path.abspath(inp) for inp in corresponding_input
                        ]
                    else:
                        outputs[name] = os.path.abspath(corresponding_input)
        return outputs

    def _format_arg(self, name, spec, value):
        if name in list(self._outputs_filenames.keys()):
            if isinstance(value, bool):
                if value:
                    value = os.path.abspath(self._outputs_filenames[name])
                else:
                    return ""
        return super(SEMLikeCommandLine, self)._format_arg(name, spec, value)


class LibraryBaseInterface(BaseInterface):
    _pkg = None
    imports = ()

    def __init__(self, check_import=True, *args, **kwargs):
        super(LibraryBaseInterface, self).__init__(*args, **kwargs)
        if check_import:
            import importlib
            failed_imports = []
            for pkg in (self._pkg,) + tuple(self.imports):
                try:
                    importlib.import_module(pkg)
                except ImportError:
                    failed_imports.append(pkg)
            if failed_imports:
                iflogger.warning(
                    'Unable to import %s; %s interface may fail to '
                    'run', failed_imports, self.__class__.__name__)

    @property
    def version(self):
        if self._version is None:
            import importlib
            try:
                self._version = importlib.import_module(self._pkg).__version__
            except (ImportError, AttributeError):
                pass
        return super(LibraryBaseInterface, self).version


class PackageInfo(object):
    _version = None
    version_cmd = None
    version_file = None

    @classmethod
    def version(cls):
        """Memoize version of package"""
        if cls._version is None:
            if cls.version_cmd is not None:
                try:
                    clout = CommandLine(
                        command=cls.version_cmd,
                        resource_monitor=False,
                        terminal_output='allatonce').run()
                except IOError:
                    return None

                raw_info = clout.runtime.stdout
            elif cls.version_file is not None:
                try:
                    with open(cls.version_file, 'rt') as fobj:
                        raw_info = fobj.read()
                except OSError:
                    return None
            else:
                return None

            cls._version = cls.parse_version(raw_info)

        return cls._version

    @staticmethod
    def parse_version(raw_info):
        """Method that should be implemented by the package"""
        raise NotImplementedError
