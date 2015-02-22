# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parallel workflow execution via multiprocessing

Support for child processes running as non-daemons based on
http://stackoverflow.com/a/8963618/1183453
"""

from multiprocessing import Process, Pool, cpu_count, pool, Lock
from traceback import format_exception
import sys
import time
import os.path as op
from .base import (DistributedPluginBase, report_crash)


def run_node(node, updatehash, lock=None):
    result = dict(result=None, traceback=None)

    if lock is not None:
        lock.acquire()

    try:
        result['result'] = node.run(updatehash=updatehash)
    except:
        etype, eval, etr = sys.exc_info()
        result['traceback'] = format_exception(etype, eval, etr)
        result['result'] = node.result

    if lock is not None:
        lock.release()

    return result


class NonDaemonProcess(Process):
    """A non-daemon process to support internal multiprocessing.
    """
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(pool.Pool):
    """A process pool with non-daemon processes.
    """
    Process = NonDaemonProcess


class MultiProcPlugin(DistributedPluginBase):
    """
    Execute workflow with multiprocessing

    The plugin_args input to run can be used to control the multiprocessing
    execution. Currently supported options are:

    - n_procs : number of processes to use
    - non_daemon : boolean flag to execute as non-daemon processes

    """

    def __init__(self, plugin_args=None):
        super(MultiProcPlugin, self).__init__(plugin_args=plugin_args)
        self._taskresult = {}
        self._taskid = 0
        self._lock = Lock()
        self._non_daemon = True
        self._n_procs = cpu_count()

        maxtasks = None
        if plugin_args:
            if 'n_procs' in plugin_args:
                self._n_procs = plugin_args['n_procs']
            if 'non_daemon' in plugin_args:
                self._non_daemon = plugin_args['non_daemon']
            if 'maxtasksperchild' in plugin_args:
                maxtasks = plugin_args['maxtasksperchild']

        # set maxtasksperchild 5 to refresh workers with that frequency
        if self._non_daemon:
            # run the execution using the non-daemon pool subclass
            self.pool = NonDaemonPool(processes=self._n_procs,
                                      maxtasksperchild=maxtasks)
        else:
            self.pool = Pool(processes=self._n_procs,
                             maxtasksperchild=maxtasks)

    def _get_result(self, taskid):
        if taskid not in self._taskresult:
            raise RuntimeError('Multiproc task %d not found' % taskid)
        if not self._taskresult[taskid].ready():
            return None
        return self._taskresult[taskid].get()

    def _submit_job(self, node, updatehash=False):
        self._taskid += 1
        try:
            if node.inputs.terminal_output == 'stream':
                node.inputs.terminal_output = 'allatonce'
        except:
            pass

        allworkers = False
        try:
            allworkers = not node._interface._singleworker
        except:
            pass

        if allworkers:
            self._taskresult[self._taskid] = self.pool.apply(
                run_node, (node, updatehash, self._lock))
        else:
            self._taskresult[self._taskid] = self.pool.apply_async(
                run_node, (node, updatehash, None))
        return self._taskid

    def _report_crash(self, node, result=None):
        if result and result['traceback']:
            node._result = result['result']
            node._traceback = result['traceback']
            return report_crash(node,
                                traceback=result['traceback'])
        else:
            return report_crash(node)

    def _clear_task(self, taskid):
        del self._taskresult[taskid]
