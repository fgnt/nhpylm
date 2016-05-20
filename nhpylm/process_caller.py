## ----------------------------------------------------------------------------
##
##   File: nhpylm.pyx
##   Copyright (c) <2013> <University of Paderborn>
##   Permission is hereby granted, free of charge, to any person
##   obtaining a copy of this software and associated documentation
##   files (the "Software"), to deal in the Software without restriction,
##   including without limitation the rights to use, copy, modify and
##   merge the Software, subject to the following conditions:
##
##   1.) The Software is used for non-commercial research and
##       education purposes.
##
##   2.) The above copyright notice and this permission notice shall be
##       included in all copies or substantial portions of the Software.
##
##   3.) Publication, Distribution, Sublicensing, and/or Selling of
##       copies or parts of the Software requires special agreements
##       with the University of Paderborn and is in general not permitted.
##
##   4.) Modifications or contributions to the software must be
##       published under this license. The University of Paderborn
##       is granted the non-exclusive right to publish modifications
##       or contributions in future versions of the Software free of charge.
##
##   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
##   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
##   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
##   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
##   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
##   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
##   OTHER DEALINGS IN THE SOFTWARE.
##
##   Persons using the Software are encouraged to notify the
##   Department of Communications Engineering at the University of Paderborn
##   about bugs. Please reference the Software in your publications
##   if it was used for them.
##
##
##   Author: Jahn Heymann
##
## ----------------------------------------------------------------------------

import subprocess
from warnings import warn
import os

DEBUG_MODE = False
DEFAULT_ENV = os.environ.copy()

def run_processes(cmds, sleep_time=0.1, ignore_return_code=False,
                  environment=DEFAULT_ENV, warn_on_ignore=True,
                  inputs=None):
    """ Starts multiple processes, waits and returns the outputs when available

    :param cmd: A list with the commands to call
    :param sleep_time: Intervall to poll the running processes (in seconds)
    :param ignore_return_code: If true, ignores non zero return codes.
        Otherwise an exception is thrown.
    :param environment: environment (e.g. path variable) for commands
    :param warn_on_ignore: warn if return code is ignored but non zero
    :param inputs: A list with the text inputs to be piped to the called commands
    :return: Stdout, Stderr and return code for each process
    """

    # Ensure its a list if a single command is passed
    cmds = cmds if isinstance(cmds, list) else [cmds]
    if inputs is None:
        inputs = len(cmds) * [None]
    else:
        inputs = inputs if isinstance(inputs, list) else [inputs]

    if DEBUG_MODE:
        [print('Calling: {}'.format(cmd)) for cmd in cmds]
    pipes = [subprocess.Popen(cmd,
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              shell=True,
                              universal_newlines=True,
                              env=environment) for cmd in cmds]
    return_codes = len(cmds) * [None]
    stdout = len(cmds) * [None]
    stderr = len(cmds) * [None]

    # Recover output as the processes finish
    for i, p in enumerate(pipes):
        stdout[i], stderr[i] = p.communicate(inputs[i])
        return_codes[i] = p.returncode

    raise_error_txt = ''
    for idx, code in enumerate(return_codes):
        txt = 'Command {} returned with return code {}.\n' \
                'Stdout: {}\n' \
                'Stderr: {}'.format(cmds[idx], code, stdout[idx], stderr[idx])
        if code != 0 and not ignore_return_code:
            raise_error_txt += txt + '\n'
        if code != 0 and ignore_return_code and warn_on_ignore:
            warn('Returncode for command {} was {} but is ignored.\n'
                 'Stderr: {}'.format(
                cmds[idx], code, stderr[idx]))
        if DEBUG_MODE:
            print(txt)
    if raise_error_txt != '':
        raise EnvironmentError(raise_error_txt)

    return stdout, stderr, return_codes
