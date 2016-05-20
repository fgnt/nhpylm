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
##   Author: Oliver Walter
##
## ----------------------------------------------------------------------------

__author__ = 'walter'

from nhpylm.process_caller import run_processes
from nhpylm.fst import build_fst_for_sequence, fstcompile_cmd, fstaddselfloops_cmd, fstcompose_cmd
from nhpylm.fst import fstshortestpath_cmd, fstprint_cmd
from nhpylm.kaldi import get_kaldi_env

KALDI_ENV = get_kaldi_env()

def decode_sequence(sequence, eow, eoc, phi, L_G=None, L=None, G=None, phicompose=False):
    """
    A simple sequence decoder. Decode an input sequence given a lexicon fst and a language model fst
    into a sequence of sequence of integers

    :param sequence: sequence of input labels to decode
    :param eow: end of word symbol (has to match with lexicon)
    :param eoc: end of character sequence symbol (has fo match with lexicon and language model)
    :param phi: phi symbol for fallback transitions (has to match lexicon and language model)
    :param L_G: path to binaty fst with composition of L and G
    :param L: path to binary fst for lexicon
    :param G: path to binaty fst for language model
    :param phicompose: true: do normal composition of I with L and phi composition of I_L with G,
                       false: do normal composition of I with L_G
    :return: integer sequence of decoding result
    """

    cmd = fstcompile_cmd(arcsort=False)
    if not phicompose:
        cmd += fstaddselfloops_cmd(disambig_in=[eow, eoc, phi], disambig_out=[eow, eoc, phi], sort_type='olabel')
        cmd += fstcompose_cmd(right_fst=L_G)
    else:
        cmd += fstaddselfloops_cmd(disambig_in=[eow, eoc], disambig_out=[eow, eoc], sort_type='olabel')
        cmd += fstcompose_cmd(right_fst=L)
        cmd += fstcompose_cmd(right_fst=G, phi=phi)

    cmd += fstshortestpath_cmd(project=True, project_output=True, rmepsilon=True, topsort=True)
    cmd += fstprint_cmd(pipe=False)

    res = list()
    for line in run_processes(cmd, inputs=build_fst_for_sequence(sequence).get_txt(), environment=KALDI_ENV)[0][0].split('\n'):
        split_line = line.split('\t')
        if len(split_line) > 3:
            res.append(split_line[3])

    return res
