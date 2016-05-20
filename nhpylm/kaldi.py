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

import os
kaldi_root = os.getenv('KALDI_ROOT', '/net/ssd/software/kaldi')


def get_kaldi_env():
    env = os.environ.copy()
    env['PATH'] += ':{}/src/bin'.format(kaldi_root)
    env['PATH'] += ':{}/tools/openfst/bin'.format(kaldi_root)
    env['PATH'] += ':{}/src/fstbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/gmmbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/featbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/lm/'.format(kaldi_root)
    env['PATH'] += ':{}/src/sgmmbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/sgmm2bin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/fgmmbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/latbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/nnetbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/nnet2bin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/kwsbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/online2bin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/ivectorbin/'.format(kaldi_root)
    env['PATH'] += ':{}/src/lmbin/'.format(kaldi_root)
    if 'LD_LIBRARY_PATH' in env.keys():
        env['LD_LIBRARY_PATH'] += ':{}/tools/openfst/lib'.format(kaldi_root)
    else:
        env['LD_LIBRARY_PATH'] = ':{}/tools/openfst/lib'.format(kaldi_root)
    env['LC_ALL'] = 'C'
    env['OMP_NUM_THREADS'] = '1'
    return env