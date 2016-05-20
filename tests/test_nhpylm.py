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

import unittest
from nhpylm.c_core.nhpylm import NHPYLM_wrapper as NHPYLM

symbols = ['A', 'B']

class TestNHPYLM(unittest.TestCase):

    def setUp(self):
        self.lm = NHPYLM(symbols, 2, 1)

    def test_word_order_property(self):
        self.assertEqual(self.lm.word_order, 2)

    def test_character_order_property(self):
        self.assertEqual(self.lm.character_order, 1)

    def test_get_word_id(self):
        word = ['EOS']
        word_id = self.lm.word2id(word)
        self.assertEqual(word_id,
                         self.lm.sentence_boundary_id)

    def test_word_list_to_id_list(self):
        word_list = [['A', 'A'], ['B', 'A']]
        id_list = self.lm.word_list_to_id_list(word_list)
        self.assertEqual(len(id_list), 4)
        self.assertEqual(id_list[0], self.lm.sentence_boundary_id)
        self.assertEqual(id_list[-1], self.lm.sentence_boundary_id)
        self.assertEqual(id_list[1], id_list[2]-1)

    def test_add_word_id_list(self):
        word_list = [['A', 'A'], ['B', 'A']]
        id_list = self.lm.word_list_to_id_list(word_list)
        self.lm.add_id_sentence_to_lm(id_list)
        self.assertEqual(self.lm.word_model_word_count[0], 3)

    def test_rm_word_id_list(self):
        word_list = [['A', 'A'], ['B', 'A']]
        id_list = self.lm.word_list_to_id_list(word_list)
        self.lm.add_id_sentence_to_lm(id_list)
        self.assertEqual(self.lm.word_model_word_count[0], 3)
        self.lm.rm_id_sentence_from_lm(id_list)
        self.assertEqual(self.lm.word_model_word_count[0], 0)

    def test_get_hyperparameter(self):
        params = self.lm.hyperparameter
        self.assertEqual(params['CHPYLMConcentration'], [0.1])
        self.assertEqual(params['CHPYLMDiscount'], [0.5])
        self.assertEqual(params['WHPYLMConcentration'], [0.1, 0.1])
        self.assertEqual(params['WHPYLMDiscount'], [0.5, 0.5])

    def test_set_hyperparameter(self):
        params = self.lm.hyperparameter
        new_params = dict()
        for name, param_list in params.items():
            new_params[name] = list()
            for i, p in enumerate(param_list):
                new_params[name].append(p + i + 1)
        print(new_params)
        self.lm.set_hyperparameter(new_params)
        params = self.lm.hyperparameter
        self.assertEqual(params['CHPYLMConcentration'], [1.1])
        self.assertEqual(params['CHPYLMDiscount'], [1.5])
        self.assertEqual(params['WHPYLMConcentration'], [1.1, 2.1])
        self.assertEqual(params['WHPYLMDiscount'], [1.5, 2.5])

    def test_resample_hyperparameters(self):
        params_before = self.lm.hyperparameter
        word_list = [['A', 'A'], ['B', 'A']]
        id_list = self.lm.word_list_to_id_list(word_list)
        self.lm.add_id_sentence_to_lm(id_list)
        self.lm.resample_hyperparameters()
        params_after = self.lm.hyperparameter
        for p in ['CHPYLMConcentration', 'CHPYLMDiscount',
                  'WHPYLMConcentration', 'WHPYLMDiscount']:
            for val_before, val_after in zip(params_before[p], params_after[p]):
                self.assertNotEqual(val_before, val_after)

    def test_get_transitions(self):
        word_list = [['A', 'A'], ['B', 'A']]
        id_list = self.lm.word_list_to_id_list(word_list)
        self.lm.add_id_sentence_to_lm(id_list)
        transitions = self.lm.get_transitions_for_id(
                self.lm.start_context_id)
        print(transitions)

    def test_set_base_probabilities(self):
        base_probs = len(symbols) * [1/len(symbols)]
        self.lm.set_base_probabilities(base_probs)
        probs = self.lm.base_probabilities
        for p in probs:
            self.assertEqual(p, 1/len(symbols))

    def test_char_base_probs(self):
        self.lm.set_char_base_probs({'A': 0.3,
                                     'B': 0.3,
                                     'EOW': 0.2,
                                     'EOS': 0.1})

    def test_get_ll_no_eos(self):
        word_list = [['A', 'A'], ['B', 'A']]
        id_list = self.lm.word_list_to_id_list(word_list)
        self.lm.add_id_sentence_to_lm(id_list)
        ll = self.lm.word_sequence_likelihood(word_list)
        print(ll/2)
        self.assertGreater(ll, -2)
        self.assertGreater(-1, ll)

    def test_get_ll_with_eos(self):
        word_list = [['A', 'A'], ['B', 'A']]
        id_list = self.lm.word_list_to_id_list(word_list)
        self.lm.add_id_sentence_to_lm(id_list)
        ll = self.lm.word_sequence_likelihood(word_list, True)
        print(ll/3)
        self.assertGreater(ll, -2)
        self.assertGreater(-1, ll)