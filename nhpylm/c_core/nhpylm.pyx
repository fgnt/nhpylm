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

from libcpp.string cimport string
from libcpp.vector cimport vector
from tqdm import tqdm

cdef extern from "math.h":
    float log(float x) nogil

special_symbols = [
    'EPS', 'PHI', 'SOW', 'EOW', 'SOS', 'EOS', 'EOC', 'BLANK'
]

cdef class NHPYLM_wrapper:
    """ Wrapper for a hierarchical Pitman-Yor model.

    :param symbols: List of symbols used to represent words
    :param word_model_order: Order of the word model
    :param character_model_order: Order of the character model

    """
    cdef NHPYLM *_lm
    cdef dict _sym_to_int
    cdef dict _int_to_sym
    cdef int _sentence_boundary_id
    cdef bool add_eos
    def __cinit__(self, symbols, word_model_order=2, character_model_order=8,
                  double word_base_probability=0.):

        symbols = special_symbols + symbols
        cdef int i
        self._sym_to_int = dict()
        self._int_to_sym = dict()
        for i in range(len(symbols)):
            self._sym_to_int[symbols[i]] = i
            self._int_to_sym[i] = symbols[i].encode()

        cdef vector[string] sym_vec
        for sym in symbols:
            sym_vec.push_back(sym.encode())
        self._lm = new NHPYLM(character_model_order, word_model_order,
                              sym_vec, len(special_symbols),
                              word_base_probability)
        self._sentence_boundary_id = self._add_word(['EOS'])

    cdef int _add_word(self, word):
        cdef vector[int] word_vec = [self._sym_to_int[c] for c in word]
        word_id, _ = self._lm.AddCharacterIdSequenceToDictionary(
            word_vec.const_begin(), word_vec.size()
        )
        return word_id

    cpdef set_char_base_probs(self, char_prob_dict):
        for char_id, prob in char_prob_dict.items():
            self._lm.SetCharBaseProb(self._sym_to_int[char_id], prob)

    cpdef int word2id(self, word):
        """ Returns the id for a specific word

        :param word:
        """
        cdef vector[int] word_vec = [self._sym_to_int[c] for c in word]
        return self._lm.GetWordId(word_vec.const_begin(),
                                  word_vec.size())

    cpdef id2word(self, id):
        return self._lm.GetWordVector(id)

    def id2string(self, id):
        return ''.join([self._int_to_sym[c] for c in self.id2word(id)])

    @property
    def sentence_boundary_id(self):
        return self._sentence_boundary_id

    @property
    def word_order(self):
        return self._lm.GetWHPYLMOrder()

    @property
    def character_order(self):
        return self._lm.GetCHPYLMOrder()

    @property
    def character_model_context_count(self):
        return self._lm.GetTotalCountPerLevelFor(b'CHPYLM', b'Context')

    @property
    def character_model_table_count(self):
        return self._lm.GetTotalCountPerLevelFor(b'CHPYLM', b'Table')

    @property
    def character_model_word_count(self):
        return self._lm.GetTotalCountPerLevelFor(b'CHPYLM', b'Word')

    @property
    def word_model_context_count(self):
        return self._lm.GetTotalCountPerLevelFor(b'WHPYLM', b'Context')

    @property
    def word_model_table_count(self):
        return self._lm.GetTotalCountPerLevelFor(b'WHPYLM', b'Table')

    @property
    def word_model_word_count(self):
        return self._lm.GetTotalCountPerLevelFor(b'WHPYLM', b'Word')

    @property
    def hyperparameter(self):
        return self._lm.GetNHPYLMParameters()

    # @hyperparameter.setter
    # TODO: Setter somehow won't work (Cython?!)
    def set_hyperparameter(self, val):
        ref_params = self.hyperparameter
        for param_name in ['CHPYLMDiscount', 'CHPYLMConcentration',
                      'WHPYLMDiscount', 'WHPYLMConcentration']:
            hyperparams = val[param_name]
            assert len(hyperparams) == len(ref_params[param_name])
            split = len('CHPYLM')
            model, param_type = param_name[:split], param_name[split:]
            for lvl, param_val in enumerate(hyperparams):
                self._lm.SetParameter(model.encode(), param_type.encode(),
                                      lvl, param_val)

    @property
    def base_probabilities(self):
        return self._lm.GetWHPYLMBaseProbabilitiesScale()

    def set_base_probabilities(self, base_probabilities):
        cdef vector[double] base_probs = base_probabilities
        self._lm.SetWHPYLMBaseProbabilitiesScale(base_probs)

    @property
    def start_context_id(self):
        cdef vector[int] word_vec = \
            (self.word_order-1)*[self._sentence_boundary_id]
        return self._lm.GetContextId(word_vec)

    @property
    def root_context_id(self):
        return self._lm.GetRootContextId()

    @property
    def final_context_id(self):
        return self._lm.GetFinalContextId()

    @property
    def string_ids(self):
        cdef vector[string] syms = self._lm.GetId2CharacterSequenceVector()
        strings = list()
        for idx, sym in enumerate(syms):
            if idx < len(self._sym_to_int):
                strings.append('_{}'.format(sym.decode()))
            else:
                strings.append(sym.decode())
        return strings

    @property
    def list_ids(self):
        return self._lm.GetId2SeparatedCharacterSequenceVector()

    def get_word_id_to_char_id(self):
        word_id_to_char_id = dict()
        for word_id in range(self._lm.GetWordsBegin() + 1, self._lm.GetMaxNumWords()):
            word_id_to_char_id[word_id] = self.id2word(word_id)[self.character_order-1:-1]
        return word_id_to_char_id

    def get_char_ids(self):
        return list(range(len(special_symbols), len(self._int_to_sym)))

    def sym2id(self, sym):
        return self._sym_to_int[sym]

    def id2sym(self, id):
        return self._int_to_sym[id]

    cpdef word_list_to_id_list(self, word_list):
        """ Converts a list of words to a list of word ids.

        Words have to be represented by a list or tuple of symbols.

        :param word_list: List of words
        :return: list with word ids
        """
        id_list = list()
        id_list.extend((self.word_order-1)*[self._sentence_boundary_id])
        cdef vector[int] char_vec
        for word in word_list:
            id_list.append(self._add_word(word))
        id_list.extend([self._sentence_boundary_id])
        return id_list

    cpdef word_lists_to_id_lists(self, word_lists):
        """ Converts a list of lists of words to a list of lists of word ids.

        Words have to be represented by a list or tuple of symbols.

        :param word_list: List of lists of words
        :return: list with lists of word ids
        """
        id_lists = list()
        for word_list in word_lists:
            id_lists.append(self.word_list_to_id_list(word_list))
        return id_lists

    cpdef add_id_sentence_to_lm(self, vector[int] sentence):
        """ Adds a sentence of word ids to the language model.

        The sentence has to be represented by a number of integer word ids.

        :param sentence: Sentence to add
        """
        # cdef vector[int] word_vec = sentence
        self._lm.AddWordSequenceToLm(sentence)

    cpdef rm_id_sentence_from_lm(self, sentence):
        """ Removes a sentence of word ids from the language model.

        The sentence has to be represented by a number of integer word ids.

        :param sentence: Sentence to remove
        """
        cdef vector[int] word_vec = sentence
        self._lm.RemoveWordSequenceFromLm(word_vec)

    cpdef add_id_sentence_list_to_lm(self, sentences):
        """ Adds several sentences of word ids to the language model

        :param sentences: List of word id sentences
        """
        for sentence in sentences:
            self.add_id_sentence_to_lm(sentence)

    cpdef train_with_list_of_sentences(self, sentences, iterations=3):
        """ Train the language model with a list of word sentences

        :param sentences:
        """
        id_sentence_list = self.word_lists_to_id_lists(sentences)
        self.add_id_sentence_list_to_lm(id_sentence_list)
        cdef int it
        for it in range(iterations):
            for sentence in id_sentence_list:
                self.rm_id_sentence_from_lm(sentence)
                self.add_id_sentence_to_lm(sentence)
            self.resample_hyperparameters()

    cpdef resample_hyperparameters(self):
        """ Resamples the hyperparameters of the language model

        """
        self._lm.ResampleHyperParameters()

    cpdef word_sequence_likelihood(self, word_sequence, with_eos=False):
        """ Calculates the likelihood of a given word sequence

        :param word_sequence: A sequence of words (not ids!)
        """

        id_sequence = self.word_list_to_id_list(word_sequence)
        if not with_eos:
            id_sequence = id_sequence[:-1]
        return self._lm.WordSequenceLoglikelihood(id_sequence)

    cpdef get_transitions_for_id(self, id):
        """ Calculates the transitions for a given id

        :param id: Context for the transitions
        """

        cdef vector[bool] empty_active_words = vector[bool]()
        return self._lm.GetTransitions(id, self._sentence_boundary_id,
                                       empty_active_words)

    cpdef to_fst_text_format(self, sow=None, eow=None, eos_word=None):
        cdef vector[string] fst_lines = vector[string]()
        cdef vector[bool] visited_contexts = vector[bool](self.final_context_id, 0)
        next_context = list()
        next_context.append(self.start_context_id)
        arc_list = list()
        cdef int cur_context
        cdef int dest
        cdef int label
        cdef float weight
        cdef int i
        cdef ContextToContextTransitions transitions
        progress = tqdm(desc='Visiting contexts',
                        total=self.final_context_id)
        while len(next_context) > 0:
            cur_context = next_context.pop()
            if not visited_contexts[cur_context]:
                visited_contexts[cur_context] = True
                transitions = self.get_transitions_for_id(cur_context)
                for i in range(transitions.Words.size()):
                    dest = transitions.NextContextIds[i]
                    label = transitions.Words[i]
                    weight = -log(transitions.Probabilities[i])
                    if sow is not None and cur_context == self.root_context_id and label == 1:
                        label = sow  # Use specified sow to enter char model
                    if eow is not None and label == 3:
                        label = eow  # Use specified eow to leave char model
                    if eos_word is not None and label == self._sentence_boundary_id:
                        label = eos_word  # Use specified eos_word to finish sequence
                    fst_lines.push_back(
                            '{} {} {} {} {}'.format(
                                    cur_context,
                                    dest,
                                    label, label,
                                    weight).encode())
                    arc_list.append((cur_context, dest, label, label, weight))
                    if not visited_contexts[dest]:
                        next_context.append(dest)
                progress.update()
        fst_lines.push_back('{}'.format(self.final_context_id).encode())
        arc_list.append((self.final_context_id,))
        progress.close()
        return fst_lines, arc_list
