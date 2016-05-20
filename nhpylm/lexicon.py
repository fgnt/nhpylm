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
from nhpylm import fst
from tqdm import tqdm


class Lexicon:
    """
    Class containing basic methods to build a fst based lexicon.
    An explicit implementation could be derived from this class

    Note: symbols are usually an int. They could also
          be a char or any other hashable object
    """

    def __init__(self, eps, eow, eoc=None, sil=None):
        """
        Construct lexicon base class. Should be called
        with super().__init__(eps, eow, eoc, sil) from child class

        :param eps: eps symbol
        :param eow: end of word symbol
        :param eoc: end of characters symbol to terminate
                    character sequence for new word
        :param sil: character for space/pause/silence (if available)
        """

        self.lex = fst.SimpleFST()
        self.start_state = self.lex.add_state()
        self.lex.set_start(self.start_state)
        self.lex.set_final(self.start_state)
        self.eps = eps
        self.eow = eow
        self.sil = sil
        if eoc is not None:
            self.eoc = eoc
        else:
            self.eoc = eow

    def add_self_loops(self, label):
        """
        Add self loops before each emitting state. This is usually used to
        realize back-off transitions in the language model.

        :param label: input/output symbol for self loop
        """

        self.lex.add_self_loops(self.eps, label, label)

    def add_eos(self, eos_label, eos_word, eow_label=None):
        """
        Add mapping for end of sentence. This adds transitions
        mapping eos_word to the sequence [ eos_label eow_label ]

        :param eos_label: end of sentence label symbol (character)
        :param eos_word: end of sentence word symbol (word)
        :param eow_label: end of word symbol (character), if None: eps is used
        """

        if eow_label is None:
            eow_label = self.eps

        self._add_word_linear((eos_word, [eos_label]), add_sil=False,
                              eow=eow_label)

    def get_txt(self):
        """
        Get text version of lexicon FST

        :return: text version of lexicon FST
        """

        return self.lex.get_txt()

    def write_txt(self, filename):
        """
        Write lexicon FST in text format

        :param filename: text file to write to
        """

        self.lex.write_txt(filename)

    def write_fst(self, filename, **kwargs):
        """
        Write fst in openfst format to filename.
        This only works if symbols are ints.

        :param filename: fst file name
        :param kwargs: see nt.speech_recognition.fst.SimpleFST.write_fst() for
                       further arguments. This function changes the defaults to
                       determinize=False, minimize=False and sort_type='olabel.
        """

        fst_arguments = {'determinize': False, 'minimize': False,
                         'sort_type': 'olabel'}
        fst_arguments.update(**kwargs)
        self.lex.write_fst(filename, **fst_arguments)

    def _add_characters_loop(self, labels):
        """
        Add a characters self loop passing all characters in labels.
        One additional transition with eow or eoc as input
        and output will be added back to the start state.

        :param labels: list of symbols for character loop
        """

        characters_loop_state = self.lex.add_state()
        self.lex.add_arc(characters_loop_state, self.start_state,
                         self.eoc, self.eoc)
        for label in labels:
            self.lex.add_arc(characters_loop_state, characters_loop_state,
                             label, label)

        return characters_loop_state

    def _add_character_sequence_linear(self, character_sequence):
        """
        Add linear entry for given character sequence not following existing
        transitions. Input and output symbols will be the same.

        :param character_sequence: list of symbols to add
        :return: last state id of added sequence
        """

        current_state = self.start_state
        for char in character_sequence:
            next_state = self.lex.add_state()
            self.lex.add_arc(current_state, next_state, char, char)
            current_state = next_state

        return current_state

    def _add_character_sequence_tri(self, character_sequence):
        """
        Add entry for given character sequence following existing transitions.
        This basically builds a trie. Input and output symbols will be the same.

        :param character_sequence: list of symbols to add
        :return: last state id of added sequence
        """

        current_state = self.start_state
        for character in character_sequence:
            arc = self.lex.find_arc(current_state, character, character)
            if arc:
                next_state = arc.dst
            else:
                next_state = self.lex.add_state()
                self.lex.add_arc(current_state, next_state,
                                 character, character)

            current_state = next_state

        return current_state

    def _add_word_linear(self, word, add_sil=True, eow=None):
        """
        Add linear entry for given word not following existing transitions.
        This puts the word id output on the first transition and adds a final
        transition with eow as input and eps as output back to the start state.

        :param word: tuple (word, character sequence) with word being the
                     word symbol and character sequence being
                     a list of character symbols (without eos)
        :param add_sil: True: add optional silence transition to finish word
        :param eow: alternative eow symbol, if None, the eow provided during
                    initialization is used
        """

        current_state = self.start_state
        add_word_id = True
        for char in word[1]:
            next_state = self.lex.add_state()
            if add_word_id:
                self.lex.add_arc(current_state, next_state, char, word[0])
                add_word_id = False
            else:
                self.lex.add_arc(current_state, next_state, char, self.eps)

            current_state = next_state

        if eow is None:
            eow = self.eow

        self.lex.add_arc(current_state, self.start_state, eow, self.eps)
        if self.sil is not None and add_sil:
            self.lex.add_arc(current_state, self.start_state,
                             self.sil, self.eps)

    def _add_word_trie(self, word, add_sil=True, eow=None):
        """
        Add entry for given word following existing transitions. This basically
        builds a trie. This puts the word id output at the final transition with
        eos as input back to the start state.

        :param word: tuple (word, character sequence) with word being the
                     word symbol and character sequence being
                     a list of character symbols (without eos)
        :param add_sil: True: add optional silence transition to finish word
        :param eow: alternative eow symbol, if None, the eow provided during
                    initialization is used
        """

        current_state = self.start_state
        for character in word[1]:
            arc = self.lex.find_arc(current_state, character, self.eps)
            if arc:
                next_state = arc.dst
            else:
                next_state = self.lex.add_state()
                self.lex.add_arc(current_state, next_state, character, self.eps)

            current_state = next_state

        if eow is None:
            eow = self.eow

        self.lex.add_arc(current_state, self.start_state, eow, word[0])
        if self.sil is not None and add_sil:
            self.lex.add_arc(current_state, self.start_state, self.sil, word[0])

    def _build_character_model_from_word_model(self, labels,
                                               character_loop_state):
        """
        Build character model form word model by copying the word word model as
        a character model. Output each input character, add missing transitions
        with labels from each state to character_loop_state and with eow or eoc
        from each state to start state. Only the transitions not existing in the
        word model are added to build a complementary model passing only
        character sequences which are not words.

        :param labels: list of symbols for all characters (excluding
                       special symbols like eps, eow or eos_label)
        :param character_loop_state: the character loop state. This has to be
                                     the first state after the word states!
        """

        for state_id in range(1, character_loop_state):
            self.lex.add_state()

        # Copy existing transitions from start state into character tree
        # (for prefixes).
        for arc in self.lex.get_arcs(self.start_state):
            if arc.dst < character_loop_state:
                self.lex.add_arc(self.start_state,
                                 character_loop_state + arc.dst,
                                 arc.ilabel, arc.ilabel)

        # Add remaining transitions into character_loop_state (for sequences)
        for label in labels:
            if not self.lex.find_arc(self.start_state, label):
                self.lex.add_arc(self.start_state, character_loop_state,
                                 label, label)

        # for prefixes: build character tree, add end of word after prefix
        # for sequences: add missing transitions into character_loop_state
        for state_id in range(1, character_loop_state):
            for arc in self.lex.get_arcs(state_id):
                if arc.ilabel not in [self.eow, self.sil]:
                    self.lex.add_arc(character_loop_state + state_id,
                                     character_loop_state + arc.dst,
                                     arc.ilabel, arc.ilabel)
            for label in labels:
                if not self.lex.find_arc(state_id, label):
                    self.lex.add_arc(character_loop_state + state_id,
                                     character_loop_state, label, label)
            if not self.lex.find_arc(state_id, self.eow):
                self.lex.add_arc(character_loop_state + state_id,
                                 self.start_state, self.eoc, self.eoc)


class Linear(Lexicon):
    """
    A lexicon in a linear form. Every transition is added separately not
    following existing transitions. The word label is placed on the first arc
    possibly allowing faster composition with the language model. For the
    character model, a character trie is build following existing transitions
    for the missing transitions.
    """

    def __init__(self, eps, eow, eoc=None, sil=None):
        """
        Construct lexicon trie.

        :param eps: eps symbol
        :param eow: end of word symbol
        :param eoc: end of characters symbol to terminate
                    character sequence for new word
        :param sil: character for space/pause/silence (if available)
        """

        super().__init__(eps, eow, eoc, sil)
        self.prefixes = {(): True}

    def add_word(self, word, mode='linear'):
        """
        Add linear entry for given word not following existing transitions. This
        puts the word id output on the first transition and adds a final
        transition with eow as input and eps as output back to the start state.

        :param word: tuple (word, character sequence) with word being the
                     word symbol and character sequence being
                     a list of character symbols (without eos)
        :param mode: 'linear' build linear lexicon
                     'trie' build trie lexicon
        """

        self.prefixes[tuple(word[1])] = True
        prefix_tuple = tuple()
        for char in word[1]:
            prefix_tuple += (char,)
            if prefix_tuple not in self.prefixes:
                self.prefixes[prefix_tuple] = False

        if mode == 'linear':
            self._add_word_linear(word)
        else:
            self._add_word_trie(word)

    def build_character_model(self, labels, mode='trie', sow=None):
        """
        Builds the character model passing all character sequences
        not being a word

        :param labels: list of symbols for all characters
                       (excluding special symbols like eps, eow or eos_label)
        :param mode: 'flat' for passing all character sequences,
                     'trie' for building a character trie
                     'linear' for adding linear character sequences not
                              following existing transitions and adding
                              transitions to the start state with eow for
                              prefixes and transitions to the character loop
                              state with labels for words
                     'copy' copy word trie to output characters
        :param sow: start of word symbol
        """

        # Add character_loop_state: output characters for sequences
        characters_loop_state = self._add_characters_loop(labels)

        if mode == 'flat':
            characters_start_state = self.start_state
            if sow is not None:
                characters_start_state = self.lex.add_state()
                self.lex.add_arc(self.start_state, characters_start_state,
                                 sow, sow)

            for label in labels:
                self.lex.add_arc(characters_start_state, characters_loop_state,
                                 label, label)

        elif mode == 'copy':
            # add further states: copy word trie to output characters for
            #   prefixes --> traverse tree or go to end of word after prefix
            #   sequences --> go to character_loop_state after prefix,
            #                 if no further prefix
            self._build_character_model_from_word_model(labels,
                                                        characters_loop_state)
        else:
            # add tree for prefixes
            #   add each prefix with eoc
            #   add transition to character_loop_state if no longer a prefix
            progress = tqdm(desc='Adding prefixes', total=len(self.prefixes))
            for prefix, is_word in self.prefixes.items():
                if mode == 'trie':
                    character_sequence_end_state =\
                        self._add_character_sequence_tri(prefix)
                else:
                    character_sequence_end_state =\
                        self._add_character_sequence_linear(prefix)

                if not is_word:
                    self.lex.add_arc(character_sequence_end_state,
                                     self.start_state, self.eoc, self.eoc)

                for label in labels:
                    if prefix + (label,) not in self.prefixes:
                        self.lex.add_arc(character_sequence_end_state,
                                         characters_loop_state, label, label)
                progress.update()
            progress.close()


def build_fst_for_lexicon(lexicon, eps, eow, build_character_model=False,
                          mode='trie', labels=None, sow=None, eoc=None,
                          sil=None):
    """
    Build a lexicon fst for given lexicon

    :param lexicon: dictionary containing word to label sequence mappings
                   (key: word, value: label sequence)
    :param eps: eps symbol
    :param eow: end of word symbol
    :param build_character_model: build a character model
    :param mode: mode for building the character model
                (see linear lexicon for details)
    :param labels: list of labels for character model
    :param sow: start of character sequence symbol
    :param eoc: end of character sequence symbol
    :param sil: character for space/pause/silence (if available)
    :return: lexicon fst of type linear
    """

    fst_lexicon = Linear(eps, eow, eoc, sil)

    progress = tqdm(desc='Adding words', total=len(lexicon))
    for word in lexicon.items():
        fst_lexicon.add_word(word)
        progress.update()
    progress.close()

    if build_character_model:
        fst_lexicon.build_character_model(labels, mode=mode, sow=sow)

    return fst_lexicon
