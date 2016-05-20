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
from libcpp.pair cimport pair
from libcpp cimport bool

DEF EOS = 5
DEF SOS = 4
DEF EOW = 3
DEF SOW = 2
DEF PHI = 1
DEF UNKNOWN = -32766
DEF DELETED = -32767
DEF EMPTY = -32768

ctypedef vector[int].iterator citerator
ctypedef vector[int].iterator witerator
ctypedef vector[int].iterator iiterator

ctypedef vector[int].const_iterator const_citerator
ctypedef vector[int].const_iterator const_witerator
ctypedef vector[int].const_iterator const_iiterator

ctypedef pair[int, bool] WordIdAddedPair

cdef extern from "NHPYLM/definitions.hpp":
    cdef struct ContextToContextTransitions:
            vector[int] Words
            vector[int] NextContextIds
            vector[double] Probabilities
            bool HasTransitionToSentEnd
    cdef struct NHPYLMParameters:
            const vector[double] & CHPYLMDiscount
            const vector[double] & CHPYLMConcentration
            const vector[double] & WHPYLMDiscount
            const vector[double] & WHPYLMConcentration

cdef extern from "NHPYLM/NHPYLM.hpp":
    cdef cppclass NHPYLM:
        NHPYLM(unsigned int CHPYLMOrder_, unsigned int WHPYLMOrder_,
               const vector[string] & Symbols_, int CharactersBegin_,
               const double WordBaseProbability_)
        void AddWordToLm(const const_witerator & Word)
        void AddWordSequenceToLm(const vector[int] & WordSequence)
        void RemoveWordSequenceFromLm(const vector[int] & WordSequence)
        bool RemoveWordFromLm(const const_witerator & Word)
        double WordProbability(const const_witerator & Word) const
        vector[double] WordVectorProbability(
                const vector[int] & ContextSequence,
                const vector[int] & Words) const
        double WordSequenceLoglikelihood(const vector[int] & WordSequence) const
        void ResampleHyperParameters()
        const NHPYLMParameters & GetNHPYLMParameters() const
        int GetContextId(const vector[int] & ContextSequence) const
        ContextToContextTransitions GetTransitions(
                int ContextId,
                int SentEndWordId,
                const vector[bool] & ActiveWords) const
        int GetFinalContextId() const
        int GetRootContextId() const
        int GetCHPYLMOrder() const
        int GetWHPYLMOrder() const
        vector[int] GetTotalCountPerLevelFor(const string & LM,
                                             const string & CountName) const
        vector[vector[int]] Generate(
                string Mode, int NumWorsdOrCharacters,
                int SentEndWordId,
                vector[double] *GeneratedWordLengthDistribution_)
        void SetWHPYLMBaseProbabilitiesScale(
                const vector[double] & WHPYLMBaseProbabilitiesScale_)
        const vector[double] & GetWHPYLMBaseProbabilitiesScale() const
        int GetWHPYLBaseTablesPerWord(int WordId) const
        void SetParameter(const string & LM, const string & Parameter,
                          int Level, double Value)
        # From Dictionary
        int GetMaxNumWords() const
        int GetWordsBegin() const
        int GetWordId(const const_citerator, unsigned int length)
        WordIdAddedPair AddCharacterIdSequenceToDictionary(const const_citerator,
                                                           unsigned int length)
        vector[string] GetId2CharacterSequenceVector()
        vector[vector[string]] GetId2SeparatedCharacterSequenceVector()
        vector[int] GetWordVector(int id)
        void SetCharBaseProb(const int CharId, const double prob)
