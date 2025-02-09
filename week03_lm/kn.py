from tqdm.auto import tqdm
from collections import defaultdict, Counter

from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

UNK, EOS = "_UNK_", "_EOS_"

def count_ngrams(lines, n):
    """
    Count how many times each word occured after (n - 1) previous words
    :param lines: an iterable of strings with space-separated tokens
    :returns: a dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}

    When building counts, please consider the following two edge cases:
    - if prefix is shorter than (n - 1) tokens, it should be padded with UNK. For n=3,
      empty prefix: "" -> (UNK, UNK)
      short prefix: "the" -> (UNK, the)
      long prefix: "the new approach" -> (new, approach)
    - you should add a special token, EOS, at the end of each sequence
      "... with deep neural networks ." -> (..., with, deep, neural, networks, ., EOS)
      count the probability of this token just like all others.
    """
    counts = defaultdict(Counter)
    # counts[(word1, word2)][word3] = how many times word3 occured after (word1, word2)

    for line in lines:
        line = [UNK] * (n - 1) + line.split(" ") + [EOS]
        for word_idx in range(n - 1, len(line)):
            word = line[word_idx]
            context = tuple(line[word_idx - n + 1 : word_idx])
            counts[context][word] += 1
    
    return counts


class NGramLanguageModel:    
    def __init__(self, lines, n):
        """ 
        Train a simple count-based language model: 
        compute probabilities P(w_t | prefix) given ngram counts
        
        :param n: computes probability of next token given (n - 1) previous words
        :param lines: an iterable of strings with space-separated tokens
        """
        assert n >= 1
        self.n = n
    
        counts = count_ngrams(lines, self.n)
        
        # compute token proabilities given counts
        self.probs = defaultdict(Counter)
        # probs[(word1, word2)][word3] = P(word3 | word1, word2)
        
        # populate self.probs with actual probabilities
        for prefix, words in counts.items():
            num_all_worlds = sum(words.values())
            for word, count in words.items():
                self.probs[prefix][word] = count / num_all_worlds
            
    def get_possible_next_tokens(self, prefix):
        """
        :param prefix: string with space-separated prefix tokens
        :returns: a dictionary {token : it's probability} for all tokens with positive probabilities
        """
        prefix = prefix.split()
        prefix = prefix[max(0, len(prefix) - self.n + 1):]
        prefix = [ UNK ] * (self.n - 1 - len(prefix)) + prefix
        return self.probs[tuple(prefix)]
    
    def get_next_token_prob(self, prefix, next_token):
        """
        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :returns: P(next_token|prefix) a single number, 0 <= P <= 1
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)


class KneserNeyLanguageModel(NGramLanguageModel): 
    """ A template for Kneser-Ney language model. Default delta may be suboptimal. """
    def __init__(self, lines, n, delta=1.0):
        self.n = n
        self.probs = None
        
        for ngramm_size in range(1, self.n + 1):
            if ngramm_size == 1:
                self.probs = self._compute_initial_probas(lines)
                self.vocab = set(self.probs[()].keys())
            else:
                ngram_counts = count_ngrams(lines, n=ngramm_size)
                kn_probas = defaultdict(Counter)

                for prefix, next_words in tqdm(ngram_counts.items(), leave=False):
                    total_counts = sum(next_words.values())
                    lmbda  = delta * len(next_words) / total_counts
                    kn_probas[prefix] = {word : lmbda * proba for word, proba in self.probs[prefix[1:]].items()}
                    
                    for word, count in next_words.items():
                        curr_proba = max(0, count - delta) / total_counts
                        kn_probas[prefix][word] = curr_proba + kn_probas[prefix][word]
                    
                self.probs = kn_probas.copy()
                kn_probas.clear()

    def _build_vocab(self, lines):
        vocab = set()
        for line in lines:
            for word in line.split():
                vocab.add(word)
        vocab.add(UNK)
        vocab.add(EOS)
        return vocab

    def _compute_initial_probas(self, lines):
        words2bigrams = defaultdict(set)
        
        for line in tqdm(lines, leave=False):
            tokens = [UNK] + line.split() + [EOS]
            for i in range(1, len(tokens)):
                words2bigrams[tokens[i]].add(tuple(tokens[i - 1 : i + 1]))
        N = sum(len(bigrams) for bigrams in words2bigrams.values())
        
        probas = defaultdict(Counter)
        probas[()] = {word : len(bigrams) / N for word, bigrams in words2bigrams.items()}
        probas[()][UNK] = 0
        return probas
        
    def get_possible_next_tokens(self, prefix):
        """
        :param prefix: string with space-separated prefix tokens
        :returns: a dictionary {  : it's probability} for all tokens with positive probabilities
        """
        next_tokens = super().get_possible_next_tokens(prefix)
        return next_tokens
        # missing_proba = 1 - sum(next_tokens.values())
        # missing_proba = missing_proba / max(1, len(self.vocab) - len(next_tokens))
        # return {token: next_tokens.get(token, missing_proba) for token in self.vocab}
    
    def get_next_token_prob(self, prefix, next_token):
        """
        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :returns: P(next_token|prefix) a single number, 0 <= P <= 1
        """
        proba = self.get_possible_next_tokens(prefix).get(next_token, 0)
        return proba
    

def tokenize(sentence: str):
    sentence = sentence.lower()
    tokens = tokenizer.tokenize(sentence)
    return " ".join(tokens)


def perplexity(lm, lines, min_logprob=np.log(10 ** -50.)):
    """
    :param lines: a list of strings with space-separated tokens
    :param min_logprob: if log(P(w | ...)) is smaller than min_logprop, set it equal to min_logrob
    :returns: corpora-level perplexity - a single scalar number from the formula above
    
    Note: do not forget to compute P(w_first | empty) and P(eos | full_sequence)
    
    PLEASE USE lm.get_next_token_prob and NOT lm.get_possible_next_tokens
    """
    N = 0
    perplexity = 0
    for line in lines:
        tokens = line.split() + [EOS]
        N += len(tokens)

        for i in range(len(tokens)):
            curr_token = tokens[i]
            prefix = " ".join(tokens[:i])
            
            porb = lm.get_next_token_prob(prefix, curr_token)
            perplexity += np.log(porb) if np.exp(min_logprob) < porb else min_logprob
    
    return np.exp(- perplexity / N)


if __name__ == "__main__":
    data = pd.read_json("./arxivData.json")
    lines = data.apply(lambda row: row['title'] + ' ; ' + row['summary'].replace("\n", ' '), axis=1).tolist()

    tokenizer = WordPunctTokenizer()
    lines = [tokenize(line) for line in lines]

    train_lines, test_lines = train_test_split(lines, test_size=0.25, random_state=42)
    for n in (1, 2, 3):
        lm = KneserNeyLanguageModel(train_lines, n=n, delta=0.4)
        ppx = perplexity(lm, test_lines)
        print("N = %i, Perplexity = %.5f" % (n, ppx), flush=True)