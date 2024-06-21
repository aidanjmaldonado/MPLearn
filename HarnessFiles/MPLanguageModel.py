from math import *
import numpy as np
import collections as clt
import random

class LanguageModel():

    def tokenize(self, input, type):            
        if type == "Train":
            # Read the file
            with open(input, 'r') as f:
                sentences = f.readlines()

            # Count each token
            self.token_occurences = clt.Counter()
            self.vocabulary = set()

            # Create Stop and Unique tokens
            stop = "<STOP>"
            unique = "<UNK>"

            # Tokenize the data, count the number of their occurences and build the vocabulary
            for x, sentence in enumerate(sentences):

                # Remove punctuation and newline
                sentences[x] = sentence.split()

                # Append "Stop" to end of each sentence
                sentences[x].append(stop)

                # Update the number of token occurences
                self.token_occurences.update(sentences[x])

                # Update the vocabulary
                self.vocabulary.update(sentences[x])


            ## Replace uncommon words with <UNK>
        
            # Initlize set of removed tokens to avoid double-removing
            self.removed = set()

            # Loop over sentences
            for x, sentence in enumerate(sentences):

                # Loop over words
                for i, token in enumerate(sentence):

                    # If the word is not frequent and has not already been removed, remove it
                    if self.token_occurences[token] < 3:

                        # Update the text
                        sentences[x][i] = unique

                        # Remove the token
                        if token not in self.removed:
                            self.vocabulary.remove(token)

                            # Place the token in the "removed" set
                            self.removed.add(token)

                            # Add "Unique" to the vocabulary, at least once
                            self.vocabulary.add(unique)
                        
                        # Increment the number of times UNK occurs
                        self.token_occurences[unique] += 1
                
            
            # Store the processed text
            self.updated_text = sentences

            # Define a Start token, prepend to each sentence
            numstart = self.n - 1
            start = "<START>"
            if numstart > 0:
                for x in enumerate(self.updated_text):
                    for y in range(numstart):
                        self.updated_text[x[0]].insert(0, start) 
                        self.token_occurences[start] += 1

        else:
            ## Tokenize test_data
            # Read the file
            with open(input, 'r') as f:
                sentences = f.readlines()

            # Count each token
            self.test_token_occurences = clt.Counter()

            # Create Stop and Unique tokens
            stop = "<STOP>"
            unique = "<UNK>"

            # Tokenize the data, count the number of their occurences and build the vocabulary
            for x, sentence in enumerate(sentences):

                # Remove punctuation and newline
                sentences[x] = sentence.split()

                # Update the number of token occurences
                self.test_token_occurences.update(sentences[x])

            # Replace uncommon words with <UNK>
            for x, sentence in enumerate(sentences):

                # Replace new words with UNK
                if type == "Test": # Fix this, we still want to use UNK probabilities
                    for index, word in enumerate(sentence):
                        if sentences[x][index] not in self.vocabulary:
                            sentences[x][index] = "<UNK>"

                # Append "Stop" to end of each sentence
                if x != len(sentences)-1 and type == "Test":
                    sentences[x].append(stop)
            
            # Store the processed text
            self.test_updated_text = sentences

            # Define a Start token, prepend to each sentence
            numstart = self.n - 1
            start = "<START>"
            if numstart > 0:
                for x in enumerate(self.test_updated_text):
                    for y in range(numstart):
                        self.test_updated_text[x[0]].insert(0, start) 
                        self.test_token_occurences[start] += 1
        # Finished
        self.vocabulary_list = sorted(list(self.vocabulary))
        return
    
    def probabilities(self, interpolation, lambdas, context):
        ##  Count the number of *unique* n-grams

        # Initialize Unique Grams Counters
        if (context == 1) or (interpolation == False):
            self.unique_n_grams = [clt.defaultdict(clt.Counter) for x in range(self.n)] if interpolation else  clt.defaultdict(clt.Counter)
        
        # Loop through every word/sentence in the updated text
        for sentence_num, sentence in enumerate(self.updated_text):
            for word_index in range(context - 1, len(sentence)):

                # Grab the Prefix and Suffix of N-Gram
                prefix = tuple(sentence[(word_index - (context - 1)):word_index])
                suffix = sentence[word_index]
                if suffix == '<START>':
                    continue

                # Update the Unique N-Grams dictionary
                if interpolation:
                    self.unique_n_grams[context - 1][prefix][suffix] += 1
                else:
                    self.unique_n_grams[prefix][suffix] += 1
        
        ## Math time!
                
        # Initialize gram probabilities: "What's the Probability that 'Suffix' occurs given 'Prefix'?" This is count(prefix & suffix) / count(prefix)
        if (context == 1) or (interpolation == False):
            self.gram_probabilities = [clt.defaultdict(lambda: clt.defaultdict(float)) for x in range(self.n)] if interpolation else clt.defaultdict(lambda: clt.defaultdict(float))

        # Loop over every prefix 
        unique_context_grams = (self.unique_n_grams[context - 1] if interpolation else self.unique_n_grams)
        context_gram_probabilities = clt.defaultdict(lambda: clt.defaultdict(float))
        if not interpolation:
            lambdas = [1 for x in range(self.n)]
        context_alpha = 0 if interpolation else self.alpha


        for n_gram_prefix in unique_context_grams:

            # Denominator = the number of times the prefix occurs
            denominator = sum(unique_context_grams[n_gram_prefix].values())

            # Loop over every suffix combination
            for n_gram_suffix in unique_context_grams[n_gram_prefix].keys():

                # Numerator = the number of times the given suffix occurs with the prefix
                numerator = unique_context_grams[n_gram_prefix][n_gram_suffix]

                # Compute gram probability, add alpha for smoothing and multiply by lambda weight if Linear Interpolation is applicable
                context_gram_probabilities[n_gram_prefix][n_gram_suffix] += (((numerator + context_alpha) / (denominator + ((len(self.vocabulary) * context_alpha)))) * lambdas[context - 1])


        if interpolation:
            self.gram_probabilities[context - 1] = context_gram_probabilities
        else:
            self.gram_probabilities = context_gram_probabilities

        # Recurr if using Linear Interpolation to gather weighted sum of probabilities across all contexts
        if context < (self.n) and interpolation:
            LanguageModel.probabilities(self, interpolation, lambdas, context + 1)

        # Aggregate all lower-order probabilities in the event of Linear Interpolation
        if interpolation and context == (self.n):
            for n_gram_prefix in unique_context_grams:
                for n_gram_suffix in unique_context_grams[n_gram_prefix]:
                    for n in range(self.n-1):
                        self.gram_probabilities[context - 1][n_gram_prefix][n_gram_suffix] += (self.gram_probabilities[n][n_gram_prefix[self.n - (n+1):]][n_gram_suffix])
    
        
            # Finalize
            self.gram_probabilities = self.gram_probabilities[context - 1]

        # Finished
        return
    
    def perplexity(self):

        # Initialize Sentence Probabilities: "What's the probability of this whole sentence occuring?" Use log probability
        self.sentence_probabilities = []

        # Iterate over every sentence; each one has a separate probability, take the log of each sentences' probability
        for i, sentence in enumerate(self.test_updated_text):
            sum_of_log_gram_probs = 0
            
            # Grab every n-gram from each sentence and add their probabilities into temp
            for word_index in range(len(sentence) - (self.n - 1)):
                prefix = tuple(sentence[word_index:word_index + (self.n - 1)])
                suffix = sentence[word_index + (self.n - 1)]
                
                # Sum the log probabilities of every (prefix & suffix) in the sentence
                sum_of_log_gram_probs += log2(self.gram_probabilities[prefix][suffix]) if self.gram_probabilities[prefix][suffix] > 0 else 0
                
            # Take the exponent of the log sum to achieve sentence probability
            self.sentence_probabilities.append(sum_of_log_gram_probs)

        # Total number of words in corpus
        numwords = 0
        
        for sentence in self.test_updated_text:

            for word in sentence:
                numwords += 1

        self.total_probability = sum(self.sentence_probabilities)
        
        # Calculate Perplexity
        perplexity = 2 ** ((-self.total_probability)/(numwords - (self.n - 1)))
        
        # Finished
        return perplexity


class N_Gram(LanguageModel):
    def __init__(self, n=1, alpha=0):
        self.n = n
        self.alpha = alpha

    def train(self, train_data, interpolation=False, lambdas=None):

        # Tokenize the input data and construct a vocabulary
        self.tokenize(train_data, "Train")

        # Calculate the probabilities with regards to specified context range
        self.probabilities(interpolation, lambdas, 1 if interpolation else self.n)
        
    def test(self, test_data):
        self.tokenize(test_data, "Test")
        
        # Calculate Perplexity
        return self.perplexity()
    

class Generator(LanguageModel):
    def __init__(self, N_Gram):
        self.N_Gram = N_Gram
        self.n = N_Gram.n
        self.vocabulary = N_Gram.vocabulary
        self.gram_probabilities = N_Gram.gram_probabilities
        # super().__init__(n, vocabulary, gram_probabilities)
        # self.n = N_Gram.n
        # self.voc

    def vectorfill(self, starter, top_k=1, temperature=1):
        
        # Tokenize starting text
        self.tokenize(starter, "Generate")
        
        # Finish off each sentence
        for i, sentence in enumerate(self.test_updated_text):

            while self.test_updated_text[i][-1] != "<STOP>":
                k_samples = [[float('inf'), float('inf')]]
                new_prefix = tuple(self.test_updated_text[i][-(self.n - 1):]) if self.n > 1 else ()

                for k in range(top_k):
                    greedy_word_info = self.greedy_word(new_prefix, k_samples[-1][0])
                    if greedy_word_info[0] != 0:
                        k_samples.append(greedy_word_info)

                # Select next word from sample, and if no next word found, add a <UNK>
                if len(k_samples) == 1:
                    self.test_updated_text[i].append("<UNK>")
                    continue

                new_word_index = self.select_word(k_samples[1:], temperature)
                new_word = self.vocabulary_list[new_word_index]

                self.test_updated_text[i].append(new_word)
            
            # Yield the completed sentence
            yield(" ".join(self.test_updated_text[i][self.n-1:-1]))

    def greedy_word(self, prefix, ceiling=float('inf')):
        newvector = [self.gram_probabilities[prefix][y] for y in self.vocabulary_list]
        maxprob = 0
        maxindex = -1
        for i, prob in enumerate(newvector):
            if prob > maxprob and prob < ceiling:
                maxprob = prob
                maxindex = i
        return [maxprob, maxindex]
    
    def select_word(self, k_samples, temperature=1):

        # Normalizing constant
        normalizer = 0

        for i in range(len((k_samples))):
            
            # Divide by temperature
            k_samples[i][0] = k_samples[i][0] / temperature 

            # Take the exponential
            k_samples[i][0] = exp(k_samples[i][0])

            # Increment the normalization constant
            normalizer += k_samples[i][0]
        
        # Normalize every probability
        for i in range(len(k_samples)):
            
            k_samples[i][0] /= normalizer

        # Randomly pick from this weighted distribution
        toarray = np.array(k_samples).T
        index = random.choices(toarray[1], toarray[0], k=1)
        
        return int(index[0])
            


## To-Do
"""
class Generator DONE

class NaiveBayes Predictor

class chatbot

class predictive text

""" 


