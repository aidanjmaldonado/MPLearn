import numpy as np 
from math import *
import collections as clt


class LM():

    def tokenize(self, train_file):            

        # Read the file
        with open(train_file, 'r') as f:
            sentences = f.readlines()

        sentences = sentences[0:1000]

        # Count each token
        self.token_occurences = clt.Counter()
        self.unique_tokens = set()

        # Create Stop and Unique tokens
        stop = "<STOP>"
        unique = "<UNK>"

        # Remove \n and punctuation as well as split into tokens
        for x, y in enumerate(sentences):

            # Remove punctuation and newline
            # sentences[x] = y.strip("\n").strip(" .").split() # This yields 26605 unique tokens
            sentences[x] = y.split() # This yields 26602 unique tokens

            # Append "Stop" to end of each sentence
            sentences[x].append(stop)
            for z in sentences[x]:

                # Add the token to the unique token set
                self.unique_tokens.add(z)

        ## Build the vocabulary set and replace uncommon words with <UNK>
                
        # "Total" Counts the num occurences of each token totalled across every sentence
        total = 0
        for x in self.unique_tokens:
            for y in sentences:
                total += y.count(x)
            
            # If rare token, make it Unique
            if total < 3:
                self.token_occurences[unique] += total
            # Else, add to the dictionary
            else:
                self.token_occurences[x] = total
            total = 0
        self.vocabulary = set(self.token_occurences.keys())

        ## Now we have vocabulary, token_occurences, and unique_tokens

        # All that's left to do is store the updated text with Unique tokens removed!
        for w, x in enumerate(sentences):
            for y, z in enumerate(x):
                if z not in self.vocabulary:
                    sentences[w][y] = unique
        
        # Store the processed text
        self.updated_text = sentences
         
        # Finished
        return

    def probabilities(self, n, alpha):

        # Define a Start token, prepend to each sentence
        numstart = n - 1
        start = "<START>"
        if numstart > 0:
            for x in enumerate(self.updated_text):
                for y in range(numstart):
                    self.updated_text[x[0]].insert(0, start) 

        ##  Count the number of *unique* n-grams
            
        # Initialize vars
        prefex = tuple()
        suffix = ""
        self.unique_n_grams = clt.defaultdict(clt.Counter)

        # Loop through every word/sentence in the updated text
        for sentence_num, sentence in enumerate(self.updated_text):
            for word_index in range(len(sentence) - (n - 1)):

                # Grab the Prefix and Suffix of N-Gram
                prefix = tuple(sentence[word_index:word_index + (n - 1)])
                suffix = sentence[word_index + (n - 1)]

                # Update the Unique N-Grams dictionary
                self.unique_n_grams[prefix][suffix] += 1

        
        ## Math time!
        
        # Initialize gram probabilities: "What's the Probability that 'Suffix' occurs given 'Prefix'?" This is count(prefix & suffix) / count(prefix)
        self.gram_probabilities = clt.defaultdict(lambda: clt.defaultdict(float))
        ##### TO-DO: Interpolation

        # Loop over every prefix 
        for n_gram_prefix in self.unique_n_grams:

            # Denominator = the number of times the prefix occurs
            denominator = sum(self.unique_n_grams[n_gram_prefix].values())

            # Loop over every suffix combination
            for n_gram_suffix in self.unique_n_grams[n_gram_prefix].keys():

                # Numerator = the number of times the given suffix occurs with the prefix
                numerator = self.unique_n_grams[n_gram_prefix][n_gram_suffix]

                ## Compute gram probability, add alpha for smoothing
                self.gram_probabilities[n_gram_prefix][n_gram_suffix] = ((numerator + alpha) / (denominator + ((len(self.vocabulary) * alpha))))
            
        # # Initialize sentence probabilities: "What's the probability of this whole sentence occuring?" Use log probability
        # self.sentence_probabilities = []

        # # Iterate over every sentence; each one has a separate probability
        # for i, sentence in enumerate(self.updated_text):
        #     temp = 0
            
        #     # Grab every n-gram from each sentence and add their probabilities into temp
        #     for word_index in range(len(sentence) - (n - 1)):
        #         prefix = tuple(sentence[word_index:word_index + (n - 1)])
        #         suffix = sentence[word_index + (n - 1)]
                
        #         # Sum the probability of every (prefix & suffix) in the sentence
        #         temp += log(self.gram_probabilities[prefix][suffix])
            
        #     # Take the exponent of the log sum to achieve sentence probability
        #     self.sentence_probabilities.append((e**(temp)))
        #     temp = 0

        # Finished
        return
    
    def perplexity(self, n, test_data):

        # Read the file
        with open(test_data, 'r') as f:
            sentences = f.readlines()

        sentences = sentences[0:1000]

        # Count each token
        self.test_token_occurences = clt.Counter()
        self.test_unique_tokens = set()

        # Create Stop and Unique tokens
        stop = "<STOP>"
        unique = "<UNK>"

        # Remove \n and punctuation as well as split into tokens
        for x, y in enumerate(sentences):

            # Remove punctuation and newline
            # sentences[x] = y.strip("\n").strip(" .").split() # This yields 26605 unique tokens
            sentences[x] = y.split() # This yields 26602 unique tokens

            # Append "Stop" to end of each sentence
            sentences[x].append(stop)
            for z in sentences[x]:

                # Add the token to the unique token set
                self.test_unique_tokens.add(z)

        ## Build the vocabulary set and replace uncommon words with <UNK>
                
        # "Total" Counts the num occurences of each token totalled across every sentence
        total = 0
        for x in self.test_unique_tokens:
            for y in sentences:
                total += y.count(x)
            
            # If rare token, make it Unique
            if total < 3 or (x not in self.vocabulary):
                self.test_token_occurences[unique] += total
            # Else, add to the dictionary
            else:
                self.test_token_occurences[x] = total
            total = 0
        self.test_vocabulary = set(self.test_token_occurences.keys())

        ## Now we have vocabulary, token_occurences, and unique_tokens

        # All that's left to do is store the updated text with Unique tokens and new words removed!
        for w, x in enumerate(sentences):
            for y, z in enumerate(x):
                if z not in self.test_vocabulary:
                    sentences[w][y] = unique
        
        # Store the processed text
        self.test_updated_text = sentences

        # Define a Start token, prepend to each sentence
        numstart = n - 1
        start = "<START>"
        if numstart > 0:
            for x in enumerate(self.test_updated_text):
                for y in range(numstart):
                    self.test_updated_text[x[0]].insert(0, start) 

         
        ## Finished with the old
        
        # Sentence Probabilities
        # Initialize sentence probabilities: "What's the probability of this whole sentence occuring?" Use log probability
        self.sentence_probabilities = []

        # Iterate over every sentence; each one has a separate probability
        for i, sentence in enumerate(self.test_updated_text):
            temp = 0
            
            # Grab every n-gram from each sentence and add their probabilities into temp
            for word_index in range(len(sentence) - (n - 1)):
                prefix = tuple(sentence[word_index:word_index + (n - 1)])
                suffix = sentence[word_index + (n - 1)]
                
                # Sum the probability of every (prefix & suffix) in the sentence
                temp += (self.gram_probabilities[prefix][suffix] + 1e-10)
            
            # Take the exponent of the log sum to achieve sentence probability
            self.sentence_probabilities.append((log(temp)))
            print("Sentence prob", e**temp)
            temp = 0

        
        # Total number of words in corpus
        M = 0
        for x in self.test_updated_text:
            for y in x:
                M += 1
        print("M", M)
        # Average log probabilty per word
        D = 0
        for p in self.sentence_probabilities:
            # D += log(p + 1e-10)
            D += p
            
        # l = average log probability per word
        l = (1 / M) * D        

        # Return perplexity
        print(l)
        return e**(-l)


class N_Gram(LM):
    def __init__(self, n=1, alpha=0):
        self.n = n
        self.alpha = alpha

    def train(self, train_data,):

        # Tokenize the data
        self.tokenize(train_data)

        # Calculate the probabilities
        self.probabilities(self.n, self.alpha)
        
    def test(self, test_data):
        
        # Calculate Perplexity
        return self.perplexity(self.n, test_data)



    
    