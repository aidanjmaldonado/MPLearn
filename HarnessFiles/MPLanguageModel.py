from math import *
import collections as clt


class LanguageModel():

    def tokenize(self, train_file):            

        # Read the file
        with open(train_file, 'r') as f:
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

        # Finished
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
    
    def perplexity(self, test_data):

        ## Tokenize test_data
        # Read the file
        with open(test_data, 'r') as f:
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
            for index, word in enumerate(sentence):
                if sentences[x][index] not in self.vocabulary:
                    sentences[x][index] = "<UNK>"

            # Append "Stop" to end of each sentence
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

        ## Now we are finished tokenizing
        # print(self.test_updated_text)

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
        self.tokenize(train_data)

        # Calculate the probabilities with regards to specified context range
        self.probabilities(interpolation, lambdas, 1 if interpolation else self.n)
        
    def test(self, test_data):
        
        # Calculate Perplexity
        return self.perplexity(test_data)