import MPLanguageModel as mplm


# Input Data
train_data = "./A2-Data/1b_benchmark.train.tokens"
test_data = "./A2-Data/1b_benchmark.test.tokens"
sentence_starters = "./A2-Data/starter.txt"

# Create an N-Gram Language Model
# Note: First parameter is 'n' grams, ie number of tokens in a gram. n-1 tokens for prefix context, 1 token for current suffix
# Note: Second parameter is alpha-value for smoothing. Closer to 0 tends towards less Perplexity
Trigram = mplm.N_Gram(3, 1e-10)

# Train the N-Gram Language Model
# Note: Linear Interpolation helps reduce perplexity, test with different lambda coefficients
print("\nTraining . . .")
Trigram.train(train_data, interpolation=True, lambdas=[0.1, 0.3, 0.6])

# Test the Perplexity of the N-Gram Model
print("\nTesting . . .")
print(f"Perplexity: {Trigram.test(test_data)}")

# Create a Text Generator
TextGen = mplm.Generator(Trigram)

# Generate an end to each sentene in the input file
# Note: top_k means the sample size of most probable 'Next Words' most to randomly choose from
# Note: temperature refers to 'how random' the selection is, choose close to 0 for Greedy, close to infinity for Uniform
print("\nGenerating . . .")
for filled in TextGen.vectorfill(sentence_starters, top_k=3, temperature = 0.05):
    print(filled)