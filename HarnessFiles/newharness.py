import MPLanguageModel as mplm


# Input Data
train_data = "./A2-Data/1b_benchmark.train.tokens"
test_data = "./A2-Data/1b_benchmark.test.tokens"
sentence_starters = "./A2-Data/starter.txt"

# Create an N-Gram Language Model
bi = mplm.N_Gram(3, 1e-10)

# Train the N-Gram Language Model
print("\nTraining . . .")
bi.train(train_data)

# Test the Perplexity of the N-Gram Model
print("\nTesting . . .")
print(f"Perplexity: {bi.test(test_data)}")

# Create a Text Generator
textGen = mplm.Generator(bi)

# Generate an end to each sentene in the input file
print("\nGenerating . . .")
for filled in textGen.vectorfill(sentence_starters, top_k=10, temperature = 0.01):
    print(filled)