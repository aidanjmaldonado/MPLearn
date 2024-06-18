import MPLanguageModel as mplm

bi = mplm.N_Gram(2, 1e-10)

train_data = "./A2-Data/1b_benchmark.train.tokens"
test_data = "./A2-Data/starter.txt"

print("Training Bigram")
bi.train(train_data)

print("Testing Bigram")
print(f"Perplexity: {bi.test(test_data)}")

sentence_starter = "./A2-Data/starter.txt"
print(f"\nUsing the sentence starter:\n      {sentence_starter}\n We are going to generate an end to this sentence.")

# Create a Generator
textGen = mplm.Generator(bi)
textGen.vectorfill(sentence_starter, "Sample", top_k=100)