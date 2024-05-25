from MPLanguageModel import *



def main():

    # Data to be trained
    train_data = "./A2-Data/1b_benchmark.train.tokens"

    # Dev Data
    dev_data = "./A2-Data/1b_benchmark.dev.tokens"

    # Data to be tested
    test_data = "./A2-Data/1b_benchmark.test.tokens"

    # Initialize an N-Gram and Alpha
    n = 3
    # alpha = 1e-10
    alpha = 0.0001
    LM = N_Gram(n, alpha)

    # Train the Unigram on the Training Data
    print("Training {}-gram...".format(n))
    LM.train(train_data)
    print("Done\n")

        # Test the trained Unigram on the Dev Data
    print(f"Testing {n}-gram, alpha = {alpha} on Training Data...")
    print(f"Perplexity = {LM.test(train_data)}\n")

     # Test the trained Unigram on the Dev Data
    print(f"Testing {n}-gram, alpha = {alpha} on Dev Data...")
    print(f"Perplexity = {LM.test(dev_data)}\n")

    # Test the trained Unigram on the Testing Data
    print("Testing {}-gram, alpha = {} on Test Data...".format(n, alpha))
    print(f"Perplexity = {LM.test(test_data)}\n")



if __name__ == "__main__":
    main()
