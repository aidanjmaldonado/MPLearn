from MPLanguageModel import *



def main():

    # Data to be tokenized
    # training = ['./A2-Data/1b_benchmark.train.tokens', './A2-Data/small.train.tokens', './A2-Data/smaller.train.tokens']
    training = ['./A2-Data/1b_benchmark.train.tokens', './A2-Data/myfile.tokens', './A2-Data/smaller.train.tokens']
    training_data = training[0]

    # Data to be tested
    testing_data = "./A2-Data/1b_benchmark.test.tokens"
    # testing_data = training[1]

    # Initialize an N-Graph and Alpha
    n = 1
    alpha = 1e-10
    LM = N_Gram(n, alpha)

    # Train the Unigram on the training data
    LM.train(training_data)

    # Test the trained Unigram on the testing data
    print("{}-gram, alpha = {}         :-=-=:         Perplexity = {}".format(n, alpha, LM.test(testing_data)))



if __name__ == "__main__":
    main()
