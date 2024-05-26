from MPLanguageModel import *
import sys



def main():

    # Data to be trained
    train_data = "./A2-Data/1b_benchmark.train.tokens"
    # train_data = "./A2-Data/short.train.tokens"

    # Dev Data
    dev_data = "./A2-Data/1b_benchmark.dev.tokens"
    # dev_data = "./A2-Data/short.train.tokens"

    # Data to be tested
    test_data = "./A2-Data/1b_benchmark.test.tokens"
    # test_data = "./A2-Data/short.test.tokens"


    # Initialize N-Gram Models and Alpha
    # alpha = 1e-10
    alpha = 0.01 # Larger alpha works better for this large of a dataset

    """
    n = 1
    LM = N_Gram(n, alpha)
    """

    Uni = N_Gram(1, alpha)
    Bi = N_Gram(2, alpha)
    Tri = N_Gram(3, alpha)

    """
    ## N-Gram showcase
    print("\n\n-=-=-=-=- N-Gram Showcase -=-=-=-=-\n")

    # Train the N-Gram on the Training Data
    print("Training {}-gram...".format(n))
    LM.train(train_data)
    print("Done\n")

    # Test the trained N-Gram on the Dev Data
    print(f"Testing {n}-gram, alpha = {alpha} on Training Data...")
    print(f"Perplexity = {LM.test(test_data)}\n")

     # Test the trained N-Gram on the Dev Data
    print(f"Testing {n}-gram, alpha = {alpha} on Dev Data...")
    print(f"Perplexity = {LM.test(dev_data)}\n")

    # Test the trained N-Gram on the Testing Data
    print("Testing {}-gram, alpha = {} on Test Data...".format(n, alpha))
    print(f"Perplexity = {LM.test(test_data)}\n")
    """


    ## Uni-Gram showcase
    print("\n\n-=-=-=-=- Uni-Gram Showcase -=-=-=-=-\n")

    # Train the Uni-Gram on the Training Data
    print("Training {}-gram...".format(1))
    Uni.train(train_data)
    print("Done\n")

    # Test the trained Uni-Gram on the Dev Data
    print(f"Testing {1}-gram, alpha = {alpha} on Train Data...")
    print(f"Perplexity = {Uni.test(train_data)}\n")

     # Test the trained Uni-Gram on the Dev Data
    print(f"Testing {1}-gram, alpha = {alpha} on Dev Data...")
    print(f"Perplexity = {Uni.test(dev_data)}\n")

    # Test the trained Uni-Gram on the Testing Data
    print("Testing {}-gram, alpha = {} on Test Data...".format(1, alpha))
    print(f"Perplexity = {Uni.test(test_data)}\n")


    ## Bi-Gram showcase
    print("\n\n-=-=-=-=- Bi-Gram Showcase -=-=-=-=-\n")

    # Train the Bi-Gram on the Training Data
    print("Training {}-gram...".format(2))
    Bi.train(train_data)
    print("Done\n")

    # Test the trained N-Gram on the Dev Data
    print(f"Testing {2}-gram, alpha = {alpha} on Train Data...")
    print(f"Perplexity = {Bi.test(train_data)}\n")

     # Test the trained N-Gram on the Dev Data
    print(f"Testing {2}-gram, alpha = {alpha} on Dev Data...")
    print(f"Perplexity = {Bi.test(dev_data)}\n")

    # Test the trained N-Gram on the Testing Data
    print("Testing {}-gram, alpha = {} on Test Data...".format(2, alpha))
    print(f"Perplexity = {Bi.test(test_data)}\n")


    ## Tri-Gram showcase
    print("\n\n-=-=-=-=- Tri-Gram Showcase -=-=-=-=-\n")

    # Train the Tri-Gram on the Training Data
    print("Training {}-gram...".format(3))
    Tri.train(train_data)
    print("Done\n")

    # Test the trained Tri-Gram on the Dev Data
    print(f"Testing {3}-gram, alpha = {alpha} on Train Data...")
    print(f"Perplexity = {Tri.test(train_data)}\n")

     # Test the trained Tri-Gram on the Dev Data
    print(f"Testing {3}-gram, alpha = {alpha} on Dev Data...")
    print(f"Perplexity = {Tri.test(dev_data)}\n")

    # Test the trained Tri-Gram on the Testing Data
    print("Testing {}-gram, alpha = {} on Test Data...".format(3, alpha))
    print(f"Perplexity = {Tri.test(test_data)}\n")

    



if __name__ == "__main__":
    main()
