from MPLanguageModel import *

def main():

    # Data to be trained
    train_data = "./A2-Data/1b_benchmark.train.tokens"
    # train_data = "./A2-Data/myfile.train.tokens"

    # Dev Data
    dev_data = "./A2-Data/1b_benchmark.dev.tokens"
    # dev_data = "./A2-Data/short.train.tokens"

    # Data to be tested
    test_data = "./A2-Data/1b_benchmark.test.tokens"
    # test_data = "./A2-Data/short.test.tokens"
    # test_data = "./A2-Data/myfile.train.tokens"



    ### Initialize N-Gram Models

    # Uni Gram Model
    Uni = N_Gram(1, alpha=1e-8)

    # Bi Gram Model
    Bi = N_Gram(2, alpha=1e-8)

    # Tri Gram Model
    Tri = N_Gram(3, alpha=1e-8)

    # Quad Gram Model
    Quad = N_Gram(4, alpha=1e-8)
    
    # Penta Gram Model
    Penta = N_Gram(5, alpha=1e-8)

    # Fifty Gram Model (Overkill)
    Fifty = N_Gram(5, alpha=1e-8)



    print(f"\n\n    -----   Showcase Without Linear Interpolation  -----   ")
    print(f"\nUni-Gram Perplexity:")
    Uni.train(train_data)
    print(f"Train: {Uni.perplexity(train_data)}")
    print(f"Dev:   {Uni.perplexity(dev_data)}")
    print(f"Test:  {Uni.perplexity(test_data)}")

    print(f"\nBi-Gram Perplexity:")
    Bi.train(train_data)
    print(f"Train: {Bi.perplexity(train_data)}")
    print(f"Dev:   {Bi.perplexity(dev_data)}")
    print(f"Test:  {Bi.perplexity(test_data)}")

    print(f"\nTri-Gram Perplexity:")
    Tri.train(train_data)
    print(f"Train: {Tri.perplexity(train_data)}")
    print(f"Dev:   {Tri.perplexity(dev_data)}")
    print(f"Test:  {Tri.perplexity(test_data)}")

    print(f"\nQuad-Gram Perplexity:")
    Quad.train(train_data)
    print(f"Train: {Quad.perplexity(train_data)}")
    print(f"Dev:   {Quad.perplexity(dev_data)}")
    print(f"Test:  {Quad.perplexity(test_data)}")

    print(f"\nPenta-Gram Perplexity:")
    Penta.train(train_data)
    print(f"Train: {Penta.perplexity(train_data)}")
    print(f"Dev:   {Penta.perplexity(dev_data)}")
    print(f"Test:  {Penta.perplexity(test_data)}")

    # print(f"\nFifty-Gram Perplexity:")
    # Fifty.train(train_data)
    # print(f"Train: {Fifty.perplexity(train_data)}")
    # print(f"Dev:   {Fifty.perplexity(dev_data)}")
    # print(f"Test:  {Fifty.perplexity(test_data)}")



    print(f"\n\n    -----   Showcase With Linear Interpolation  -----   ")
    print(f"\nUni-Gram Perplexity::")
    Uni.train(train_data, interpolation=True, lambdas=[1.0])
    print(f"Train: {Uni.perplexity(train_data)}")
    print(f"Dev:   {Uni.perplexity(dev_data)}")
    print(f"Test:  {Uni.perplexity(test_data)}")

    print(f"\nBi-Gram Perplexity::")
    Bi.train(train_data, interpolation=True, lambdas=[0.3, 0.7])
    print(f"Train: {Bi.perplexity(train_data)}")
    print(f"Dev:   {Bi.perplexity(dev_data)}")
    print(f"Test:  {Bi.perplexity(test_data)}")

    print(f"\nTri-Gram Perplexity::")
    Tri.train(train_data, interpolation=True, lambdas=[0.1, 0.3, 0.6])
    print(f"Train: {Tri.perplexity(train_data)}")
    print(f"Dev:   {Tri.perplexity(dev_data)}")
    print(f"Test:  {Tri.perplexity(test_data)}")

    print(f"\nQuad-Gram Perplexity::")
    Quad.train(train_data, interpolation=True, lambdas=[0.2, 0.2, 0.3, 0.3])
    print(f"Train: {Quad.perplexity(train_data)}")
    print(f"Dev:   {Quad.perplexity(dev_data)}")
    print(f"Test:  {Quad.perplexity(test_data)}")

    print(f"\nPenta-Gram Perplexity::")
    Penta.train(train_data, interpolation=True, lambdas=[0.1, 0.2, 0.2, 0.2, 0.3])
    print(f"Train: {Penta.perplexity(train_data)}")
    print(f"Dev:   {Penta.perplexity(dev_data)}")
    print(f"Test:  {Penta.perplexity(test_data)}")

    # print(f"\Fifty-Gram Perplexity::")
    # Fifty.train(train_data, interpolation=True, lambdas=[0.1, 0.2, 0.2, 0.2, 0.3])
    # print(f"Train: {Fifty.perplexity(train_data)}")
    # print(f"Dev:   {Fifty.perplexity(dev_data)}")
    # print(f"Test:  {Fifty.perplexity(test_data)}")



if __name__ == "__main__":
    main()
