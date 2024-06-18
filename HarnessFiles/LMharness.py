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
    print(f"Train: {Uni.test(train_data)}")
    print(f"Dev:   {Uni.test(dev_data)}")
    print(f"Test:  {Uni.test(test_data)}")

    print(f"\nBi-Gram Perplexity:")
    Bi.train(train_data)
    print(f"Train: {Bi.test(train_data)}")
    print(f"Dev:   {Bi.test(dev_data)}")
    print(f"Test:  {Bi.test(test_data)}")

    print(f"\nTri-Gram Perplexity:")
    Tri.train(train_data)
    print(f"Train: {Tri.test(train_data)}")
    print(f"Dev:   {Tri.test(dev_data)}")
    print(f"Test:  {Tri.test(test_data)}")

    # print(f"\nQuad-Gram Perplexity:")
    # Quad.train(train_data)
    # print(f"Train: {Quad.test(train_data)}")
    # print(f"Dev:   {Quad.test(dev_data)}")
    # print(f"Test:  {Quad.test(test_data)}")

    # print(f"\nPenta-Gram Perplexity:")
    # Penta.train(train_data)
    # print(f"Train: {Penta.test(train_data)}")
    # print(f"Dev:   {Penta.test(dev_data)}")
    # print(f"Test:  {Penta.test(test_data)}")

    # print(f"\nFifty-Gram Perplexity:")
    # Fifty.train(train_data)
    # print(f"Train: {Fifty.test(train_data)}")
    # print(f"Dev:   {Fifty.test(dev_data)}")
    # print(f"Test:  {Fifty.test(test_data)}")



    print(f"\n\n    -----   Showcase With Linear Interpolation  -----   ")
    print(f"\nUni-Gram Perplexity::")
    Uni.train(train_data, interpolation=True, lambdas=[1.0])
    print(f"Train: {Uni.test(train_data)}")
    print(f"Dev:   {Uni.test(dev_data)}")
    print(f"Test:  {Uni.test(test_data)}")

    print(f"\nBi-Gram Perplexity::")
    Bi.train(train_data, interpolation=True, lambdas=[0.3, 0.7])
    print(f"Train: {Bi.test(train_data)}")
    print(f"Dev:   {Bi.test(dev_data)}")
    print(f"Test:  {Bi.test(test_data)}")

    print(f"\nTri-Gram Perplexity::")
    Tri.train(train_data, interpolation=True, lambdas=[0.1, 0.3, 0.6])
    print(f"Train: {Tri.test(train_data)}")
    print(f"Dev:   {Tri.test(dev_data)}")
    print(f"Test:  {Tri.test(test_data)}")

    # print(f"\nQuad-Gram Perplexity::")
    # Quad.train(train_data, interpolation=True, lambdas=[0.2, 0.2, 0.3, 0.3])
    # print(f"Train: {Quad.test(train_data)}")
    # print(f"Dev:   {Quad.test(dev_data)}")
    # print(f"Test:  {Quad.test(test_data)}")

    # print(f"\nPenta-Gram Perplexity::")
    # Penta.train(train_data, interpolation=True, lambdas=[0.1, 0.2, 0.2, 0.2, 0.3])
    # print(f"Train: {Penta.test(train_data)}")
    # print(f"Dev:   {Penta.test(dev_data)}")
    # print(f"Test:  {Penta.test(test_data)}")

    # print(f"\Fifty-Gram Perplexity::")
    # Fifty.train(train_data, interpolation=True, lambdas=[0.1, 0.2, 0.2, 0.2, 0.3])
    # print(f"Train: {Fifty.test(train_data)}")
    # print(f"Dev:   {Fifty.test(dev_data)}")
    # print(f"Test:  {Fifty.test(test_data)}")



if __name__ == "__main__":
    main()
