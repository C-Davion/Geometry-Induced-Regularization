'''
Use to generate the teacher function (true Y), of the form max(min Ax, A1x, min Bx, B1x) after x has been augmented by rho.
Samples values of A, A1, B, B1 at random, then purges those that have either
-small amount of pieces in the X range
-small total variation
Then the user hand select the final few.
Parameters A,A1, B,B1 are saved in LOG_File
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from relu_network import LinearFeatureAttention
import sys
#LOG_FILE = "interesting_functions.jsonl" #to rho(x)=(1,x,x^2)
LOG_FILE="intresting_function_attention.jsonl"


# def rho(x):
#     x = np.asarray(x)
#     return np.vstack((np.ones_like(x), x, x**2))



def f(A, A1, B, B1, X):
    Ax = X@ A
    A1x = X@ A1
    Bx = X@B 
    B1x = X@B1 

    mA = torch.minimum(Ax, A1x)
    mB = torch.minimum(Bx, B1x)

    y = torch.maximum(mA, mB)

    return y, Ax, A1x, Bx, B1x



if __name__ == "__main__":
    # Example coefficients
    # A = np.array([1, 2, -1])
    # A1 = np.array([0, 1, -1])
    # B = np.array([0.5, -1, 0.5])
    # B1 = np.array([1, 3, -0.5])

    x_poly = torch.linspace(-3, 3, 800)
    x_attention=torch.linspace(-1,1,800)
    x_input=x_attention.reshape(-1,1) #change to x_polu if Polynomial feature map was used
    rho=LinearFeatureAttention(8,16) #change to Polynomial feature map to reproduce the one used in intresting_functions, may need to invert the order of multiplication in f: X@A-> A@X
    output_dim=rho.output_dim
    A = torch.empty(output_dim).uniform_(-3, 3)
    A1 = torch.empty(output_dim).uniform_(-3, 3)
    B = torch.empty(output_dim).uniform_(-3, 3)
    B1 = torch.empty(output_dim).uniform_(-3, 3)
    X = rho(x_input)
    X=torch.reshape(X,(X.shape[0],-1))
    y, Ax, A1x, Bx, B1x =f(A, A1, B, B1, X)
    plt.plot(x_attention.detach().numpy(),y.detach().numpy(),)

    plt.xlabel("x")
    plt.ylabel("f(rho(x))")
    plt.show()  
    liked=[]
    for trial in range(1000):
        rho=LinearFeatureAttention(8,16) #change to Polynomial feature map to reproduce the one used in intresting_functions, may need to invert the order of multiplication in f: X@A-> A@X
        output_dim=rho.output_dim
        X = rho(x_input)
        X=torch.reshape(X,(X.shape[0],-1))
        A = torch.empty(output_dim).uniform_(-3, 3)
        A1 = torch.empty(output_dim).uniform_(-3, 3)
        B = torch.empty(output_dim).uniform_(-3, 3)
        B1 = torch.empty(output_dim).uniform_(-3, 3)
        
        y, Ax, A1x, Bx, B1x = f(A, A1, B, B1, X)
        x_attention.detach().numpy() #for plotting
        y=y.detach().numpy()
        Ax=Ax.detach().numpy()
        A1x=A1x.detach().numpy()
        Bx=Bx.detach().numpy()
        B1x=B1x.detach().numpy()

        use_A = np.minimum(Ax, A1x) >= np.minimum(Bx, B1x)

        labels = np.empty(len(x_attention), dtype=int)
        labels[use_A & (Ax <= A1x)] = 0
        labels[use_A & (Ax > A1x)] = 1
        labels[~use_A & (Bx <= B1x)] = 2
        labels[~use_A & (Bx > B1x)] = 3

        switches = np.where(labels[:-1] != labels[1:])[0]

        num_segments = len(switches) + 1

        # ignore boring functions
        if num_segments < 6:
            continue

        # finite-difference derivative
        dy = np.gradient(y, x_attention)

        kink_strength = 0
        for i in switches:
            if i < 3 or i > len(x_attention)-4:
                continue
            kink_strength += abs(dy[i-1] - dy[i+1])

        score = num_segments + 0.25 * kink_strength

        if score < 30:
            continue

        plt.figure(figsize=(7,4))

        plt.plot(x_attention, np.where(labels==0, y, np.nan), lw=3, label="A")
        plt.plot(x_attention, np.where(labels==1, y, np.nan), lw=3, label="A1")
        plt.plot(x_attention, np.where(labels==2, y, np.nan), lw=3, label="B")
        plt.plot(x_attention, np.where(labels==3, y, np.nan), lw=3, label="B1")

        plt.title(f"score={score:.2f}")
        plt.legend()
        plt.show()

        ans = input("keep? [y/n/q] ")

        if ans == "y":
            record = {
                "score": float(score),
                # Parameters of LinearFeatureAttention
                "rho": rho.get_parameters(), #comment this line if not using the Linear Attention Feature map
                "A":  A.tolist(),
                "A1": A1.tolist(),
                "B":  B.tolist(),
                "B1": B1.tolist(),
            }

            liked.append(record)

            with open(LOG_FILE, "a") as file:
                file.write(json.dumps(record) + "\n")

            print("Saved.")

        if ans == "q":
            break