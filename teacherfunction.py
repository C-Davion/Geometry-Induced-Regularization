'''
Use to generate the teacher function (true Y), of the form max(min Ax, A1x, min Bx, B1x) after x has been augmented by rho.
Samples values of A, A1, B, B1 at random, then purges those that have either
-small amount of pieces in the X range
-small total variation
Then the user hand select the final few.
Parameters A,A1, B,B1 are saved in LOG_File
'''

import numpy as np
import matplotlib.pyplot as plt
import json

LOG_FILE = "interesting_functions.jsonl"


def rho(x):
    x = np.asarray(x)
    return np.vstack((np.ones_like(x), x, x**2))


def f(A, A1, B, B1, x):
    X = rho(x)

    Ax = A @ X
    A1x = A1 @ X
    Bx = B @ X
    B1x = B1 @ X

    mA = np.minimum(Ax, A1x)
    mB = np.minimum(Bx, B1x)

    y = np.maximum(mA, mB)

    return y, Ax, A1x, Bx, B1x


if __name__ == "__main__":
    # Example coefficients
    # A = np.array([1, 2, -1])
    # A1 = np.array([0, 1, -1])
    # B = np.array([0.5, -1, 0.5])
    # B1 = np.array([1, 3, -0.5])

    x = np.linspace(-2, 2, 800)

    liked = []

    for trial in range(1000):

        A  = np.random.uniform(-3, 3, 3)
        A1 = np.random.uniform(-3, 3, 3)
        B  = np.random.uniform(-3, 3, 3)
        B1 = np.random.uniform(-3, 3, 3)

        y, Ax, A1x, Bx, B1x = f(A, A1, B, B1, x)

        use_A = np.minimum(Ax, A1x) >= np.minimum(Bx, B1x)

        labels = np.empty(len(x), dtype=int)
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
        dy = np.gradient(y, x)

        kink_strength = 0
        for i in switches:
            if i < 3 or i > len(x)-4:
                continue
            kink_strength += abs(dy[i-1] - dy[i+1])

        score = num_segments + 0.25 * kink_strength

        if score < 12:
            continue

        plt.figure(figsize=(7,4))

        plt.plot(x, np.where(labels==0, y, np.nan), lw=3, label="A")
        plt.plot(x, np.where(labels==1, y, np.nan), lw=3, label="A1")
        plt.plot(x, np.where(labels==2, y, np.nan), lw=3, label="B")
        plt.plot(x, np.where(labels==3, y, np.nan), lw=3, label="B1")

        plt.title(f"score={score:.2f}")
        plt.legend()
        plt.show()

        ans = input("keep? [y/n/q] ")

        if ans == "y":
            record = {
                "score": float(score),
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