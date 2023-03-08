#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

filenames = ["recorded_alignment_1d.txt", "recorded_alignment_2d.txt"]


if __name__ == "__main__":
    import pandas as pd
    plt.figure()
    for idx, f in enumerate(filenames):
        df = pd.read_csv(f)
        vals = df.values * 100
        trimmed = np.trim_zeros(vals, trim="f")[:500]
        plt.plot(np.arange(500) / 30, trimmed)
        print(np.mean(abs(trimmed)))
    plt.legend(["no filter", "filtered"])
    plt.title("Displacement dimension filtering")
    plt.xlabel("Traveled distance [m]")
    plt.ylabel("Estimated displacement [\%]")
    plt.tight_layout()
    plt.savefig("action_commands.pdf")
    plt.show()
