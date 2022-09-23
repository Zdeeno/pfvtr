import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


tab = pd.read_csv("./output_dist.csv")
b2 = tab.iloc[:, 3].to_numpy()
p1 = tab.iloc[:, 7].to_numpy()
p2 = tab.iloc[:, 11].to_numpy()

b2 = b2[~np.isnan(b2)]
p1 = p1[~np.isnan(p1)]
p2 = p2[~np.isnan(p2)]


turn_b2 = b2[561:956] * 2.0
turn_p1 = p1[561:956] * 2.0
turn_p2 = p2[561:956] * 2.0


line_b2 = b2[956:1195] * 2.0
line_p1 = p1[956:1195] * 2.0
line_p2 = p2[956:1195] * 2.0


print(np.var(turn_b2), np.var(turn_p1), np.var(turn_p2))
print(np.var(line_b2), np.var(line_p1), np.var(line_p2))

time = np.arange(np.size(turn_b2))
time = time * (14/320.0)
time2 = np.arange(np.size(line_b2))
time2 = time2 * (14/320.0)


plt.plot(time, turn_b2, "y")
plt.plot(time, turn_p1, "g")
plt.plot(time, turn_p2, "b")
plt.legend(["classic", "1D filter", "2D filter"])
plt.title("Estimated correction - turn")
plt.xlabel("Distance [m]")
plt.ylabel("Correction [rad/s]")
plt.show()
plt.close()


plt.plot(time2, line_b2, "y")
plt.plot(time2, line_p1, "g")
plt.plot(time2, line_p2, "b")
plt.legend(["classic", "1D filter", "2D filter"])
plt.title("Estimated correction - straight line")
plt.xlabel("Distance [m]")
plt.ylabel("Correction [rad/s]")
plt.show()
