from matplotlib import pyplot as plt
import numpy as np


def calculate_err(arr1, arr2):
    diff = np.mean(np.abs(arr1 - arr2))
    print(diff)


day_csv = np.loadtxt("./sim_day.csv", delimiter=",")
switch_csv = np.loadtxt("./sim_switching.csv", delimiter=",")
ts_csv = np.loadtxt("./sim_ts.csv", delimiter=",")
map_csv = np.loadtxt("./map_ts.csv", delimiter=",")


dists_day = day_csv[:, 2]
dists_switch = switch_csv[:, 2]
dists_ts = ts_csv
dists_map = map_csv[:-2]
dists_ts = np.interp(np.arange(np.size(dists_switch)), np.linspace(0, np.size(dists_switch), np.size(dists_ts)), dists_ts)
dists_map = np.interp(np.arange(np.size(dists_switch)), np.linspace(0, np.size(dists_switch), np.size(dists_map)), dists_map)

map_idxs = switch_csv[:, 1]
parts = [0, 308, 608, 907, 1206, 1500]
real_maps = np.zeros(1500)
real_maps[parts[1]:parts[2]] = 1
real_maps[parts[2]:parts[3]] = 2
real_maps[parts[3]:parts[4]] = 1

for i in range(len(parts) - 1):
    print("part", i)
    calculate_err(dists_day[parts[i]:parts[i+1]], dists_map[parts[i]:parts[i+1]])
    calculate_err(dists_switch[parts[i]:parts[i+1]], dists_map[parts[i]:parts[i+1]])
print("whole map")
calculate_err(dists_day, dists_map)
calculate_err(dists_switch, dists_map)

plt.plot(dists_day)
plt.plot(dists_switch)
# plt.plot(dists_ts)
plt.plot(dists_map, "--")
plt.legend(["day", "switching", "total_station"])
plt.grid()
plt.show()
plt.close()

plt.plot(map_idxs)
plt.plot(real_maps, "--")
plt.legend(["Estimated map", "Real map"])
plt.yticks([0, 1, 2], ["Day", "Evening", "Night"])
plt.grid()
plt.title("Map estimation")
plt.xlabel("Timestamp")
plt.show()
