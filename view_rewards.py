import matplotlib.pyplot as plt
import glob
import numpy as np
from pandas import DataFrame
# from scipy.interpolate import spline

file_names = []
plots = []
for file_name in glob.glob("npy_outputs/*"):
    steps = []
    rewards = []
    # if not "sky" in file_name: continue
    # if not ("simple" in file_name or "skyline" in file_name): continue
    file_names.append(file_name)

    # <<<<<<< HEAD

    with open(file_name, "r") as open_file:
        lines = list(open_file.readlines())
        for i, l in enumerate(lines):
            if "reward" in l:
                step = int(lines[i + 1].split()[3].replace(",", ""))
                reward = float(l.split()[3])
                steps.append(step)
                rewards.append(reward)

    w = 50

    plots.append(
        plt.plot(steps[:1000], DataFrame(rewards[:1000]).rolling(w).mean().shift(1 - w))[0])

plt.legend(plots, file_names)
# =======
# with open("log.txt", "r") as open_file:
#     lines = list(open_file.readlines())
#     for i, l in enumerate(lines):
#         if "reward" in l:
#             step = int(lines[i + 1].split()[3].replace(",", ""))
#             reward = float(l.split()[3])
#             steps.append(step)
#             rewards.append(reward)

# plt.plot(steps, rewards)
# >>>>>>> 7559421d08365f774ad88bcf7e48cb85801a291d
plt.show()
