import matplotlib.pyplot as plt

steps = []
rewards = []

with open("log.txt", "r") as open_file:
    lines = list(open_file.readlines())
    for i, l in enumerate(lines):
        if "reward" in l:
            step = int(lines[i + 1].split()[3].replace(",", ""))
            reward = float(l.split()[3])
            steps.append(step)
            rewards.append(reward)

plt.plot(steps, rewards)
plt.show()
