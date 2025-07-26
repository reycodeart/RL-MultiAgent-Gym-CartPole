import matplotlib.pyplot as plt

def plot_rewards(rewards, title, filename):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(filename)
    plt.close()
