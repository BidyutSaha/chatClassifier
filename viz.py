import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


x0 = int(sys.argv[2])
xn = int(sys.argv[3])
y0 = int(sys.argv[4])
yn = int(sys.argv[5])


def update(frame):
    global ax1, ax2
    global train_loss
    global val_loss
    global train_acc
    global val_acc

    data = np.genfromtxt(sys.argv[1], delimiter=",")
    #print(type(data), data.shape)
    xdata = data[:, 0].tolist()
    tl = data[:, 1].tolist()
    ta = data[:, 2].tolist()
    vl = data[:, 3].tolist()
    va = data[:, 4].tolist()

    #print(xdata, tl, ta, vl, va)
    # tl = np.random.randint(0, 1500, 10)
    # vl = np.random.randint(0, 1500, 10)
    # ta = np.random.randint(0, 1500, 10)
    # va = np.random.randint(0, 1500, 10)

    train_loss.set_data(xdata, tl)
    val_loss.set_data(xdata, vl)
    train_acc.set_data(xdata, ta)
    val_acc.set_data(xdata, va)

    return train_loss, val_loss, train_acc, val_acc


# Some example data to display
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Train VS Validation')

train_loss, = ax1.plot([], [], lw=2, label="Train", c="blue")
val_loss, = ax1.plot([], [], lw=2, label="Validation", c="green")
train_acc, = ax2.plot([], [], lw=2, label="Train", c="blue")
val_acc, = ax2.plot([], [], lw=2, label="Validation", c="green")

ax1.legend()
ax2.legend()


def init():
    global ax1, ax2
    global train_loss
    global val_loss
    global train_acc
    global val_acc

    ax1.grid(True, linestyle='dotted')
    ax1.set_title("Loss")
    ax1.set_xlim(x0, xn)
    ax1.set_ylim(y0, 1.5)

    ax2.grid(True, linestyle='dotted')
    ax2.set_title("Accuracy")
    ax2.set_xlim(x0, xn)
    ax2.set_ylim(y0, 1)

    return train_loss, val_loss, train_acc, val_acc


ani = FuncAnimation(fig, update,
                    init_func=init, interval=1000, blit=True, repeat=True)
plt.show()
