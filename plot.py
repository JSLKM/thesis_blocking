import random


def random_RGB(n):
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    return r, g, b


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def plot_embeddings(tsne_embeddings, colors, chosen_labels):
    x = []
    y = []
    for emb in tsne_embeddings:
        x.append(emb[0])
        y.append(emb[1])

    print("rows: {0}".format(len(x)))
    print("classes: {0}".format(len(colors)))

    plt.figure(figsize=(13, 13))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=colors[chosen_labels[i]])
    # plt.annotate(i,
    #         xy=(x[i], y[i]),
    #         xytext=(5, 2),
    #         textcoords='offset points',
    #         ha='right',
    #         va='bottom')
    plt.show()
