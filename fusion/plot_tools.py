import random
def _random_color(n):
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
    return r,g,b

def _rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def plotChart(classes, embeddings):
    x = []
    y = []
    for emb in embeddings:
        x.append(emb[0])
        y.append(emb[1])

    print("rows: {0}".format(len(x)))
    print("classes: {0}".format(len(y)))

    plt.figure(figsize=(13, 13)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i],c='red')
        plt.annotate(classes[i] + ' ' + str(i),
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.show()

def plotCluster(blocks, classes, num_clusters, embeddings):
    x_blocks = []
    y_blocks = []
    for emb in embeddings:
        x_blocks.append(emb[0])
        y_blocks.append(emb[1])
    plt.figure(figsize=(13, 13)) 
    print('num of classes: {0}'.format(num_clusters))
    for i in range(num_clusters):
        colorInit = 1 + i
        r, g, b = random_color(colorInit)
        selected_color = rgb2hex(r,g,b)
        block = blocks[i]
        for e in block:
            plt.scatter(x[e],y[e], c=selected_color)
            plt.annotate(classes[e] + ' ' + str(e),
                         xy=(x[e], y[e]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.show()