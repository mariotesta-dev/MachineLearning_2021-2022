import numpy
import matplotlib.pyplot as plt

def load(f):
    values = []
    labels = []

    dict_labels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    for line in f:
        line = line.rstrip().split(",")
        label = dict_labels[line[4:][0]]
        entry = numpy.array(line[0:4], dtype=numpy.float32).reshape(4,1)

        values.append(entry)
        labels.append(label)
    
    return numpy.hstack(values), numpy.array(labels, dtype=numpy.int32)

def plot_histogram(values, labels, folder):
    v0 = values[:, labels == 0]
    v1 = values[:, labels == 1]
    v2 = values[:, labels == 2]

    features = {
        0: 'Sepal Lenght',
        1: 'Sepal Width',
        2: 'Petal Lenght',
        3: 'Petal Width',
    }

    for index in range(4):
        plt.figure()
        plt.xlabel(features[index])
        plt.hist(v0[index, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
        plt.hist(v1[index, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(v2[index, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')

        plt.legend()
        plt.savefig(folder+'/histogram_%d.pdf' % index)
    plt.show()

def plot_scatter(values, labels, folder):
    v0 = values[:, labels == 0]
    v1 = values[:, labels == 1]
    v2 = values[:, labels == 2]

    features = {
        0: 'Sepal Lenght',
        1: 'Sepal Width',
        2: 'Petal Lenght',
        3: 'Petal Width',
    }

    for index_1 in range(4):
        for index_2 in range(4):
            if index_1 == index_2:
                continue
            else:
                plt.figure()
                plt.xlabel(features[index_1])
                plt.ylabel(features[index_2])
                plt.scatter(v0[index_1, :], v0[index_2, :], label = 'Setosa')
                plt.scatter(v1[index_1, :], v1[index_2, :], label = 'Versicolor')
                plt.scatter(v2[index_1, :], v2[index_2, :], label = 'Virginica')

                plt.legend()
                plt.savefig(folder+'/scatter_%d_%d.pdf' % (index_1, index_2))
        plt.show()

def compute_mean(values):
    return values.mean(1)


if __name__ == '__main__':
    f = open("iris.csv", "r")
    (values, labels) = load(f)

    #Visualize the dataset
    plot_histogram(values, labels, "results")
    plot_scatter(values, labels, "results")

    #Subtract mean from data to get centered values
    values_centered = values - values.mean(1).reshape((values.shape[0], 1))
    #Visualize the centered dataset
    plot_histogram(values_centered, labels, "centered_results")
    plot_scatter(values_centered, labels, "centered_results")









