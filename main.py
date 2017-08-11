import math
import operator

import util

# Created by DimRu on 04-Aug-17


def euclidean_distance(instance1, instance2, length):
    """find the euclidean distance with a training set instance with a testing set instance"""
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbours(training_set, test_instance, k):
    """find euclidean distance from a test data set to training set instances and sort distances
    then return k number of neighbours as we defined"""
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    """Sort number of neighbours in each class and return the class which has maximum no of neighbours"""
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    """Check the accuracy of the testing data set"""
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    print('total no of inputs : ', len(test_set))
    print("Accurately predicted : ", correct)
    return (correct / float(len(test_set))) * 100.0


def main():
    # prepare data
    print("Split ratio should be a decimal value between 0 and 1")
    split = float(input("Enter split ratio for  the training and testing set : "))
    print("enter a positive odd integer value for K nearest neighbour : ")
    k = int(input("define K for the evaluation : "))
    training_set, test_set = util.file_reader(split)

    print('Train set: ' + repr(len(training_set)))
    print('Test set: ' + repr(len(test_set)))
    # generate predictions
    predictions = []
    for x in range(len(test_set)):
        neighbors = get_neighbours(training_set, test_set[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == '__main__':

    main()
