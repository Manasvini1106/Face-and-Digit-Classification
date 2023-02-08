#Authors: Anna Godin & Kimberly Wolak


import Image
import numpy as np
import NearestNeighborTracker
from collections import Counter

def nearest_neighbor(TestingData, trainingData):
    answer =[]
    for image in trainingData:
        distances = []
        for image2 in TestingData:
            distance = np.linalg.norm(image2.class_features-image.class_features)
            distances.append(NearestNeighborTracker.NearestNeighborTracker(image2.class_label,distance))
        distances.sort(key=lambda x: x.class_distance, reverse=False)

        vote_array = [0]*3
        vote_array[0] = distances[0].class_label
        vote_array[1] = distances[1].class_label
        vote_array[2] = distances[2].class_label

        result = most_frequent(vote_array)
        answer.append(result)

    return answer

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]



