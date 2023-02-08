import numpy as np
import Image
import Perceptron
import NaiveBayes
import KNN
from random import sample
import math
import time
import timeit
import matplotlib.pyplot as plt
import main

face_train_data = main.read_data("facedata/facedatatrain", 70)
face_train_labels = main.read_labels("facedata/facedatatrainlabels")
face_map_data_labels_train = main.extract_features(face_train_data, face_train_labels)

face_test_data = main.read_data("facedata/facedatatest", 70)
face_test_labels = main.read_labels("facedata/facedatatestlabels")
face_map_data_labels_test = main.extract_features(face_test_data, face_test_labels)

face_map_data_labels_train_matrix = main.extract_features_matrix(face_train_data, face_train_labels)
face_map_data_labels_test_matrix = main.extract_features_matrix(face_test_data, face_test_labels)

num_faces = len(face_train_labels)
amount_testing_data_faces = []

count = 0.1
for i in range(10):
    amount_testing_data_faces.append(int(math.ceil(num_faces * count)))
    count += 0.1

face_naive_accuracy = []
face_naive_time = []
face_perceptron_accuracy = []
face_perceptron_time = []
face_knn_accuracy = []
face_knn_time = []

mean_accuracy_nb = []
mean_standard_deviation_nb = []
time_nb = []
pred_err_nb = []

for i in range(0, 10):
    for j in range(100):
        sampled_face_train_data = sample(face_map_data_labels_train, amount_testing_data_faces[i])
        face_naive_bayes, face_time_naivebayes = NaiveBayes.naivebayes_face(sampled_face_train_data,
                                                                            face_map_data_labels_test, 100)
        face_accuracy_naivebayes = main.get_accuracy(face_naive_bayes, face_test_labels)
        face_naive_accuracy.append(round(face_accuracy_naivebayes, 3))
        err = main.prediction_error_face(face_naive_bayes, face_test_labels)

    avg = main.mean(face_naive_accuracy)
    std = main.stand_dev(face_naive_accuracy)
    pred_err_nb.append(round(err, 3))
    mean_accuracy_nb.append(round(avg, 3))
    mean_standard_deviation_nb.append(round(std, 3))
    time_nb.append(round(face_time_naivebayes, 3))
print("MEAN ACCURACY FOR NAIVE BAYES:", mean_accuracy_nb)
print("MEAN STANDARD DEVIATION FOR NAIVE BAYES", mean_standard_deviation_nb)
print("TIME TAKEN FOR NAIVE BAYES TRAINING AND TESTING:", time_nb)
print("PREDICTION ERROR FOR NAIVE BAYES FACE:", pred_err_nb)
print("-----------------------------------------------------------------------------------------")
y_values5 = np.array(mean_accuracy_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values5 = np.array(y)
plt.title('GRAPH 1: ACCURACY OF NAIVE BAYES FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Accuracy')
plt.plot(x_values5, y_values5, linestyle=':', linewidth=2, color='green')
plt.show()
y_values6 = np.array(mean_standard_deviation_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values6 = np.array(y)
plt.title('GRAPH 2: STANDARD DEVIATION OF NAIVE BAYES FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Standard Deviation')
plt.plot(x_values6, y_values6, linestyle=':', linewidth=2, color='green')
plt.show()
y_values7 = np.array(time_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values7 = np.array(y)
plt.title('GRAPH 3: TIME TAKEN BY NAIVE BAYES FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Time Taken in sec')
plt.plot(x_values7, y_values7, linestyle=':', linewidth=2, color='green')
plt.show()
y_values10 = np.array(pred_err_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values10 = np.array(y)
plt.title('GRAPH 4: PREDICTION ERROR FOR NAIVE BAYES FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Prediction Error')
plt.plot(x_values10, y_values10, linestyle=':', linewidth=2, color='green')
plt.show()
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
mean_accuracy_p = []
mean_standard_deviation_p = []
time_p = []
pred_err_p = []
for i in range(0, 10):
    for j in range(100):
        sampled_face_train_data = sample(face_map_data_labels_train, amount_testing_data_faces[i])
        face_perceptron, face_time_perceptron = Perceptron.perceptron_face(sampled_face_train_data,
                                                                           face_map_data_labels_test, 100)
        face_accuracy_perceptron = main.get_accuracy(face_perceptron, face_test_labels)
        face_perceptron_accuracy.append(round(face_accuracy_perceptron, 3))
        err = main.prediction_error_face(face_perceptron, face_test_labels)
    avg = main.mean(face_perceptron_accuracy)
    std = main.stand_dev(face_perceptron_accuracy)
    mean_accuracy_p.append(round(avg, 3))
    mean_standard_deviation_p.append(round(std, 3))
    time_p.append(round(face_time_perceptron, 3))
    pred_err_p.append(round(err, 3))
print("MEAN ACCURACY FOR PERCEPTRON:", mean_accuracy_p)
print("MEAN STANDARD DEVIATION FOR PERCEPTRON", mean_standard_deviation_p)
print("TIME TAKEN FOR PERCEPTRON TRAINING AND TESTING:", time_p)
print("PREDICTION ERROR FOR PERCEPTRON FACE:", pred_err_p)
print("-----------------------------------------------------------------------------------------")
y_values1 = np.array(mean_accuracy_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values1 = np.array(y)
plt.title('GRAPH 5: ACCURACY OF PERCEPTRON FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Accuracy')
plt.plot(x_values1, y_values1, linestyle=':', linewidth=2, color='red')
plt.show()
y_values2 = np.array(mean_standard_deviation_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values2 = np.array(y)
plt.title('GRAPH 6: STANDARD DEVIATION OF PERCEPTRON FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Standard Deviation')
plt.plot(x_values2, y_values2, linestyle=':', linewidth=2, color='red')
plt.show()
y_values8 = np.array(time_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values8 = np.array(y)
plt.title('GRAPH 7: TIME TAKEN BY PERCEPTRON FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Time Taken in sec')
plt.plot(x_values8, y_values8, linestyle=':', linewidth=2, color='red')
plt.show()
y_values11 = np.array(pred_err_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values11 = np.array(y)
plt.title('GRAPH 8: PREDICTION ERROR FOR PERCEPTRON FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Prediction Error')
plt.plot(x_values11, y_values11, linestyle=':', linewidth=2, color='red')
plt.show()
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

mean_accuracy_knn = []
mean_standard_deviation_knn = []
time_knn = []
pred_err_knn = []
for i in range(0, 10):
    for j in range(100):
        sampled_knn_train_data = sample(face_map_data_labels_train_matrix, amount_testing_data_faces[i])
        face_knn, face_time_knn = KNN.nearest_neighbor(sampled_knn_train_data, face_map_data_labels_test_matrix)
        face_accuracy_knn = main.get_accuracy(face_knn, face_test_labels)
        face_knn_accuracy.append(round(face_accuracy_knn, 3))
        err = main.prediction_error_face(face_knn, face_test_labels)
    avg = main.mean(face_knn_accuracy)
    std = main.stand_dev(face_knn_accuracy)
    mean_accuracy_knn.append(round(avg, 3))
    mean_standard_deviation_knn.append(round(std, 3))
    time_knn.append(round(face_time_knn, 3))
    pred_err_knn.append(round(err, 3))
print("MEAN ACCURACY FOR KNN:", mean_accuracy_knn)
print("MEAN STANDARD DEVIATION FOR KNN", mean_standard_deviation_knn)
print("TIME TAKEN FOR KNN TRAINING AND TESTING:", time_knn)
print("PREDICTION ERROR FOR KNN FACE:", pred_err_knn)
print("-----------------------------------------------------------------------------------------")
y_values3 = np.array(mean_accuracy_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values3 = np.array(y)
plt.title('GRAPH 9: ACCURACY OF KNN FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Accuracy')
plt.plot(x_values3, y_values3, linestyle=':', linewidth=2, color='blue')
plt.show()
y_values4 = np.array(mean_standard_deviation_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values4 = np.array(y)
plt.title('GRAPH 10: STANDARD DEVIATION OF KNN FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Standard Deviation')
plt.plot(x_values4, y_values4, linestyle=':', linewidth=2, color='blue')
plt.show()
y_values9 = np.array(time_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values9 = np.array(y)
plt.title('GRAPH 11: TIME TAKEN BY KNN FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Time Taken in sec')
plt.plot(x_values9, y_values9, linestyle=':', linewidth=2, color='blue')
plt.show()
y_values12 = np.array(pred_err_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values12 = np.array(y)
plt.title('GRAPH 12: PREDICTION ERROR FOR KNN FOR FACE')
plt.xlabel('Sample Size in %')
plt.ylabel('Prediction Error')
plt.plot(x_values12, y_values12, linestyle=':', linewidth=2, color='blue')
plt.show()
# -----------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
