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

digit_train_data = main.read_data("digitdata/trainingimages", 28)
digit_train_labels = main.read_labels("digitdata/traininglabels")
digit_map_data_labels_train = main.extract_features(digit_train_data, digit_train_labels)

digit_test_data = main.read_data("digitdata/testimages", 28)
digit_test_labels = main.read_labels("digitdata/testlabels")
digit_map_data_labels_test = main.extract_features(digit_test_data, digit_test_labels)

digit_map_data_labels_train_matrix = main.extract_features_matrix(digit_train_data, digit_train_labels)
digit_map_data_labels_test_matrix = main.extract_features_matrix(digit_test_data, digit_test_labels)

num_digits = len(digit_train_labels)
amount_testing_data_digits = []

count = 0.1
for i in range(10):
    amount_testing_data_digits.append(int(math.ceil(num_digits * count)))
    count += 0.1

digit_naive_accuracy = []
digit_naive_time = []
digit_perceptron_accuracy = []
digit_perceptron_time = []
digit_knn_accuracy = []
digit_knn_time = []

mean_accuracy_digit_nb = []
mean_standard_deviation_digit_nb = []
time_digit_nb = []
pred_err_nb = []
for i in range(0, 10):
    for j in range(10):
        sampled_digit_train_data = sample(digit_map_data_labels_train, amount_testing_data_digits[i])
        digit_naive_bayes, digit_time_naivebayes = NaiveBayes.naivebayes_digit(sampled_digit_train_data,
                                                                               digit_map_data_labels_test, 100)
        digit_accuracy_naivebayes = main.get_accuracy(digit_naive_bayes, digit_test_labels)
        digit_naive_accuracy.append(round(digit_accuracy_naivebayes, 3))
        err = main.prediction_error_digit(digit_naive_bayes,digit_test_labels)
    avg = main.mean(digit_naive_accuracy)
    std = main.stand_dev(digit_naive_accuracy)
    mean_accuracy_digit_nb.append(round(avg, 3))
    mean_standard_deviation_digit_nb.append(round(std, 3))
    time_digit_nb.append(round(digit_time_naivebayes, 3))
    pred_err_nb.append(round(err,3))
print("MEAN ACCURACY FOR DIGIT NAIVE BAYES:", mean_accuracy_digit_nb)
print("MEAN STANDARD DEVIATION FOR DIGIT NAIVE BAYES", mean_standard_deviation_digit_nb)
print("TIME TAKEN FOR NAIVE BAYES DIGIT TRAINING AND TESTING:", time_digit_nb)
print("PREDICTION ERROR FOR NAIVE BAYES FOR DIGIT :", pred_err_nb)
print("-----------------------------------------------------------------------------------------")
y_values5 = np.array(mean_accuracy_digit_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values5 = np.array(y)
plt.title('GRAPH 1: ACCURACY OF NAIVE BAYES FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Accuracy')
plt.plot(x_values5, y_values5, linestyle=':', linewidth=2, color='green')
plt.show()
y_values6 = np.array(mean_standard_deviation_digit_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values6 = np.array(y)
plt.title('GRAPH 2: STANDARD DEVIATION OF NAIVE BAYES FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Standard Deviation')
plt.plot(x_values6, y_values6, linestyle=':', linewidth=2, color='green')
plt.show()
y_values7 = np.array(time_digit_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values7 = np.array(y)
plt.title('GRAPH 3: TIME TAKEN BY NAIVE BAYES FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Time Taken in sec')
plt.plot(x_values7, y_values7, linestyle=':', linewidth=2, color='green')
plt.show()
y_values8 = np.array(pred_err_nb)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values8 = np.array(y)
plt.title('GRAPH 4: PREDICTION ERROR FOR NAIVE BAYES FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Prediction Error')
plt.plot(x_values8, y_values8, linestyle=':', linewidth=2, color='green')
plt.show()
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
mean_accuracy_digit_p = []
mean_standard_deviation_digit_p = []
time_digit_p = []
pred_err_p = []
for i in range(0, 10):
    for j in range(10):
        sampled_digit_train_data = sample(digit_map_data_labels_train, amount_testing_data_digits[i])
        digit_perceptron, digit_time_perceptron = Perceptron.perceptron_digit(sampled_digit_train_data,
                                                                              digit_map_data_labels_test, 100)
        digit_accuracy_perceptron = main.get_accuracy(digit_perceptron, digit_test_labels)
        digit_perceptron_accuracy.append(round(digit_accuracy_perceptron, 3))
        err = main.prediction_error_digit(digit_perceptron,digit_test_labels)
    avg = main.mean(digit_perceptron_accuracy)
    std = main.stand_dev(digit_perceptron_accuracy)
    mean_accuracy_digit_p.append(round(avg, 3))
    mean_standard_deviation_digit_p.append(round(std, 3))
    time_digit_p.append(round(digit_time_perceptron, 3))
    pred_err_p.append(round(err,3))
print("MEAN ACCURACY FOR DIGIT PERCEPTRON:", mean_accuracy_digit_p)
print("MEAN STANDARD DEVIATION FOR DIGIT PERCEPTRON", mean_standard_deviation_digit_p)
print("TIME TAKEN FOR PERCEPTRON DIGIT TRAINING AND TESTING:", time_digit_p)
print("PREDICTION ERROR FOR PERCEPTRON FOR DIGIT :", pred_err_p)
print("-----------------------------------------------------------------------------------------")
y_values1 = np.array(mean_accuracy_digit_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values1 = np.array(y)
plt.title('GRAPH 5: ACCURACY OF PERCEPTRON FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Accuracy')
plt.plot(x_values1, y_values1, linestyle=':', linewidth=2, color='red')
plt.show()
y_values2 = np.array(mean_standard_deviation_digit_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values2 = np.array(y)
plt.title('GRAPH 6: STANDARD DEVIATION OF PERCEPTRON FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Standard Deviation')
plt.plot(x_values2, y_values2, linestyle=':', linewidth=2, color='red')
plt.show()
y_values8 = np.array(time_digit_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values8 = np.array(y)
plt.title('GRAPH 7: TIME TAKEN BY PERCEPTRON FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Time Taken in sec')
plt.plot(x_values8, y_values8, linestyle=':', linewidth=2, color='red')
plt.show()
y_values9 = np.array(pred_err_p)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values9 = np.array(y)
plt.title('GRAPH 8: PREDICTION ERROR FOR PERCEPTRON FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Prediction Error')
plt.plot(x_values9, y_values9, linestyle=':', linewidth=2, color='red')
plt.show()
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

mean_accuracy_digit_knn = []
mean_standard_deviation_digit_knn = []
time_digit_knn = []
pred_err_knn = []
'''for i in range(0, 10):
    for j in range(1):
        sampled_knn_train_data = sample(digit_map_data_labels_train_matrix, amount_testing_data_digits[i])
        digit_knn, digit_time_knn = KNN.nearest_neighbor(sampled_knn_train_data, digit_map_data_labels_test_matrix)
        digit_accuracy_knn = main.get_accuracy(digit_knn, digit_test_labels)
        digit_knn_accuracy.append(round(digit_accuracy_knn, 3))
        err = main.prediction_error_digit(digit_knn, digit_test_labels)
    avg = main.mean(digit_knn_accuracy)
    std = main.stand_dev(digit_knn_accuracy)
    mean_accuracy_digit_knn.append(round(avg, 3))
    mean_standard_deviation_digit_knn.append(round(std, 3))
    time_digit_knn.append(round(digit_time_knn, 3))
    pred_err_knn.append(round(err, 3))'''
for i in range(10):
    sampled_knn_train_data = sample(digit_map_data_labels_train_matrix, amount_testing_data_digits[i])
    digit_knn, digit_time_knn = KNN.nearest_neighbor(sampled_knn_train_data, digit_map_data_labels_test_matrix)
    digit_accuracy_knn = main.get_accuracy(digit_knn, digit_test_labels)
    digit_knn_accuracy.append(round(digit_accuracy_knn, 3))
    err = main.prediction_error_digit(digit_knn, digit_test_labels)
    avg = main.mean(digit_knn_accuracy)
    std = main.stand_dev(digit_knn_accuracy)
    mean_accuracy_digit_knn.append(round(avg, 3))
    mean_standard_deviation_digit_knn.append(round(std, 3))
    time_digit_knn.append(round(digit_time_knn, 3))
    pred_err_knn.append(round(err, 3))
print("MEAN ACCURACY FOR KNN FOR DIGIT:", mean_accuracy_digit_knn)
print("MEAN STANDARD DEVIATION FOR KNN FOR DIGIT", mean_standard_deviation_digit_knn)
print("TIME TAKEN FOR KNN FOR DIGIT TRAINING AND TESTING:", time_digit_knn)
print("PREDICTION ERROR FOR KNN FOR DIGIT", pred_err_knn)
print("-----------------------------------------------------------------------------------------")
y_values3 = np.array(mean_accuracy_digit_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values3 = np.array(y)
plt.title('GRAPH 9: ACCURACY OF KNN FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Accuracy')
plt.plot(x_values3, y_values3, linestyle=':', linewidth=2, color='blue')
plt.show()
y_values4 = np.array(mean_standard_deviation_digit_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values4 = np.array(y)
plt.title('GRAPH 10: STANDARD DEVIATION OF KNN FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Standard Deviation')
plt.plot(x_values4, y_values4, linestyle=':', linewidth=2, color='blue')
plt.show()
y_values9 = np.array(time_digit_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values9 = np.array(y)
plt.title('GRAPH 11: TIME TAKEN BY KNN FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Time Taken in sec')
plt.plot(x_values9, y_values9, linestyle=':', linewidth=2, color='blue')
plt.show()
y_values10 = np.array(pred_err_knn)
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x_values10 = np.array(y)
plt.title('GRAPH 12: PREDICTION ERROR FOR KNN FOR DIGIT')
plt.xlabel('Sample Size in %')
plt.ylabel('Prediction Error')
plt.plot(x_values10, y_values10, linestyle=':', linewidth=2, color='blue')
plt.show()
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
