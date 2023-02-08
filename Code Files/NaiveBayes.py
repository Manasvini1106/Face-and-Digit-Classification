import Image
import numpy as np
import time
import timeit
import PercentageTracker


def naivebayes_face(train_data, test_data, size):
    start = timeit.default_timer()
    count = 0
    faces_count = 0
    not_faces_count = 0
    number_of_features = 451  # ----------> n

    for image in train_data:
        if image.class_label == '0':
            not_faces_count += 1  # -------->  Number of times y_i is false in X train
        elif image.class_label == '1':
            faces_count += 1  # -------->  Number of times y_i is true in X train
        for class_feature in image.class_features:
            if class_feature > count:
                count = class_feature

    faces = np.empty([size, count + 1])
    not_faces = np.empty([size, count + 1])
    not_faces.fill(0.01)
    faces.fill(0.01)

    for image in train_data:
        for i in range(size):
            for j in range(count + 1):
                if image.class_label == '0':
                    if image.class_features[i] == j:
                        not_faces[i][j] += 1

                elif image.class_label == '1':
                    if image.class_features[i] == j:
                        faces[i][j] += 1
    prob_y_true = faces_count / 451  # --------> P(y = true)
    prob_y_false = not_faces_count / 451  # -------> P(y = false)

    for i in range(size):
        for j in range(count + 1):
            not_faces[i][j] = (not_faces[i][j] / not_faces_count)  # -------> P(phi_j(x) | y = false)
            faces[i][j] = (faces[i][j] / faces_count)  # -------> P(phi_j(x) | y = true)

    prediction = []

    for image in test_data:
        count_faces = 1
        count_not_faces = 1
        index = 0
        for class_feature in image.class_features:
            if class_feature <= count:
                count_faces = (count_faces * faces[index][class_feature])
                count_not_faces = (count_not_faces * not_faces[index][class_feature])
            else:
                count_faces = count_faces * 0.01
                count_not_faces = count_not_faces * 0.01
            index += 1
        if count_faces >= count_not_faces:
            prediction.append('1')
        if count_faces < count_not_faces:
            prediction.append('0')
    stop = timeit.default_timer()
    time_taken = stop - start
    return prediction, time_taken


def naivebayes_digit(digit_image_data_training, digit_image_data_testing, feature_size):
    start = timeit.default_timer()
    count = 0
    digit_data_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 to 9
    for image in digit_image_data_training:
        if image.class_label == '0':  # -------->  Number of times y_i is 0 in X train
            digit_data_total[0] += 1

        elif image.class_label == '1':  # -------->  Number of times y_i is 1 in X train
            digit_data_total[1] += 1

        elif image.class_label == '2':  # -------->  Number of times y_i is 2 in X train
            digit_data_total[2] += 1

        elif image.class_label == '3':  # -------->  Number of times y_i is 3 in X train
            digit_data_total[3] += 1

        elif image.class_label == '4':  # -------->  Number of times y_i is 4 in X train
            digit_data_total[4] += 1

        elif image.class_label == '5':  # -------->  Number of times y_i is 5 in X train
            digit_data_total[5] += 1

        elif image.class_label == '6':  # -------->  Number of times y_i is 6 in X train
            digit_data_total[6] += 1

        elif image.class_label == '7':  # -------->  Number of times y_i is 7 in X train
            digit_data_total[7] += 1

        elif image.class_label == '8':  # -------->  Number of times y_i is 8 in X train
            digit_data_total[8] += 1

        elif image.class_label == '9':  # -------->  Number of times y_i is 9 in X train
            digit_data_total[9] += 1

        for class_feature in image.class_features:
            if class_feature > count:
                count = class_feature

    bayes_matrix_zero = np.empty([feature_size, count + 1])
    bayes_matrix_one = np.empty([feature_size, count + 1])
    bayes_matrix_two = np.empty([feature_size, count + 1])
    bayes_matrix_three = np.empty([feature_size, count + 1])
    bayes_matrix_four = np.empty([feature_size, count + 1])
    bayes_matrix_five = np.empty([feature_size, count + 1])
    bayes_matrix_six = np.empty([feature_size, count + 1])
    bayes_matrix_seven = np.empty([feature_size, count + 1])
    bayes_matrix_eight = np.empty([feature_size, count + 1])
    bayes_matrix_nine = np.empty([feature_size, count + 1])
    bayes_matrix_zero.fill(0.001)
    bayes_matrix_one.fill(0.001)
    bayes_matrix_two.fill(0.001)
    bayes_matrix_three.fill(0.001)
    bayes_matrix_four.fill(0.001)
    bayes_matrix_five.fill(0.001)
    bayes_matrix_six.fill(0.001)
    bayes_matrix_seven.fill(0.001)
    bayes_matrix_eight.fill(0.001)
    bayes_matrix_nine.fill(0.001)
    prob_y = []
    for i in range(len(digit_data_total)):
        p = digit_data_total[i] / len(digit_data_total)  # --------> P(y = 1,2,3,4,5,6,7,8,9) iterates one by one
        prob_y.append(p)

    for image in digit_image_data_training:
        for i in range(feature_size):
            for j in range(count + 1):
                if image.class_label == '0':
                    if image.class_features[i] == j:
                        bayes_matrix_zero[i][j] += 1

                elif image.class_label == '1':
                    if image.class_features[i] == j:
                        bayes_matrix_one[i][j] += 1

                elif image.class_label == '2':
                    if image.class_features[i] == j:
                        bayes_matrix_two[i][j] += 1

                elif image.class_label == '3':
                    if image.class_features[i] == j:
                        bayes_matrix_three[i][j] += 1

                elif image.class_label == '4':
                    if image.class_features[i] == j:
                        bayes_matrix_four[i][j] += 1

                elif image.class_label == '5':
                    if image.class_features[i] == j:
                        bayes_matrix_five[i][j] += 1

                elif image.class_label == '6':
                    if image.class_features[i] == j:
                        bayes_matrix_six[i][j] += 1

                elif image.class_label == '7':
                    if image.class_features[i] == j:
                        bayes_matrix_seven[i][j] += 1

                elif image.class_label == '8':
                    if image.class_features[i] == j:
                        bayes_matrix_eight[i][j] += 1

                elif image.class_label == '9':
                    if image.class_features[i] == j:
                        bayes_matrix_nine[i][j] += 1

    for i in range(feature_size):
        for j in range(count + 1):
            bayes_matrix_zero[i][j] = (bayes_matrix_zero[i][j] / digit_data_total[0])  # -------> P(phi_j(x) | y = 0)
            bayes_matrix_one[i][j] = (bayes_matrix_one[i][j] / digit_data_total[1])  # -------> P(phi_j(x) | y = 1)
            bayes_matrix_two[i][j] = (bayes_matrix_two[i][j] / digit_data_total[2])  # -------> P(phi_j(x) | y = 2)
            bayes_matrix_three[i][j] = (bayes_matrix_three[i][j] / digit_data_total[3])  # -------> P(phi_j(x) | y = 3)
            bayes_matrix_four[i][j] = (bayes_matrix_four[i][j] / digit_data_total[4])  # -------> P(phi_j(x) | y = 4)
            bayes_matrix_five[i][j] = (bayes_matrix_five[i][j] / digit_data_total[5])  # -------> P(phi_j(x) | y = 5)
            bayes_matrix_six[i][j] = (bayes_matrix_six[i][j] / digit_data_total[6])  # -------> P(phi_j(x) | y = 6)
            bayes_matrix_seven[i][j] = (bayes_matrix_seven[i][j] / digit_data_total[7])  # -------> P(phi_j(x) | y = 7)
            bayes_matrix_eight[i][j] = (bayes_matrix_eight[i][j] / digit_data_total[8])  # -------> P(phi_j(x) | y = 8)
            bayes_matrix_nine[i][j] = (bayes_matrix_nine[i][j] / digit_data_total[9])  # -------> P(phi_j(x) | y = 9 )

    prediction = []

    for image in digit_image_data_testing:
        num_tally = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        index = 0
        percent_array = []
        for class_feature in image.class_features:
            num_tally[0] = (num_tally[0] * bayes_matrix_zero[index][class_feature])
            num_tally[1] = (num_tally[1] * bayes_matrix_one[index][class_feature])
            num_tally[2] = (num_tally[2] * bayes_matrix_two[index][class_feature])
            num_tally[3] = (num_tally[3] * bayes_matrix_three[index][class_feature])
            num_tally[4] = (num_tally[4] * bayes_matrix_four[index][class_feature])
            num_tally[5] = (num_tally[5] * bayes_matrix_five[index][class_feature])
            num_tally[6] = (num_tally[6] * bayes_matrix_six[index][class_feature])
            num_tally[7] = (num_tally[7] * bayes_matrix_seven[index][class_feature])
            num_tally[8] = (num_tally[8] * bayes_matrix_eight[index][class_feature])
            num_tally[9] = (num_tally[9] * bayes_matrix_nine[index][class_feature])
            index += 1

        for i in range(0, 10):
            percent_array.append(PercentageTracker.PercentageTracker(str(i), num_tally[i]))

        percent_array.sort(key=lambda x: x.class_percent, reverse=True)
        prediction.append(percent_array[0].class_label)
    stop = timeit.default_timer()
    time_taken = stop - start
    return prediction, time_taken
