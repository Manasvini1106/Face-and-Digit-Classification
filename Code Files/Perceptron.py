import time
import timeit
import decimal
import random


def perceptron_face(face_train_data, face_test_data, feature_size):
    start = timeit.default_timer()
    w = []  # ------ w[0]....w[100], w[0] is the bias
    accuracy_percent = 0
    n_round = 0
    n_correct = 0
    for i in range(0, feature_size + 1):  # ----- assigning random weights
        w.append(decimal.Decimal(random.randrange(-50, 50)) / 100)
    while accuracy_percent < .85 and n_round < 1000:
        for image in face_train_data:
            fx = 0
            for j in range(1, feature_size + 1):  # 1 to 100
                fx += w[j] * image.class_features[j - 1]  # calculate f(x)
            if fx < 0 and image.class_label == '0':  # ------ "correct prediction"
                n_correct += 1
            elif fx >= 0 and image.class_label == '1':  # ------ "correct prediction"
                n_correct += 1
            elif fx < 0 and image.class_label == '1':  # ------ "wrong prediction"
                for i in range(1, feature_size + 1):
                    w[i] += image.class_features[i - 1]
                w[0] += 1
            elif fx >= 0 and image.class_label == '0':  # ------ "wrong prediction"
                for i in range(1, feature_size + 1):
                    w[i] -= image.class_features[i - 1]
                w[0] -= 1
            n_round += 1
        accuracy_percent = float(n_correct) / n_round

    # TESTING

    guess_array = []
    for image in face_test_data:
        fx = 0
        for j in range(1, feature_size + 1):  # 1 to 100
            fx += w[j] * image.class_features[j - 1]  # ----- calculate f(x)
        if fx < 0:
            guess_array.append('0')
        else:
            guess_array.append('1')
    stop = timeit.default_timer()
    time_taken = stop - start
    return guess_array, time_taken


def perceptron_digit(digit_train_data, digit_test_data, feature_size):
    start = timeit.default_timer()
    d_w = []  # ---  storing weights for each digit
    # digit_weights[1] contains the 100 weights for the digit 1 and so on..
    n_correct = 0
    n_round = 0

    for i in range(10):
        single_digit_weights = []
        for j in range(0, feature_size + 1):  # assigning random weights
            single_digit_weights.append(decimal.Decimal(random.randrange(-50, 50)) / 100)
        d_w.append(single_digit_weights)

    for image in digit_train_data:
        fxs = []
        for i in range(10):
            fx = 0
            for j in range(1, feature_size + 1):  # ----- 1 to 100 (for every feature)
                fx += d_w[i][j] * image.class_features[j - 1]  # ---- calculating f(x) for digit i
            fxs.append(fx)

        maximum_value = max(fxs)
        pred_digit = fxs.index(maximum_value)

        if pred_digit == int(image.class_label):  # ----- "correct prediction"
            n_correct += 1
        else:
            # "wrong prediction"
            for i in range(1, feature_size + 1):
                d_w[int(image.class_label)][i] += image.class_features[i - 1] # adding to the actual digit weight
            d_w[int(image.class_label)][0] += 1

            for i in range(1, feature_size + 1):
                d_w[pred_digit][i] -= image.class_features[i - 1] # subtracting from the predicted weight
            d_w[pred_digit][0] -= 1

        n_round += 1

    # TESTING

    guess_array = []
    for image in digit_test_data:
        fxs = []
        for i in range(10):
            fx = 0
            for j in range(1, feature_size + 1):  # ----- 1 to 100 (for every feauture)
                fx += d_w[i][j] * image.class_features[j - 1]  # ---- calculate f(x) for digit i
            fxs.append(fx)

        maximum_value = max(fxs)
        pred_digit = fxs.index(maximum_value)
        guess_array.append(str(pred_digit))
    stop = timeit.default_timer()
    time_taken = stop - start
    return guess_array, time_taken
