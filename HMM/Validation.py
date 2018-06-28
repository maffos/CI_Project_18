from Main import HMM
import numpy as np
from sklearn import metrics


# A kfold cross validation method for the HMM.
def kfold_cross_validation(k, pos_path, neg_path):

    # Positive entries in the training set.
    pos_len = 175
    # Negative entries in the training set.
    neg_len = 551
    pos_data = []
    neg_data = []

    with open(pos_path, "r") as f:
        for line in f:
            pos_data.append(line.split()[0])

    with open(neg_path, "r") as f:
        for line in f:
            neg_data.append(line.split()[0])
    scores = np.zeros(k)

    # Partition the training data and write the partitions to txt files.
    for i in range(k):
        with open("Cross_Validation/data_positive.txt", "w") as pos_file:
            for j in range(int(i * pos_len / k)):
                pos_file.write(pos_data[j] + "\n")

        with open("Cross_Validation/test_data.txt", "w") as test_file:
            y_true = []
            for j in range(int(i * pos_len / k), int((i + 1) * pos_len / k)):
                test_file.write(pos_data[j] + "\n")
                y_true.append(1)

        with open("Cross_Validation/data_positive.txt", "a") as pos_file:
            for j in range(int((i + 1) * pos_len / k), pos_len):
                pos_file.write(pos_data[j] + "\n")

        with open("Cross_Validation/data_negative.txt", "w") as neg_file:
            for j in range(int(i * neg_len / k)):
                neg_file.write(neg_data[j] + "\n")

        with open("Cross_Validation/test_data.txt", "a") as test_file:
            for j in range(int(i * neg_len / k), int((i + 1) * neg_len / k)):
                test_file.write(neg_data[j] + "\n")
                y_true.append(0)

        with open("Cross_Validation/data_negative.txt", "a") as neg_file:
            for j in range(int((i + 1) * neg_len / k), neg_len):
                neg_file.write(neg_data[j] + "\n")

        # Train an HMM on the training data and get the prediction on the
        # test data.
        hmm = HMM("Cross_Validation/data_positive.txt",
                  "Cross_Validation/data_negative.txt",
                  "Cross_Validation/test_data.txt",
                  "Cross_Validation/prediction.txt")
        hmm.train()
        y_prediction = hmm.predict()
        score = metrics.roc_auc_score(y_true, y_prediction)
        scores[i] = score

    print(np.mean(scores))
    print(scores)
    return np.mean(scores)


if __name__ == "__main__":

    kfold_cross_validation(10, "data_positive.txt", "data_negative.txt")
