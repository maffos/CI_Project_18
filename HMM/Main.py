from InitMatrices import InitMatrices
from Data import Data
from HMMConstruct import HMMConstruct
# from Validation import kfold_cross_validation


class HMM():

    def __init__(self, pos_path, neg_path, test_path, output_path):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.test_path = test_path
        self.output_path = output_path
        self.model_pos = None
        self.model_neg = None

    def train(self):

        init_data = Data(self.pos_path, self.neg_path)
        positive_pairs, negative_pairs = init_data.split_in_pairs()

        init_pos = InitMatrices(positive_pairs)
        trans_matrix_pos = init_pos.compute_transitions()
        emission_matrix_pos = init_pos.compute_emissions()

        init_neg = InitMatrices(negative_pairs)
        trans_matrix_neg = init_neg.compute_transitions()
        emission_matrix_neg = init_neg.compute_emissions()

        self.model_pos = HMMConstruct(trans_matrix_pos, emission_matrix_pos,
                                      self.pos_path)
        self.model_pos.build_eval_HMM()

        self.model_neg = HMMConstruct(trans_matrix_neg, emission_matrix_neg,
                                      self.neg_path)
        self.model_neg.build_eval_HMM()

    def predict(self):
        text_file = open(self.output_path, "w")
        text_file.write("Id,Prediction1\n")
        prediction = []
        # Extract only the sequences from our file.
        with open(self.test_path, 'r') as tf:
            data = [line[:9] for line in tf]

        # For each sequence in data calculate the negative log likelihood of it
        # being a binder (positive model) or a non-binder (negative model).
        for line in data:
            test_data = []
            # Split current sequence in single characters.
            for i in range(len(line)):
                test_data.append(line[i])

            # Viterbi-Algorithm calculates the negative log likelihood of each
            # sequence.
            neg_log_likelihood_1 = self.model_pos.log_probability(test_data)
            neg_log_likelihood_2 = self.model_neg.log_probability(test_data)

            # Decides which outcome is more likely and writes it to a file at
            # given path.
            if (neg_log_likelihood_1 >= neg_log_likelihood_2):
                text_file.write(line + ',1\n')
                prediction.append(1)
                # print(line + ',1')
            else:
                text_file.write(line + ',0\n')
                prediction.append(0)
                # print(line + ',0')

        text_file.close()
        print('Process successful.')
        return prediction


if __name__ == "__main__":
    pos_path = "data_positive.txt"
    neg_path = "data_negative.txt"
    test_path = "kaggle.txt"
    output_path = 'output.txt'
    hmm = HMM(pos_path, neg_path, test_path, output_path)
    hmm.train()
    prediction = hmm.predict()
