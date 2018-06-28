from pomegranate import *
from InitMatrices import InitMatrices
from Data import Data
from HMMConstruct import HMMConstruct


class HMM():

    if __name__ == '__main__':

        pos_path = 'D:\Dropbox\Working directory\#Material\data_positive.txt'
        neg_path = 'D:\Dropbox\Working directory\#Material\data_negative.txt'
        test_path = 'D:\Dropbox\Working directory\#Material\\kaggle.txt'
        output_path = 'D:\Dropbox\Working directory\#Material\output.txt'

        init_data = Data(pos_path, neg_path)
        positive_pairs, negative_pairs = init_data.split_in_pairs()

        init_pos = InitMatrices(positive_pairs)
        trans_matrix_pos = init_pos.compute_transitions()
        emission_matrix_pos = init_pos.compute_emissions()

        init_neg = InitMatrices(negative_pairs)
        trans_matrix_neg = init_neg.compute_transitions()
        emission_matrix_neg = init_neg.compute_emissions()

        model_pos = HMMConstruct(trans_matrix_pos, emission_matrix_pos,
                                 pos_path)
        model_pos.build_eval_HMM()

        model_neg = HMMConstruct(trans_matrix_neg, emission_matrix_neg,
                                 neg_path)
        model_neg.build_eval_HMM()

        text_file = open(output_path, "w")
        text_file.write("Id,Prediction1\n")

        # Extract only the sequences from our file.
        with open(test_path, 'r') as tf:
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
            neg_log_likelihood_1 = model_pos.log_probability(test_data)
            neg_log_likelihood_2 = model_neg.log_probability(test_data)

            # Decides which outcome is more likely and writes it to a file at
            # given path.
            if (neg_log_likelihood_1 >= neg_log_likelihood_2):
                text_file.write(line + ',1\n')
                # print(line + ',1')
            else:
                text_file.write(line + ',0\n')
                # print(line + ',0')

        text_file.close()
        print('Process successful.')
