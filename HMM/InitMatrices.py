import numpy as np
import itertools


class InitMatrices():

    # List of amino acids.
    amino_acids = ['A', 'R', 'N', 'D', 'C',
                   'Q', 'E', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P',
                   'S', 'T', 'W', 'Y', 'V']

    # Numpy array which groups amino acid pairs that share multiple common
    # physicochemical properties. The ten non empty groups each represent
    # a state of the HMM.
    multi_property_group = np.array([['', '', '', '', ''],
                                     ['L', 'I', 'V', '', ''],
                                     ['V', 'C', 'G', 'A', ''],
                                     ['A', 'S', 'T', 'G', 'C'],
                                     ['N', 'C', 'S', 'T', 'D'],
                                     ['E', 'D', 'R', 'K', 'H'],
                                     ['K', 'C', 'Y', 'W', 'H'],
                                     ['Y', 'F', 'W', 'H', ''],
                                     ['M', '', '', '', ''],
                                     ['Q', '', '', '', ''],
                                     ['P', '', '', '', '']])

    # Lists the number of amino acids, which belong to each state.
    # Each index refers to an identically numbered state.
    # start[0] -> [1] -> ... -> end[11]
    amino_acid_in = [0, 3, 4, 5, 5, 5, 5, 4, 1, 1, 1, 0]

    # Lists the number of amino acids, which aren't part the
    # corresponding state.
    # Each index refers to an identically numbered state.
    # start[0] -> [1] -> ... -> end[11]
    amino_acid_notin = [20, 17, 16, 15, 15, 15, 15, 16, 19, 19, 19, 20]

    def __init__(self, data):
        self.data = data

    # Counts transitions of each state to another, by iterating over all
    # adjacent pairs of characters in each sequence and using the multiple
    # property grouping to determine which states emit the next symbol. State 0
    # references the start state and state 11 the end state.
    def compute_transitions(self):

        # Initialize transition matrix with zeros.
        trans_matrix = np.zeros((12, 12), dtype=float)

        # Initialize counting matrix with zeros.
        trans_counts = np.zeros((12, 12), dtype=int)

        for entry in self.data:
            for p in entry:
                # Set to_state to state 11 if we are at the end of a sequence.
                if (p[1] == ')'):
                    to_state = [11]
                # Save the row index of each group which contain the next
                # character in the variable to_state. The row index is equal to
                # the state enumeration, thus the empty row in the MPG array.
                else:
                    to_state = np.where(self.multi_property_group == p[1])[0]
                # Set from_state to state 0 if we are at the beginning of a
                # sequence.
                if (p[0] == '('):
                    from_state = [0]
                # Compute all possible combinations of state transitions, which
                # could produce the next symbol and increment their respective
                # count by one.
                for j, k in itertools.product(to_state, from_state):
                    trans_counts[j][k] += 1
                from_state = to_state

        trans_per_state = []

        # Computes the total number of transitions per state.
        for i in range(0, 12):
            col_sum = 0
            for k in range(0, 12):
                col_sum += trans_counts[k][i]
            trans_per_state += [col_sum]

        # Omit first row and last column as they are no transitions from the
        # end state or to the start state. That way division by zero is
        # avoided.
        for i in range(0, 11):
            for k in range(1, 12):
                # Frequency counts are used to estimate the transition
                # probabilities.
                trans_matrix[k][i] = (trans_counts[k][i] / trans_per_state[i])

        return trans_matrix

    # Iteratively populates the emission matrix by weighting emission
    # probabilities of aa's which belong to a state and those who don't
    # differently. The respective probability is then divided by the number
    # of aa's which do belong or do not belong to the current state. The start
    # and the end state are ignored, because they do not emit any symbols.
    def compute_emissions(self):

        # Arbitrary probability to emit an amino acid which belongs to a state.
        EMISSION_PROB_IN_STATE = 0.9
        # Arbitrary probability to emit an amino acid which does not belongs to
        # a state.
        EMISSION_PROB_NOT_IN_STATE = 0.1

        # Initialize an emission matrix, which is expected to hold dictionaries
        # that map each amino acid to their emission probability at the ten
        # possible states.
        emission_matrix = []

        for state in range(1, 11):
            edict = {}
            for i in range(0, 20):
                if (self.amino_acids[i] in self.multi_property_group[state]):
                    edict[self.amino_acids[i]] = (EMISSION_PROB_IN_STATE /
                                                  self.amino_acid_in[state])
                else:
                    edict[self.amino_acids[i]] = (EMISSION_PROB_NOT_IN_STATE /
                                                  self.amino_acid_notin[state])
            emission_matrix.append(edict)

        return emission_matrix
