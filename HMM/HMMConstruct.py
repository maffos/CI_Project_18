from pomegranate import *


class HMMConstruct():
    # Parameters: Expects numpy array of all possible transitions, a list
    # of dictionaries which contain the emission probabilities for each symbol
    # in ascending order of state and a path of to the sequences which will be
    # used for training the HMM model.
    def __init__(self, transition_matrix, emission_matrix, data):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.data = data
        self.hmm = HiddenMarkovModel()

    def build_eval_HMM(self):

        # Maps the emission probabilities of each symbol to a discrete
        # distribution. This step is repeated for each state.
        d1 = DiscreteDistribution(self.emission_matrix[0])
        d2 = DiscreteDistribution(self.emission_matrix[1])
        d3 = DiscreteDistribution(self.emission_matrix[2])
        d4 = DiscreteDistribution(self.emission_matrix[3])
        d5 = DiscreteDistribution(self.emission_matrix[4])
        d6 = DiscreteDistribution(self.emission_matrix[5])
        d7 = DiscreteDistribution(self.emission_matrix[6])
        d8 = DiscreteDistribution(self.emission_matrix[7])
        d9 = DiscreteDistribution(self.emission_matrix[8])
        d10 = DiscreteDistribution(self.emission_matrix[9])

        # Initialize states.
        s1 = State(d1, name="s1")
        s2 = State(d2, name="s2")
        s3 = State(d3, name="s3")
        s4 = State(d4, name="s4")
        s5 = State(d5, name="s5")
        s6 = State(d6, name="s6")
        s7 = State(d7, name="s7")
        s8 = State(d8, name="s8")
        s9 = State(d9, name="s9")
        s10 = State(d10, name="s10")

        states = [self.hmm.start, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                  self.hmm.end]
        self.hmm.add_states(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)

        # Assigns transition probabilities to each possible transition
        # according to given transition matrix.
        for i in range(0, 11):
            for k in range(1, 12):
                self.hmm.add_transition(states[i], states[k],
                                        self.transition_matrix[k][i])

        # Finalize the topology of the model and assign a numerical index to
        # every state. This step also automatically normalizes all transitions
        # to make sure they sum to 1.0.
        self.hmm.bake(verbose=True)

        with open(self.data, 'r') as dt:
            data_training = [line[:9] for line in dt]

        # Train the HMM model using the Baum-Welch algorithm.
        # Add parameter verbose=True to show training progress.
        self.hmm.fit(data_training, max_iterations=23)

    # Accepts a list of individual string characters and uses the
    # Viterbi-Algorithm to calculate the negative log likelihood
    # of them appearing in this order.
    def log_probability(self, sequence):
        prob = self.hmm.log_probability(sequence)

        return prob
