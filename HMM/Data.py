import itertools


class Data():

    # Expect relative paths to .txt files containing trainings data.
    def __init__(self, positive_data, negative_data):
        self.positive_data = positive_data
        self.negative_data = negative_data

    # Method which returns all adjacent pairs of characters in a string by
    # utilizing two independent iterators.
    def pairwise(self, myStr):
        a, b = itertools.tee(myStr)
        next(b, None)

        return [s1 + s2 for s1, s2 in zip(a, b)]

    # Reads given trainings data and omits all but the sequences of length nine
    # at the beginning of each line. Each sequence is enclosed by brackets,
    # which represent start and end state respectively. Following that, the
    # sequences are split into their adjacent pairs using the pairwise method.
    def split_in_pairs(self):
        with open(self.positive_data, 'r') as pd:
            pos_data = [self.pairwise('(' + line[:9] + ')') for line in pd]

        with open(self.negative_data, 'r') as ng:
            neg_data = [self.pairwise('(' + line[:9] + ')') for line in ng]

        return pos_data, neg_data
