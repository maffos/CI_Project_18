class Parser():

    def __init__(self):
        self.proteins = []
        self.labels = []

    def parse_train(self, file):
        self.proteins = []
        self.labels = []

        with open(file, "r") as raw_data:
            raw_data = raw_data.readlines()[1:]

            for line in raw_data:
                self.proteins.append(line.split()[0])
                self.labels.append(line.split()[2])

        return self.proteins, list(map(int, self.labels))

    def parse_predict(self, file):
        self.proteins = []
        self.labels = []

        with open(file, "r") as raw_data:
            raw_data = raw_data.readlines()[1:]

            for line in raw_data:
                self.proteins.append(line.split(',')[0])

        return self.proteins

    @staticmethod
    def _reshape(proteins, f, t):

        reshaped_prots = []
        for protein in proteins:
            reshaped_prots.append(protein[f:(t + 1)])

        return reshaped_prots