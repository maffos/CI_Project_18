class Parser:

    """
    Comprises methods to parse input-data in .csv or .tsv format.

    Reads a file and records samples and labels internally as lists.
    By reading a file, previously recorded data is emptied.
    """

    def __init__(self):
        self.proteins = []
        self.labels = []

    def parse_train(self, file):

        """
        Reads a file for training purpose. Samples and labels are recorded internally as lists.

        :param file: Path to file to be read.
        :return: proteins: Stored samples (String list), labels: Stored labels (Integer list)
        """

        # Empties previously recorded data.
        self.__empty_records()

        # Separator used to split lines in input file.
        separator = self.__choose_separator(file)

        with open(file, "r") as input_file:

            # Remove header from file.
            input_file = input_file.readlines()[1:]

            for line in input_file:

                # Splits line at separator-symbol.
                data = line.split(separator)

                # Appends samples and labels to respective list.
                self.proteins.append(data[0])
                self.labels.append(data[2])

        return self.proteins, list(map(int, self.labels))

    def parse_predict(self, file):

        """
        Reads a file for prediction. Samples are recorded internally as list.

        :param file: Path to file to be read.
        :return: proteins: Internally stored samples (String)
        """

        # Empties previously recorded data.
        self.__empty_records()

        # Separator used to split lines in input file.
        separator = self.__choose_separator(file)

        with open(file, "r") as input_file:

            # Remove header from file.
            input_file = input_file.readlines()[1:]

            for line in input_file:

                # Appends samples to respective list.
                self.proteins.append(line.split(separator)[0])

        return self.proteins

    def __empty_records(self):
        """
        Private method used for emptying previous records.

        :return: None
        """

        self.proteins = []
        self.labels = []

        return None

    @staticmethod
    def __choose_separator(file):

        """
        Chooses separator in dependence on file format. Tab separation is default.

        :param file: Path to file to choose separator.
        :return: separator: Separator specified by file format.
        """

        # Separator used to split lines in input file.
        separator = '\t'

        # If file is .csv format, use ',' as separator instead.
        if file.split('.')[1] == 'csv':
            separator = ','

        return separator
