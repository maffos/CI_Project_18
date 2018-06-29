from Parser import Parser


class Predictor:

    """
    Comprises methods to predict input-data and write a prediction to file.

    Uses a previously trained model for prediction. Input-data is read
    in from a .csv or .tsv file using Parser and Encoder objects.
    Input data is recorded object internally and will be emptied once a new prediction is made.
    """

    def __init__(self, descriptor):
        self.pars = Parser()
        self.desc = descriptor
        self.prediction = []
        self.proteins = []

    def predict(self, model, file):

        """
        Reads data from 'file' and uses 'model' to predict this data.

        :param model: Trained model suitable for predicting sparse encoded data.
        :param file: Path to samples to be predicted.
        :return: prediction: Calculated prediction.
        """

        self.proteins = []
        self.prediction = []

        # Empties previously recorded data.
        self.__empty_records()

        self.proteins = self.pars.parse_predict(file)
        self.prediction = model.predict(self.desc.encode_proteins(self.proteins))

        # Information for user.
        print('>Prediction completed')

        return self.prediction

    def write_prediction(self, file):

        """
        Writes a internally recorded prediction to 'file'. 'file' is created if not existing.

        :param file: Path to file to be generated and written to.
        :return: Generates a new file with predicted output data.
        """

        separator = self.__choose_separator(file)

        file.writelines('Id,Prediction1\n')

        for index in range(len(self.proteins)):
            file.writelines(self.proteins[index] + separator + str(self.prediction[index]) + str('\n'))

        file.close()

    def __empty_records(self):
        """
        Private method used for emptying previous records.

        :return: None
        """

        self.proteins = []
        self.prediction = []

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
