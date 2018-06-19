from Parser import Parser
from Encoder import Encoder


class Predictor():

    def __init__(self):
        self.pars = Parser()
        self.enc = Encoder()
        self.prediction = []
        self.proteins = []

    def predict(self, svm, file):
        self.proteins = []
        self.prediction = []
        self.proteins = self.pars.parse_predict(file)
        self.prediction = svm.predict(self.enc.bin_encode_proteins(self.proteins))

        return self.prediction

    def write_prediction(self, file):

        file.writelines('Id,Prediction1\n')

        for index in range(len(self.proteins)):
            file.writelines(self.proteins[index] + ',' + str(self.prediction[index]) + str('\n'))

        file.close()