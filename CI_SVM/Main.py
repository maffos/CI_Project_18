from Descriptor import Descriptor
from Encoder import Encoder
from Parser import Parser
from Predictor import Predictor
from Trainer import Trainer


if __name__ == "__main__":

    #Usage example
    enc = Encoder()
    pars = Parser()
    trn = Trainer()
    desc = Descriptor()
    prd = Predictor()

    proteins, labels = pars.parse_train('project_training.txt')
    enc_proteins = enc.bin_encode_proteins(proteins)
    test = trn.cross_validate(enc_proteins, labels)
    print(test.mean())