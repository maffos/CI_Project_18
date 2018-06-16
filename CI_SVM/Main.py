import Training as train
import Model as model

'''
TODO: Description
'''


if __name__ == "__main__":

    # svm,sparse_encoder,int_encoder = train.train_dimension_reduction("project_training.txt")
    svm, sparse_encoder, int_encoder = train.train_sparse("project_training.txt")

    print("Best params:")
    print(svm.best_params_)
    #print("Scores:")
    #print(svm.best_score_)

    model.predict("test_input.txt", svm, sparse_encoder, int_encoder)
