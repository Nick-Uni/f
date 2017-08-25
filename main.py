import pickle as pkl
import numpy as np
from predict import predict

def main():
    percentage_correct = lambda y, y_hat: 100. * np.mean(y == y_hat)

    dataset_path = 'test.pkl'
    print('Loading data form \'%s\'...' % dataset_path)
    train_pkl = pkl.load(open(dataset_path, 'rb'))

    X_data, y_data = train_pkl
    X = X_data[:]
    y = y_data[:]

    print('Predicting first %d examples...' % X.shape[0])    
    y_hat = predict(X)
    print(y_hat)
    print('Accuracy: %.2f%%' % percentage_correct(y, y_hat))
    print('Done! =J')

if __name__ == '__main__':
    main()