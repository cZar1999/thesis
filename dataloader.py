import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from pyts.datasets import fetch_ucr_dataset

# For the MIT-BIH Arrythmia dataset, you must specify the folder where mitbih_train.csv 
# and mitbih_test.csv are located
# You can download these data at https://www.kaggle.com/datasets/shayanfazeli/heartbeat

class LoadData:
    def __init__(self, datafolder, deep=False):
        np.random.seed(seed=42) # DON'T modify, for the sake of reproducibility
        self.datafolder = datafolder
        self.deep = deep # Whether or not to use the datasets for deep learning, or for 5cv 
        # For each get_<DatasetName>_dat() of this class
        # If set deep = False, the method will return: X_train_val, Y_train_val, cv, X_test, Y_test
        # If set deep = True, the method will return: X_train, X_val, X_test, Y_val, Y_test
        self.scaler = MinMaxScaler()

    @staticmethod
    def gen_cv_folds(X_train, X_val, Y_val):
        np.random.seed(seed=42)
        X_train_val = np.concatenate((X_train, X_val))
        Y_train_val = np.concatenate((np.zeros(len(X_train)), Y_val))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #  Don't change
        custom_skf = list()
        for train_ind, val_ind in skf.split(X_train_val, Y_train_val):
            curr_train_ind = list(set(train_ind) - set(set(np.argwhere(Y_train_val == 1).flatten())))
            curr_val_ind = list(set(val_ind).union(set(set(np.argwhere(Y_train_val == 1).flatten()))))
            custom_skf.append((curr_train_ind, curr_val_ind))
        return custom_skf
   
    @staticmethod
    def unison_shuffled_copies(a, b):
        np.random.seed(seed=42)
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
   
    def get_mit_bih_data(self):
       
        H_train = pd.read_csv(self.datafolder + "/mitbih_train.csv", header=None)
        H_test = pd.read_csv(self.datafolder + "/mitbih_test.csv", header=None)
        H = pd.concat([H_train, H_test])
        H.rename(columns={i: str(i) for i in range(187)}, inplace=True)
        H.rename(columns={187: "label"}, inplace=True)
        H["label"] = H["label"].apply(int)
        X = H.drop(columns=["label"]).to_numpy()
        Y = H["label"].to_numpy()

        X = np.vstack((X[Y == 0], X[Y == 1])) # Only retain the data points with label 0 and 1
        Y  = np.concatenate((Y[Y==0], Y[Y==1])) # Retain necessary labels

       
        X = self.scaler.fit_transform(X)

        np.random.seed(seed=42)
        X, Y = LoadData.unison_shuffled_copies(X,Y)

        # Arbitrarily split the data according to  0, 1 labels
        # We make sure instances in (train + validation) and test don't overlap  

        X_test  = X[int(0.93 * len(X)):]
        Y_test = Y[int(0.93 * len(X)):]

        _X = X[:int(0.92 * len(X))]
        _Y = Y[:int(0.92 * len(Y))]
       
        # Generates train val data
       
        def subsample_train_val(X, Y):
            # Normal data
            X_n = X[Y == 0]
            X_train = X_n[:int(len(X_n) * 0.15)] # 20 % percent uses for training
            X_n = X[Y == 0][int(len(X_n) * 0.5):] # make sure we don't have the same in training and validation
            X_val = X_n[int(len(X_n) * 0.7):int(len(X_n) * 0.78)]
            Y_val = np.zeros(len(X_val))

            # Anomalous data
            X_a = X[Y == 1]  # None of them figure in train data
            X_val_a = X_a[:int(len(X_a) * 0.02)] # 0.2 % of class 1 is used
            Y_val_a = np.ones(len(X_val_a))

            # Join the sets
            X_val = np.concatenate((X_val, X_val_a))
            Y_val = np.concatenate((Y_val, Y_val_a)).astype(int)

            return X_train, X_val, Y_val


        X_train, X_val, Y_val = subsample_train_val(_X,_Y)
       
        if self.deep == True:
            X_train = X_train.reshape(len(X_train), 187, 1)
            X_val = X_val.reshape(len(X_val), 187, 1)
            X_test = X_test.reshape(len(X_test), 187, 1)
            return X_train, X_val, X_test, Y_val, Y_test
       
        else:
            cv = LoadData.gen_cv_folds(X_train,X_val, Y_val)
            X_train_val = np.concatenate((X_train, X_val))
            Y_train_val = np.concatenate((np.zeros(len(X_train)), Y_val))
            return X_train_val, Y_train_val, cv, X_test, Y_test
       
       
    def get_yoga_data(self):
        return NotImplementedError
   
       
   
    def get_mote_strain_data(self):
       
        data = fetch_ucr_dataset("MoteStrain")
        X = np.concatenate((data["data_train"], data["data_test"]))
        # X = self.scaler.fit_transform(X)
       
        Y = np.concatenate((data["target_train"], data["target_test"]))
        Y = np.array([0 if i == 1 else 1 for i in Y])
        np.random.seed(seed=42)
        X, Y = LoadData.unison_shuffled_copies(X,Y)
        # Now split into train val and test sets
        X_n = X[Y == 0]
        X_train = X_n[:int(len(X_n) * 0.65)]
        X_val_n = X_n[int(len(X_n) * 0.65):int(len(X_n) * 0.75)]
        Y_val_n = np.zeros(len(X_val_n))
        X_test_n = X_n[int(len(X_n) * 0.80):]
        Y_test_n = np.zeros(len(X_test_n))


        X_a = X[Y == 1]  # None of them figure in train data
        X_val_a = X_a[:int(len(X_a) * 0.02)]
        Y_val_a = np.ones(len(X_val_a))
        X_test_a = X_a[int(len(X_a) * 0.96):]
        Y_test_a = np.ones(len(X_test_a))


        X_val = np.concatenate((X_val_n, X_val_a))
        X_test = np.concatenate((X_test_n, X_test_a))
        Y_val = np.concatenate((Y_val_n, Y_val_a))
        Y_test = np.concatenate((Y_test_n, Y_test_a))
       
        if self.deep == True:
            X_train = X_train.reshape(len(X_train), 84, 1)
            X_val = X_val.reshape(len(X_val), 84, 1)
            X_test = X_test.reshape(len(X_test), 84, 1)
            return X_train, X_val, X_test, Y_val, Y_test
       
        else:
            cv = LoadData.gen_cv_folds(X_train, X_val, Y_val)
            X_train_val = np.concatenate((X_train, X_val))
            Y_train_val = np.concatenate((np.zeros(len(X_train)), Y_val))
            return X_train_val, Y_train_val, cv, X_test, Y_test
       
       
    def get_strawberry_data(self):
        data = fetch_ucr_dataset(dataset="Strawberry")
        X = np.concatenate((data["data_train"], data["data_test"]))

        X = self.scaler.fit_transform(X)
        Y = np.concatenate((data["target_train"], data["target_test"]))
        Y = np.array([0 if i == 2 else 1 for i in Y])
        X, Y = LoadData.unison_shuffled_copies(X,Y)
        np.random.seed(seed=42)
        X, Y = LoadData.unison_shuffled_copies(X,Y)

        X_n = X[Y == 0]
        X_train = X_n[:int(len(X_n) * 0.65)]
        X_val_n = X_n[int(len(X_n) * 0.65):int(len(X_n) * 0.80)]
        Y_val_n = np.zeros(len(X_val_n))
        X_test_n = X_n[int(len(X_n) * 0.80):]
        Y_test_n = np.zeros(len(X_test_n))


        X_a = X[Y == 1]  # None of them figure in train data
        X_val_a = X_a[:int(len(X_a) * 0.05)]
        Y_val_a = np.ones(len(X_val_a))
        X_test_a = X_a[int(len(X_a) * 0.96):]
        Y_test_a = np.ones(len(X_test_a))


        X_val = np.concatenate((X_val_n, X_val_a))
        X_test = np.concatenate((X_test_n, X_test_a))
        Y_val = np.concatenate((Y_val_n, Y_val_a))
        Y_test = np.concatenate((Y_test_n, Y_test_a))
       
        if self.deep == True:
            X_train = X_train.reshape(len(X_train), 235, 1)
            X_val = X_val.reshape(len(X_val), 235, 1)
            X_test = X_test.reshape(len(X_test), 235, 1)
            return X_train, X_val, X_test, Y_val, Y_test
       
        else:
            cv = LoadData.gen_cv_folds(X_train, X_val, Y_val)
            X_train_val = np.concatenate((X_train, X_val))
            Y_train_val = np.concatenate((np.zeros(len(X_train)), Y_val))
            return X_train_val, Y_train_val, cv, X_test, Y_test
   
   
    def get_chlorine_data(self):
        data = fetch_ucr_dataset(dataset="ChlorineConcentration")
        X = np.concatenate((data["data_train"], data["data_test"]))

        X = self.scaler.fit_transform(X)
        Y = np.concatenate((data["target_train"], data["target_test"]))
        Y = np.array([0 if i == 3 else 1 for i in Y])
        np.random.seed(seed=42)
        X, Y = LoadData.unison_shuffled_copies(X,Y)

        X_n = X[Y == 0]
        X_train = X_n[:int(len(X_n) * 0.72)]
        X_val_n = X_n[int(len(X_n) * 0.72):int(len(X_n) * 0.85)]
        Y_val_n = np.zeros(len(X_val_n))
        X_test_n = X_n[int(len(X_n) * 0.85):]
        Y_test_n = np.zeros(len(X_test_n))


        X_a = X[Y == 1]  # None of them figure in train data
        X_val_a = X_a[:int(len(X_a) * 0.009)]
        Y_val_a = np.ones(len(X_val_a))
        X_test_a = X_a[int(len(X_a) * 0.991):]
        Y_test_a = np.ones(len(X_test_a))


        X_val = np.concatenate((X_val_n, X_val_a))
        X_test = np.concatenate((X_test_n, X_test_a))
        Y_val = np.concatenate((Y_val_n, Y_val_a))
        Y_test = np.concatenate((Y_test_n, Y_test_a))
               
        if self.deep == True:
            X_train = X_train.reshape(len(X_train), 166, 1)
            X_val = X_val.reshape(len(X_val), 166, 1)
            X_test = X_test.reshape(len(X_test), 166, 1)
            return X_train, X_val, X_test, Y_val, Y_test
       
        else:
            cv = LoadData.gen_cv_folds(X_train, X_val, Y_val)
            X_train_val = np.concatenate((X_train, X_val))
            Y_train_val = np.concatenate((np.zeros(len(X_train)), Y_val))
            return X_train_val, Y_train_val, cv, X_test, Y_test
   
    def get_wafer_data(self):
        data = fetch_ucr_dataset(dataset="Wafer")
        X = np.concatenate((data["data_train"], data["data_test"]))

        X = self.scaler.fit_transform(X)
        Y = np.concatenate((data["target_train"], data["target_test"]))
        Y = np.array([0 if i == 1 else 1 for i in Y])
        print(Counter(Y))
        np.random.seed(seed=42)
        X, Y = LoadData.unison_shuffled_copies(X,Y)

        X_n = X[Y == 0]
        X_train = X_n[:int(len(X_n) * 0.72)]
        X_val_n = X_n[int(len(X_n) * 0.72):int(len(X_n) * 0.85)]
        Y_val_n = np.zeros(len(X_val_n))
        X_test_n = X_n[int(len(X_n) * 0.85):]
        Y_test_n = np.zeros(len(X_test_n))


        X_a = X[Y == 1]  # None of them figure in train data
        X_val_a = X_a[:int(len(X_a) * 0.04)]
        Y_val_a = np.ones(len(X_val_a))
        X_test_a = X_a[int(len(X_a) * 0.95):]
        Y_test_a = np.ones(len(X_test_a))

        X_val = np.concatenate((X_val_n, X_val_a))
        X_test = np.concatenate((X_test_n, X_test_a))
        Y_val = np.concatenate((Y_val_n, Y_val_a))
        Y_test = np.concatenate((Y_test_n, Y_test_a))
       
        if self.deep == True:
            X_train = X_train.reshape(len(X_train), 152, 1)
            X_val = X_val.reshape(len(X_val), 152, 1)
            X_test = X_test.reshape(len(X_test), 152, 1)
            return X_train, X_val, X_test, Y_val, Y_test
       
        else:
            cv = LoadData.gen_cv_folds(X_train, X_val, Y_val)
            X_train_val = np.concatenate((X_train, X_val))
            Y_train_val = np.concatenate((np.zeros(len(X_train)), Y_val))
            return X_train_val, Y_train_val, cv, X_test, 
