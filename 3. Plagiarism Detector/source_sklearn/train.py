from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

from sklearn.svm import LinearSVC


def model_fn(model_dir):
    print("Loading model.")

    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")

    return model


## TODO: Complete the main code
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    ## TODO: Add any additional arguments that you will need to pass into your model

    args = parser.parse_args()

    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = train_data.iloc[:, 0]
    train_x = train_data.iloc[:, 1:]

    ## TODO: Define a model
    model = LinearSVC()

    ## TODO: Train the model
    model.fit(train_x, train_y)

    ## --- End of your code  --- ##

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))