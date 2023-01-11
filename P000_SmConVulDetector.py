import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import ipykernel

import numpy as np
import pandas
from P010_clean_fragment import clean_fragment
from P020_vectorize_fragment import FragmentVectorizer
from models.blstm import BLSTM
from models.blstm_attention import BLSTM_Attention
from models.lstm import LSTM_Model
from models.simple_rnn import Simple_RNN
from P001_parser import parameter_parser


args = parameter_parser()

for arg in vars(args):
    print(arg, getattr(args, arg))

'''
step1.vector-step1.1.parse_file
    read train_data/reentrancy_1671.txt combine to fragment
    each fragment contain:
        1. raw code (formated) splited by line
        2. label
e.g.:
    14284.sol
    function FUN1() payable public {
    uint256 VAR1 = VAR2.FUN2(VAR3, VAR4);
    require(VAR1 > 1);
    VAR4 = VAR2.FUN3(VAR4, VAR1);
    if(!VAR5.call.value(VAR1).FUN4(400000)()) {
    VAR4 = VAR2.FUN2(VAR4, VAR1);
    0
    ---------------------------------
===>
    [[
        '14284.sol'
        , 'function FUN1() payable public {'
        , 'uint256 VAR1 = VAR2.FUN2(VAR3, VAR4);'
        , 'require(VAR1 > 1);'
        , 'VAR4 = VAR2.FUN3(VAR4, VAR1);'
        , 'if(!VAR5.call.value(VAR1).FUN4(400000)()) {'
        , 'VAR4 = VAR2.FUN2(VAR4, VAR1);'
    ], 0]
'''
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 33 in line and fragment:
                yield fragment, fragment_val
                fragment = []
            elif stripped.split()[0].isdigit():
                if fragment:
                    if stripped.isdigit():
                        fragment_val = int(stripped)
                    else:
                        fragment.append(stripped)
            else:
                fragment.append(stripped)

"""
Assuming all fragments can fit in memory, build list of fragment dictionaries
Dictionary contains fragments and vulnerability indicator
Add each fragment to fragmentVectorizer
Train fragmentVectorizer model, prepare for vectorization
Loop again through list of fragments
Vectorize each fragment and put vector into new list
Convert list of dictionaries to dataframe when all fragments are processed

step1.vector-step1.2vector
    1. build global token dict
    2. vectorize fragment
fragmentVector shape (100, 300)
"""
def get_vectors_df(filename, vector_length=300):
    # 1. build global token dict
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vector_length)
    for fragment, val in parse_file(filename):
        count += 1
        print("Collecting fragments...", count, end="\r")
        vectorizer.add_fragment(fragment)
        row = {"fragment": fragment, "val": val}
        fragments.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    vectorizer.train_model()
    print()
    # 2. vectorize fragment
    vectors = []
    count = 0
    for fragment in fragments:
        count += 1
        print("Processing fragments...", count, end="\r")
        vector = vectorizer.vectorize(fragment["fragment"])
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df


"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""

# tensorflow 1.15 && keras 2.2.5
def main():
    filename = args.dataset
    parse_file(filename)
    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_fragment_vectors.pkl"
    vector_length = args.vector_dim
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filename, vector_length)
        df.to_pickle(vector_filename)


    # print('-' * 80)
    # print(base) # reentrancy_1671
    # print(vector_filename) # reentrancy_1671_fragment_vectors.pkl
    # print('-' * 80)
    print('-'*33)
    print('BLSTM-ATT')
    print('-'*33)
    model = BLSTM_Attention(df, name=base)
    model.train()
    model.test()

    print('-'*33)
    print('BLSTM')
    print('-'*33)
    model = BLSTM(df, name=base)
    model.train()
    model.test()

    print('-'*33)
    print('LSTM_Model')
    print('-'*33)
    model = LSTM_Model(df, name=base)
    model.train()
    model.test()

    print('-'*33)
    print('Simple_RNN')
    print('-'*33)
    model = Simple_RNN(df, name=base)
    model.train()
    model.test()



if __name__ == "__main__":
    main()
