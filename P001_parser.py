import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contracts Reentrancy Detection')

    parser.add_argument('-D', '--dataset', type=str, default='train_data/infinite_loop_1317.txt')
    parser.add_argument('-M', '--model', type=str, default='BLSTM_Attention',
                        choices=['BLSTM', 'BLSTM_Attention', 'LSTM_Model', 'Simple_RNN'])
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--vector_dim', type=int, default=300, help='dimensions of vector') # droped
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='threshold')

    return parser.parse_args()
