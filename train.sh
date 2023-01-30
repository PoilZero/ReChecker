#!/usr/bin/env bash

for i in $(seq 1 10);
do
python P000_SmConVulDetector.py -D train_data/delegatecall_196_directlly_from_dataset.txt | tee logs/delegatecall_196_directlly_from_dataset.txt_"$i".log;
python P000_SmConVulDetector.py -D train_data/integeroverflow_275_directlly_from_dataset.txt | tee logs/integeroverflow_275_directlly_from_dataset.txt_"$i".log;
#python P000_SmConVulDetector.py -D train_data/reentrancy_273_directlly_from_dataset.txt | tee logs/reentrancy_273_directlly_from_dataset.txt_"$i".log;
#python P000_SmConVulDetector.py -D train_data/timestamp_349_directlly_from_dataset.txt | tee logs/timestamp_349_directlly_from_dataset.txt_"$i".log;
done
