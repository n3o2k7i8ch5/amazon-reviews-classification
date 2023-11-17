# Amazon reviews classifier

This project was conducted for the WEDT (Wprowadzenie do eksploracji danych tekstowych) course at Warsaw University of Technology during the winter semester in 2019.  

A regular LSTM model was implemented and compared to a modified version of the LSTM model which took into account the POS (part of sentence) tags of each word.  

Project authors:  
- Daniel Iwanicki
- Emil Bałdyga
- Kamil Zych

## How to run
Download two datasets: `test.fr.txt.bz2` and `train.fr.txt.bz2` and put them into `data/` directory.  

The datasets can be found at kaggle: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews  

From the activated venv run `python -m spacy download en_core_web_sm`

Run `main.py` for the regular LSTM network.  
Run `main_pos.py` for the LSTM with POS network.