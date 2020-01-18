# Handwritten Text Recognition

> Most of the functions implemented here have been possible due to the repo at https://github.com/githubharald/SimpleHTR, please check out his work.

> Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.

The data-loader expects the IAM dataset in the data/ directory. Follow these instructions to get the dataset:

Register for free at IAM dataset website.
Download words/words.tgz.
Download ascii/words.txt.
Put words.txt into the data/ directory.
Create the directory data/words/.
Put the content (directories a01, a02, ...) of words.tgz into data/words/.
Go to data/ and run python checkDirs.py for a rough check if everything is ok.