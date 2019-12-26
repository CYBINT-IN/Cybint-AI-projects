# DeepFashion Category and Attribute Prediction

> Most of the functions implemented here have been possible due to the repo at https://github.com/abhishekrana/DeepFashion, please check out his work.

> Updated Implementation on TeslaM60

> Implementation done for Python 3.6

### Steps to follow for execution from Scratch:

1.If creating a new dataset please make sure there is no dataset or bottleneck folder, then run dataset_create.ipynb (change the generate_categories list to include the categories that you want data generated for)

2.Reduce logging statements, if kernel lags use terminal and type in "find . -type f | wc -l", and if two consecutive calls generate same output, your script has completed executing.

3.You can add categories on top as there is no rm.shutil command, please be aware of overwriting existing files (both in bottleneck and dataset folder)

4.You can run the fashion_train_alt.ipynb script, generate bottlenecks on first run, train infinite times

PS: Can change any and all configs (training configs as well) inside config.py