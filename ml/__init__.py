"""Machine-learning pipeline for 1X2 match-outcome prediction.

This package is the modelling counterpart to the ``etl`` package. It consumes
the match data that the ETL pipeline persisted in MongoDB and trains a deep
neural network to predict the home/draw/away outcome of a football match.

Modules
-------
config
    Centralised, environment-driven configuration (Mongo connection, feature
    parameters, model hyper-parameters, temporal-split fractions).
features
    Causal feature engineering. Streams the schedule in chronological order and
    derives leak-free rolling-form features plus categorical encodings, writing
    a processed feature collection back to MongoDB.
data_loader
    A paginated ``IterableDataset`` that streams the processed feature
    collection from MongoDB in bounded memory, plus helpers to compute
    normalisation statistics and class weights without loading the data.
model
    The PyTorch ``nn.Module`` (categorical embeddings + regularised MLP).
train
    The reproducible, seeded training / validation / test driver.
"""
