# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess(path="adult_income_ml\\data\\adult.csv"):
    df = pd.read_csv(path)

    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    df["income"] = df["income"].map({" <=50K": 0, " >50K": 1})

    df.drop(columns=["fnlwgt"], inplace=True)

    X = df.drop("income", axis=1)
    y = df["income"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
