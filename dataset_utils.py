"""
Handles dataset-related logics. Download from Kaggle.com and preprocess the data.
"""
import os

import kagglehub
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class ResumeDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

        self.df["received_callback"] = pd.to_numeric(
            self.df["received_callback"], errors='coerce')
        self.labels = self.df["received_callback"].values

        self.df["resume_quality"] = (
            self.df["resume_quality"] == "low").astype(int)

        self.string_columns = ["job_industry", "job_type", "job_ownership"]
        self.label_encoders = {col: LabelEncoder()
                               for col in self.string_columns}

        for col in self.string_columns:
            self.df[col] = self.df[col].astype(str)
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col])

        drop_columns = ["received_callback",
                        "job_ad_id", "job_city", "firstname"]

        features_df = self.df.drop(columns=drop_columns)

        for col in features_df.columns:
            features_df[col] = pd.to_numeric(
                features_df[col], errors='coerce').fillna(-1)

        self.features = features_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = torch.tensor(
            self.features.iloc[idx].values, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

    def return_label_encoders(self):
        return self.label_encoders


def create_dataloader(bs=32):
    load_dotenv()

    path = kagglehub.dataset_download(
        "utkarshx27/which-resume-attributes-drive-job-callbacks")

    data_file_path = os.path.join(path, "resume.csv")

    df = pd.read_csv(data_file_path)

    df.fillna(-1)

    dataset = ResumeDataset(df)
    label_encoders = dataset.return_label_encoders()
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    return dataloader, label_encoders


def create_dataframe():
    load_dotenv()

    path = kagglehub.dataset_download(
        "utkarshx27/which-resume-attributes-drive-job-callbacks")

    data_file_path = os.path.join(path, "resume.csv")

    df = pd.read_csv(data_file_path)

    df.fillna(-1)

    return df


if __name__ == "__main__":
    dl, label_encoders = create_dataloader()
    print("Columns: job_industry: 0, job_type: 1, job_ownership: 4")
    for batch in dl:
        features, labels = batch
        print("Features:", features[0])
        print("Labels:", labels[0])
        print("Example usage of label_encoders:")
        print(
            f"job_industry: {label_encoders['job_industry'].inverse_transform([int(features[0][0].tolist())])}")
        break

    print("-" * 100)
    df = create_dataframe()
    print(df)
