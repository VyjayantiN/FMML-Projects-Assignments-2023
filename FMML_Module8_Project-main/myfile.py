import os
import pandas as pd
import numpy as np
from autohire.utils import clean_text, parse_pdf
from autohire.model import BayesianMulticlassModel

# Directory paths
data_dir = "autohire/data"
resume_dir = os.path.join(data_dir, "resumes")
resume_paths = [
    "C:\\Users\\nlnsa\\Downloads\\FMML_Module8_Project-main\\data\\resumes\\computers_1.pdf",
    "C:\\Users\\nlnsa\\Downloads\\FMML_Module8_Project-main\\data\\resumes\\computers_2.pdf"
]
dataset_file = os.path.join(data_dir, r"C:\Users\nlnsa\Downloads\FMML_Module8_Project-main\data\resume-dataset.csv")

# Load dataset
resume_df = pd.read_csv(dataset_file)
resume_df["Keywords"] = resume_df["Resume"].apply(clean_text)
x_train, y_train = resume_df["Keywords"].values, resume_df["Category"].values

# Encode labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train Bayesian multiclass model
num_classes = len(set(y_train))
num_tokens = sum(len(set(tokens)) for tokens in x_train)
model = BayesianMulticlassModel(num_classes, num_tokens)
model.fit(x_train, y_train_encoded)
print("Model trained")

# Parse resumes from PDF files
resume_texts = []
for resume_path in resume_paths:
    if resume_path.endswith(".pdf"):
        print(f"Parsing PDF: {resume_path}")
        resume_texts.append(" ".join(parse_pdf(resume_path)))
print("PDFs parsed")

# Predict categories for parsed resumes
for i, resume_text in enumerate(resume_texts):
    print(f"Processing resume {i+1}/{len(resume_texts)}")
    keywords = clean_text(resume_text)
    counts_vector = np.array([int(keywords.count(token)) for token in model.counts.flatten()])
    print(f"Counts vector length: {len(counts_vector)}")
    predicted_category = model.predict(counts_vector)[0]
    print(f"Predicted category: {predicted_category}")
print("Resumes processed")