import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset with resumes and job fit labels
data = {
    "resume": [
        "Experienced data scientist with Python, ML, and SQL expertise.",
        "Software engineer skilled in Java, Spring Boot, and microservices.",
        "Marketing specialist with SEO, digital ads, and content strategy background.",
        "AI researcher with deep learning, NLP, and reinforcement learning experience.",
        "Data analyst proficient in Excel, Tableau, and Power BI.",
    ],
    "label": ["data_science", "software_engineering", "marketing", "ai_research", "data_analytics"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["resume"], df["label"], test_size=0.2, random_state=42)

# Build a text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test on new resumes
test_resumes = [
    "Machine learning engineer with expertise in TensorFlow, PyTorch, and cloud deployment.",
    "Frontend developer proficient in React, JavaScript, and UI/UX design."
]

# Predictions
predictions = pipeline.predict(test_resumes)

# Output Results
for i, resume in enumerate(test_resumes):
    print(f"Resume: {resume}\nPredicted Job Category: {predictions[i]}\n")
