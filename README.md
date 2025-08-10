# 👫 **Match Prediction AI Agent**

An intelligent matchmaking system that combines **rule-based filtering**, **machine learning (scikit-learn)**, and a **PyTorch neural network** for compatibility prediction. Optimized for **soccer enthusiasts** with special boosts for users via **Africa Soccer Kings** compatibility scoring.

---

## 📑 **Table of Contents**

1. [Features](#-features)
2. [Tech Stack](#-tech-stack)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Machine Learning Workflow](#-machine-learning-workflow)
7. [PyTorch Neural Model](#-pytorch-neural-model)
8. [Testing](#-testing)
9. [Screenshots](#-screenshots)
10. [Contributing](#-contributing)
11. [License](#-license)

---

## ✨ **Features**

* 📝 **User Profile Input** — Age, sex, seeking preference, country, language, relationship goals, and bio.

* 🛡 **Rule-Based Filtering** — Matches based on:

  * Sex & preference alignment
  * Age range ±5 years
  * Exclusion of blocked, declined, deleted, or reported users

* 📊 **Machine Learning Predictions**:

  * **Scikit-learn GradientBoostingRegressor** for initial scoring
  * **TF-IDF bio vectorization** for text-based features
  * **PyTorch neural network** (`CompatibilityModel`) for deep compatibility learning

* ⚽ **Soccer Enthusiast Boost** — Extra scoring for “soccer” or “football” mentions in bios.

* 💾 **Model Persistence**:

  ```
  matchmaking_model.pkl
  tfidf_vectorizer.pkl
  label_encoders.pkl
  scaler.pkl
  pytorch_model.pth
  ```

* 📈 **Recommendation Output** — Top matches saved in `data/recommendations.csv` with detailed scoring.

---

## 🛠 **Tech Stack**

* **Language:** 🐍 Python 3.13

* **Web Framework:** 🌐 Streamlit

* **Machine Learning:**

  * `scikit-learn` — GradientBoostingRegressor, StandardScaler, LabelEncoder, TfidfVectorizer
  * **PyTorch** — Custom neural network for advanced scoring
  * `pandas`, `numpy`, `scipy`, `joblib`

* **Testing:** `pytest`

* **Config:** YAML (`config.yaml`)

---

## 📁 **Project Structure**

```
match_prediction_ai_agent/
├── .github/
│   └── workflows/
│       ├── pylint.yml
│       ├── tests-anaconda.yml
│       ├── deploy-azure.yml
├── data/
│   ├── Profiles.csv
│   ├── LikedUsers.csv
│   ├── MatchedUsers.csv
│   ├── BlockedUsers.csv
│   ├── DeclinedUsers.csv
│   ├── DeletedUsers.csv
│   ├── ReportedUsers.csv
├── models/
│   ├── pytorch_model.pth
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── user_to_idx.pkl
│   ├── profile_to_idx.pkl
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── recommender.py
│   ├── agent.py
│   ├── utils.py
├── ui/
│   ├── streamlit_app.py
│   ├── cli.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_agent.py
│   ├── test_recommender.py
│   ├── test_utils.py
├── config.yaml
├── requirements.txt
├── README.md
├── run.py
├── LICENSE
├── CONTRIBUTING.md
├── Dockerfile
```

---

## 📦 **Installation**

```bash
git clone https://github.com/<your-username>/match_prediction_ai_agent.git
cd match_prediction_ai_agent
python3.13 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶ **Usage**

### **Streamlit UI**

```bash
streamlit run ui/streamlit_app.py
```

### **CLI Mode**

```bash
python run.py --user_id user123 --age 25 --sex Male --seeking Female --country Kenya --language Swahili --relationship_goals "Long-term" --about_me "Love football and travel"
```

**Output:** `data/recommendations.csv`

---

## 🤖 **Machine Learning Workflow**

1. **Load Data & Preprocess**

   * Encode categorical variables
   * Vectorize bios using TF-IDF

2. **Initial Model (Scikit-learn)**

   * Train GradientBoostingRegressor
   * Store results in `ml_score`

3. **Deep Learning Model (PyTorch)**

   * `CompatibilityModel` uses user–profile interaction features + TF-IDF vectors
   * Trains with MSE loss and Adam optimizer
   * Supports incremental training & loading from `pytorch_model.pth`

4. **Final Score**

   * Weighted blend:

     ```
     final_score = ml_score * 0.7
                  + country_match * 0.1
                  + language_match * 0.1
                  + goal_match * 0.1
     ```

---

## 🔥 **PyTorch Neural Model**

The **`CompatibilityModel`** is a custom PyTorch feed-forward neural network used for deep compatibility scoring.
It works alongside the scikit-learn pipeline to improve prediction accuracy.

---

### **Model Definition**

```python
import torch
import torch.nn as nn

class CompatibilityModel(nn.Module):
    """PyTorch neural network for compatibility prediction."""
    def __init__(self, input_dim, hidden_dims):
        super(CompatibilityModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))  # Final output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

---

### **Loading a Pretrained Model**

```python
import os
import joblib
import torch

from src.agent import CompatibilityModel  # Adjust import to your structure

def load_model_and_scaler(models_dir):
    model_path = os.path.join(models_dir, "pytorch_model.pth")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = torch.load(model_path, weights_only=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        raise FileNotFoundError("Model or scaler not found.")

# Example
model, scaler = load_model_and_scaler("models/")
```

---

### **Training the Model**

```python
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

class CompatibilityDataset(Dataset):
    """Dataset for compatibility prediction."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(X_train, y_train, hidden_dims=[128, 64], lr=0.001, epochs=10, batch_size=32):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    dataset = CompatibilityDataset(X_scaled, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_scaled.shape[1]
    model = CompatibilityModel(input_dim, hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model, scaler

# Train and save
model, scaler = train_model(X_train, y_train)
torch.save(model, "models/pytorch_model.pth")
joblib.dump(scaler, "models/scaler.pkl")
```

---

### **Making Predictions**

```python
def predict_compatibility(model, scaler, X_features):
    X_scaled = scaler.transform(X_features)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(X_tensor).squeeze().numpy()

    return scores

# Example usage
predicted_scores = predict_compatibility(model, scaler, X_test)
print(predicted_scores)
```

---

## 🧪 **Testing**

```bash
pytest -v
pytest tests/test_agent.py -v
```

<img width="3456" height="2052" alt="image" src="https://github.com/user-attachments/assets/443e32ae-5d98-4698-802f-188b7b7a9700" />

<img width="3456" height="2052" alt="image" src="https://github.com/user-attachments/assets/73d49f98-e7fd-4f34-8b5a-e5c9faf6b423" />

<img width="3456" height="2052" alt="image" src="https://github.com/user-attachments/assets/2f752da3-cb75-4119-ba0a-72df201dbae4" />

<img width="3456" height="2052" alt="image" src="https://github.com/user-attachments/assets/e6eba357-ec5b-43cf-9c68-b2423ce4e7f5" />

---

## 📷 **Screenshots**

<img width="1710" height="621" alt="image" src="https://github.com/user-attachments/assets/474c4c4f-6ae2-4fe1-af6a-92bd61f82efd" />

<img width="1710" height="532" alt="image" src="https://github.com/user-attachments/assets/5e2c1d6b-4720-421f-9805-2b48e57f5318" />

---

## 🤝 **Contributing**

1. Fork & branch (`feature/YourFeature`)
2. Make changes, commit, and push
3. Submit PR (ensure all tests pass)

---

## 📜 **License**

MIT License — See [LICENSE](LICENSE)

---
