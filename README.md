# 👫 **Match Prediction AI Agent**

An intelligent matchmaking system that combines **rule-based filtering**, **machine learning**, and **text analysis** to recommend the most compatible profiles. Optimized for **soccer enthusiasts** with special boosts for users via **Africa Soccer Kings** compatibility scoring.

---

## 📑 **Table of Contents**

1. [Features](#-features)
2. [Tech Stack](#-tech-stack)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Testing](#-testing)
7. [Screenshots](#-screenshots)
8. [Contributing](#-contributing)
9. [License](#-license)

---

## ✨ **Features**

* 📝 **User Profile Input** — Enter **age**, **sex**, **seeking preference**, **country**, **language**, **relationship goals**, and a personal bio.
* 🛡 **Rule-Based Filtering** — Matches based on:

  * Sex & preference alignment
  * Age range ±5 years
  * Excludes blocked, declined, deleted, or reported users
* 📊 **Machine Learning Predictions** — Compatibility scoring via **Gradient Boosting Regressor** with **TF-IDF** bio vectorization.
* ⚽ **Soccer Enthusiast Boost** — Increases scores for users mentioning “soccer” or “football” in their bio.
* ⚡ **Performance Optimization** — Streamlit’s `@st.cache_resource` speeds up data load and model training.
* 💾 **Model Persistence** — Saves:

  ```
  matchmaking_model.pkl
  tfidf_vectorizer.pkl
  label_encoders.pkl
  scaler.pkl
  ```
* 📈 **Recommendation Output** — Stores top matches in `data/recommendations.csv` with reasons.

---

## 🛠 **Tech Stack**

* **Language:** 🐍 Python 3.13
* **Web Framework:** 🌐 Streamlit
* **Machine Learning:**

  * `scikit-learn` — GradientBoostingRegressor, StandardScaler, LabelEncoder, TfidfVectorizer
  * `pandas` — Data handling
  * `numpy` — Numerical operations
  * `scipy` — Sparse matrix support
* **Data Storage:** CSV
* **Serialization:** `joblib`
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
# Clone repository
git clone https://github.com/<your-username>/matchmaking_ai_agent.git
cd matchmaking_ai_agent

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶ **Usage**

**Run via Streamlit UI:**

```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` and fill in profile details.

**Run via CLI:**

```bash
python run.py --user_id user123 --age 25 --sex Male --seeking Female --country Kenya --language Swahili --relationship_goals "Long-term" --about_me "Love football and travel"
```

**Output:** Recommendations saved in `data/recommendations.csv`.

---

## 🧪 **Testing**

```bash
pytest -v
```

Run a specific test:

```bash
pytest tests/test_agent.py -v
```

<img width="1728" height="1026" alt="Test Agent" src="https://github.com/user-attachments/assets/765e2fc8-c052-47ec-8ff1-42a4ea510c3e" />

<img width="1728" height="1026" alt="Test Data Loader" src="https://github.com/user-attachments/assets/2a6c7d3a-a7b0-4106-bd7f-16a608d3b1df" />

<img width="3456" height="2052" alt="image" src="https://github.com/user-attachments/assets/6dd533bf-ffd3-428d-95d6-54d4cdfbd7af" />

<img width="3456" height="2052" alt="image" src="https://github.com/user-attachments/assets/e2d37a7b-4d66-4660-9eff-90d2ea0fd808" />

---

## 📷 **Screenshots**

**Main Interface:** <img src="https://github.com/user-attachments/assets/53fc99ac-b3be-4b9a-a3c8-97e7fd5f1e41" width="800"/>

**Recommendations:** <img src="https://github.com/user-attachments/assets/7b80bfd4-2678-46d9-8317-4d45bd89681e" width="800"/>

---

## 🤝 **Contributing**

1. Fork repo
2. Create branch `feature/YourFeature`
3. Commit changes
4. Push and open PR

Please ensure **all tests pass** before PR submission.

---

## 📜 **License**

MIT License — See [LICENSE](LICENSE) file.

---


