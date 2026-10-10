<div align="center">

# 🧬 Diabetes Digital Twin

### Understanding Glucose Patterns with Context-Aware Machine Learning

**🏆 Digital Twin Challenge 2026 &nbsp;·&nbsp; Team CodeForge &nbsp;·&nbsp; S J B Institute of Technology**

<br>

<img src="https://img.shields.io/badge/DIGITAL%20HEALTH-0F766E?style=for-the-badge&logo=heart&logoColor=white" alt="Digital Health"/>
<img src="https://img.shields.io/badge/PYTHON-2563EB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/MACHINE%20LEARNING-7C3AED?style=for-the-badge" alt="Machine Learning"/>
<img src="https://img.shields.io/badge/XGBOOST-EA580C?style=for-the-badge" alt="XGBoost"/>

<img src="https://img.shields.io/badge/GLUCOSE%20ANALYTICS-0891B2?style=flat-square" alt="Glucose Analytics"/>
<img src="https://img.shields.io/badge/HACKATHON-2026-F59E0B?style=flat-square" alt="Hackathon 2026"/>
<img src="https://img.shields.io/badge/STATUS-RESEARCH%20PROTOTYPE-DB2777?style=flat-square" alt="Research Prototype"/>
<img src="https://img.shields.io/badge/NOT%20A%20MEDICAL%20DEVICE-DC2626?style=flat-square" alt="Not a medical device"/>

<br><br>

*We combine glucose readings, meals, insulin and health information, then let machine learning look for patterns a human might miss.*

<br>

</div>

---

## 🌍 What Is This Project?

> **In one sentence:** we built a research prototype that studies a person's blood sugar (glucose) together with *what was happening around it* (food, insulin, health background) and tests which machine learning model finds the patterns best.

### 🧒 Explained simply

| Term | What it means |
|:--|:--|
| 🩸 **Diabetes** | A condition where the body has trouble controlling blood sugar. |
| 📟 **CGM** (Continuous Glucose Monitor) | A small wearable sensor that measures glucose every few minutes, day and night. |
| 👤 **Digital Twin** | A virtual "copy" of a person's health data that can be studied on a computer without touching the real patient. |
| 🤖 **Machine Learning** | Software that learns patterns from past data instead of following hand-written rules. |

### ❓ Why does it matter?

Glucose doesn't change on its own. A meal, a dose of insulin, or a person's health background can all push it up or down. Looking at glucose numbers **alone** misses that story.

**Our goal:** build a clear, repeatable workflow that brings this context together and fairly compares different models, so future researchers have a solid starting point.

---

## 🌟 Project at a Glance

<table>
<tr>
<td width="50%" valign="top">

### 🩺 Digital Health
Uses continuous glucose monitoring data plus the clinical context that goes with it.

</td>
<td width="50%" valign="top">

### 🧠 Machine Learning
Trains and compares three different classification models side by side.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔬 Context-Aware
Looks at glucose *together with* meals, insulin and clinical information.

</td>
<td width="50%" valign="top">

### 🏆 Hackathon Project
Built for the Digital Twin Challenge 2026 by **Team CodeForge**.

</td>
</tr>
</table>

---

## 💡 Our Approach in 4 Steps

<table>
<tr>
<td align="center" width="25%">

### 🗂️ 01
**Collect**

Gather glucose, meal, insulin and clinical data in one place.

</td>
<td align="center" width="25%">

### ⚙️ 02
**Prepare**

Clean the data and create useful "features" (helpful clues for the model).

</td>
<td align="center" width="25%">

### 🤖 03
**Train**

Teach three different models using the same data.

</td>
<td align="center" width="25%">

### 📊 04
**Compare**

Score every model with the same fair tests.

</td>
</tr>
</table>

---

## 🏗️ How It Works

<table>
<tr>
<td align="center" width="25%" bgcolor="#E0F2FE">

### 📥 01 · DATA INPUTS

📟 CGM Data

🍽️ Meal & Insulin Context

🩺 Clinical Metadata

</td>
<td align="center" width="25%" bgcolor="#CCFBF1">

### ⚙️ 02 · PROCESSING

Data Integration

Cleaning & Preparation

Feature Engineering

Dataset Splitting

</td>
<td align="center" width="25%" bgcolor="#EDE9FE">

### 🤖 03 · MACHINE LEARNING

🌲 Random Forest

⚡ XGBoost

📈 Logistic Regression

</td>
<td align="center" width="25%" bgcolor="#DCFCE7">

### 📊 04 · EVALUATION

Model Comparison

Glucose Pattern Analysis

</td>
</tr>
</table>

<p align="center"><b>📥 Inputs &nbsp;➡️&nbsp; ⚙️ Processing &nbsp;➡️&nbsp; 🤖 Models &nbsp;➡️&nbsp; 📊 Results</b></p>

📄 **Want the full-size picture?** [View the architecture diagram (PDF)](https://drive.google.com/file/d/1qorq1XEy9iq5TBT-sigGCrkG5YOo8DD1/view?usp=drive_link)

---

## 🤖 The Three Models We Compared

<table>
<tr>
<td width="33%" valign="top">

### 🌲 Random Forest
Imagine asking **hundreds of simple decision trees** for their opinion and going with the majority vote.

</td>
<td width="33%" valign="top">

### ⚡ XGBoost
Builds trees **one after another**, where each new tree fixes the mistakes of the previous ones.

</td>
<td width="33%" valign="top">

### 📈 Logistic Regression
A **simple, classic baseline**. If a fancier model can't beat it, the fancy model isn't worth it.

</td>
</tr>
</table>

---

## 📊 Results

> ⚠️ **Note:** these figures come from an earlier draft of the project. They should be re-checked against the latest evaluation output before final submission.

| Metric | 🌲 Random Forest | ⚡ XGBoost | 📈 Logistic Regression |
|:--|:--:|:--:|:--:|
| **Accuracy** | **67.7%** 🥇 | 63.2% | 53.7% |
| **Precision** | **0.292** 🥇 | 0.265 | 0.222 |
| **Recall** | 0.629 | 0.657 | **0.687** 🥇 |
| **F1-score** | **0.399** 🥇 | 0.378 | 0.336 |
| **ROC-AUC** | **0.721** 🥇 | 0.704 | 0.656 |
| **PR-AUC** | **0.396** 🥇 | 0.388 | 0.336 |

### 🧾 What do these numbers mean?

| Metric | In plain English |
|:--|:--|
| 🎯 **Accuracy** | Out of all predictions, how many were correct? |
| 🔍 **Precision** | When the model raises a flag, how often is it *right*? (fewer false alarms = higher) |
| 📡 **Recall** | Of all the *real* cases, how many did the model catch? (fewer misses = higher) |
| ⚖️ **F1-score** | One number that balances precision and recall. |
| 📈 **ROC-AUC** | How well the model separates the two groups. **0.5 = coin flip, 1.0 = perfect.** |
| 📉 **PR-AUC** | Similar to ROC-AUC, but more informative when one group is much rarer than the other. |

### 🔎 Key Takeaways

- 🌲 **Random Forest** performed best overall: highest accuracy, F1-score, ROC-AUC and PR-AUC.
- ⚡ **XGBoost** landed in the middle on most measures.
- 📈 **Logistic Regression** caught the most real cases (highest recall) but raised the most false alarms.
- 🤔 **No model is perfect.** Precision is below 0.30 for all three, so most flags raised are false alarms. Which model is "best" depends on what matters more: *missing a real case* or *raising a false alarm*.

---

## 🧰 Technology Stack

| | Technology | What we use it for |
|:-:|:--|:--|
| 🐍 | **Python** | Core programming language |
| 🐼 | **Pandas & NumPy** | Handling and calculating with data |
| 🤖 | **Scikit-learn** | Machine learning and evaluation |
| ⚡ | **XGBoost** | Gradient-boosting model |
| 🐙 | **Git & GitHub** | Version control and teamwork |

*Exact libraries and versions are listed in `requirements.txt`.*

---

## 🚀 Run It Yourself

**You'll need:** Python, Git, and a code editor such as VS Code.

<details open>
<summary><b>1️⃣ Download the project</b></summary>

```bash
git clone https://github.com/kartiksrathod/diabetes-digital-twin.git
cd diabetes-digital-twin
```
</details>

<details open>
<summary><b>2️⃣ Create a virtual environment</b> (keeps the project's packages separate)</summary>

```bash
python -m venv .venv
```

Then activate it:

**🪟 Windows**
```bash
.venv\Scripts\activate
```

**🍎 macOS / 🐧 Linux**
```bash
source .venv/bin/activate
```
</details>

<details open>
<summary><b>3️⃣ Install the required packages</b></summary>

```bash
pip install -r requirements.txt
```
</details>

<details open>
<summary><b>4️⃣ Run the project</b></summary>

Check the entry-point instructions in the source code, especially the `ml/`, `digital_twin/` and `mcp_server/` folders, before running any script.
</details>

---

## 📁 What's Inside the Repository

```text
diabetes-digital-twin/
├── 📂 .agents/          # AI-agent configuration
├── 📂 data/             # Datasets
├── 📂 digital_twin/     # Digital twin logic
├── 📂 mcp_server/       # Server components
├── 📂 ml/               # Machine learning code
├── 📄 mcp_server.py     # Server entry point
├── 📄 requirements.txt  # Python packages needed
├── 📄 .gitignore
└── 📄 README.md         # You are here
```

---

## 🌱 Where This Could Lead

<table>
<tr>
<td width="50%" valign="top">

### 📉 Glucose Trend Analysis
See glucose readings alongside meals, insulin and other context.

</td>
<td width="50%" valign="top">

### 🧪 Model Comparison
Fairly test which type of model works best on health data.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🧠 Personalised Analytics
A foundation for studying one person's unique glucose patterns.

</td>
<td width="50%" valign="top">

### 🔬 Digital Twin Research
A starting point for experiments with combined health data and models.

</td>
</tr>
</table>

> These are **research possibilities**, not claims of clinically proven performance.

---

## 🔮 Future Scope

- [ ] 📊 Interactive glucose trend dashboards
- [ ] ⏱️ Time-series forecasting experiments
- [ ] 🧠 Personalised models for individual patients
- [ ] 📡 Real-time data integration, where available
- [ ] 🧪 Testing on independent datasets
- [ ] 📈 Better evaluation and visualisation
- [ ] 🔍 Deeper study of false alarms and missed cases

---

## 👥 Meet Team CodeForge

<div align="center">

| 👨‍💻 Kartik S Rathod | 👨‍💻 J M Prajwal | 👨‍💻 John S Mark | 👨‍💻 Nikhil KP |
|:--:|:--:|:--:|:--:|

**🏫 S J B Institute of Technology** &nbsp;·&nbsp; **🏆 Digital Twin Challenge 2026**

</div>

---

## 🔗 Project Resources

| Resource | Link |
|:--|:--|
| 💻 **GitHub Repository** | [Open repository](https://github.com/kartiksrathod/diabetes-digital-twin) |
| 🏗️ **Architecture Diagram** | [View PDF](https://drive.google.com/file/d/1qorq1XEy9iq5TBT-sigGCrkG5YOo8DD1/view?usp=drive_link) |
| 🎤 **Presentation** | 🔜 Coming soon |
| 🎬 **Demo Video** | 🔜 Coming soon |

---

## ⚠️ Important Disclaimer

> 🚫 **This is NOT a medical device.**
> This project is for **education and research only**. It must **not** be used to diagnose, treat, or make decisions about anyone's diabetes care. Always talk to a qualified doctor about health decisions. Any model would need thorough independent validation before real-world healthcare use.

---

## 📜 License

No license has been confirmed yet. Before calling the project open source, add a `LICENSE` file (for example MIT or Apache-2.0) to the repository.

---

<div align="center">

### 💙 Made with curiosity by Team CodeForge

**Digital Twin Challenge 2026**

*Exploring the intersection of healthcare data and machine learning.*

⭐ If you found this project interesting, consider giving it a star!

</div>
