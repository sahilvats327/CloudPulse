\# CloudPulse — Cloud Resource Anomaly Detection



CloudPulse is a machine-learning project that analyzes cloud resource metrics to identify unusual behavior and potential resource anomalies. It compares multiple anomaly-detection approaches and presents their results through an interactive Streamlit dashboard.



\## Features



\- \*\*Resource monitoring:\*\* Analyzes CPU utilization, memory utilization, and network traffic.

\- \*\*Multiple ML models:\*\* Compares LSTM, GRU, XGBoost, a hybrid detection approach, and an ARIMA-based network detector.

\- \*\*Interactive dashboard:\*\* Displays model performance and anomaly-detection results.

\- \*\*Model evaluation:\*\* Compares accuracy, precision, recall, F1-score, and confusion matrices.

\- \*\*Anomaly analysis:\*\* Examines different anomaly patterns, including CPU spikes, memory drift, and network drops.



\## Technology Stack



\- Python

\- Streamlit

\- Pandas and NumPy

\- TensorFlow / Keras

\- Scikit-learn

\- XGBoost

\- Statsmodels

\- Matplotlib



\## Project Structure



```text

CloudPulse/

├── app.py

├── data/

│   └── cloud\_metrics.csv

├── models/

│   ├── lstm\_model.keras

│   ├── gru\_model.keras

│   └── other trained model artifacts and configurations

├── outputs/

│   └── evaluation results, reports, and visualizations

├── requirements.txt

├── generate\_data.py

├── lstm\_model.py

├── gru\_model.py

├── xgboost\_model.py

└── test\_anomaly\_explainer.py

```



\## Setup and Run



\### 1. Install Python



Use Python 3.12 and ensure Python is available from Command Prompt.



\### 2. Clone the repository



```cmd

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd CloudPulse

```



Replace `YOUR\_GITHUB\_REPOSITORY\_URL` with the repository URL.



\### 3. Create and activate a virtual environment



On Windows:



```cmd

py -3.12 -m venv .venv

.venv\\Scripts\\activate

```



\### 4. Install dependencies



```cmd

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### 5. Launch the dashboard



```cmd

streamlit run app.py

```



Streamlit will provide a local URL to open the dashboard in your browser.



\## Model Evaluation



CloudPulse evaluates its models using held-out test data and compares their anomaly-detection performance using standard classification metrics.



The current experiments use \*\*synthetic cloud resource metrics\*\*. The results demonstrate the project's experimental performance and do not represent production results from live AWS infrastructure.



\## Future Improvements



\- Integrate real cloud metrics from AWS CloudWatch.

\- Add continuous monitoring and configurable alerting.

\- Improve anomaly explanations and root-cause analysis.

\- Evaluate model performance on real-world cloud workloads.

\- Explore deployment as a cloud-hosted monitoring service.



\## Disclaimer



CloudPulse is an educational and experimental project. Its outputs should not be treated as proof of a real cloud incident without further validation.



\## Author



\*\*Sahil Vats\*\*  

B.Tech — Cloud Computing and Automation  





