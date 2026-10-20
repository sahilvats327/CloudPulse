\# CloudPulse: Unified Model Evaluation Report



\## 1. Project Overview



CloudPulse is a prototype cloud-resource anomaly detection system that analyzes CPU utilization, memory utilization, and network traffic metrics. It compares machine-learning, deep-learning, statistical time-series, and hybrid detection approaches.



The project uses a synthetic dataset containing 10,000 observations, including 9,750 normal observations and 250 labeled anomalies. The dataset includes CPU spikes, memory drift, network drops, and combined anomalies.



\*\*Important:\*\* These results are based on synthetic data, not live AWS telemetry.



\## 2. Model Comparison



The following results are from `fair\_model\_comparison.csv`, which evaluates four approaches on the same 1,499-row test set containing 120 anomalies.



| Model | Accuracy | Precision | Recall | F1 Score | False Positives |

|---|---:|---:|---:|---:|---:|

| XGBoost V3 | 97.53% | 98.82% | 70.00% | 81.95% | 1 |

| CloudPulse V3 Hybrid | 99.40% | 99.12% | 93.33% | 96.14% | 1 |

| LSTM | 98.87% | 89.31% | 97.50% | 93.23% | 14 |

| GRU | 98.53% | 87.12% | 95.83% | 91.27% | 17 |



\### Results Interpretation



\- \*\*CloudPulse V3 Hybrid\*\* achieved the highest F1 score in the common comparison.

\- \*\*LSTM\*\* achieved the highest recall, detecting 117 of 120 anomalies, but generated 14 false positives.

\- \*\*GRU\*\* achieved 95.83% recall, with 17 false positives.

\- \*\*XGBoost V3 alone\*\* generated only one false positive but missed 36 anomalies.

\- The hybrid improved XGBoost's anomaly detection by adding a network-drop detection rule.



\## 3. Additional Detection Experiments



| Detector | Test Rows | Accuracy | Precision | Recall | F1 Score | False Positives |

|---|---:|---:|---:|---:|---:|---:|

| Adaptive Network Detector V1 | 1,499 | 90.59% | 13.79% | 3.33% | 5.37% | 25 |

| ARIMA Network Detector V1 | 1,500 | 92.33% | 58.06% | 15.00% | 23.84% | 13 |



The adaptive network detector and ARIMA detector performed poorly at their selected thresholds on the recorded test data. They are retained as experimental baselines rather than recommended components of the current detection pipeline.



These results are presented separately because their test-row counts and threshold-selection methods differ from those in the common comparison.



\## 4. Anomaly-Type Analysis



The recorded XGBoost V3 and hybrid results show the following detections:



| Anomaly Type | Actual Cases | XGBoost Detected | Hybrid Detected |

|---|---:|---:|---:|

| CPU spike | 20 | 19 | 19 |

| Memory drift | 50 | 45 | 45 |

| Network drop | 30 | 2 | 30 |

| Combined anomaly | 20 | 18 | 18 |



The hybrid detected all 30 network-drop cases in this synthetic test. Its other anomaly-type results were unchanged from XGBoost V3.



This improvement is associated with the network detection rule. It should not be interpreted as proof that the same threshold will work reliably with real-world network traffic.



\## 5. Stress Testing



Stress tests explored the effect of changing traffic patterns on XGBoost and the hybrid model.



| Scenario | Model | Precision | Recall | F1 Score | False Positives |

|---|---|---:|---:|---:|---:|

| Baseline | XGBoost | 98.82% | 70.00% | 81.95% | 1 |

| Baseline | Hybrid | 99.12% | 93.33% | 96.14% | 1 |

| Low normal traffic | XGBoost | 64.44% | 72.50% | 68.24% | 48 |

| Low normal traffic | Hybrid | 29.41% | 95.83% | 45.01% | 276 |

| Overlapping traffic | XGBoost | 66.41% | 72.50% | 69.32% | 44 |

| Overlapping traffic | Hybrid | 35.64% | 85.83% | 50.37% | 186 |



\### Stress-Test Findings



The hybrid performs well on the baseline synthetic test but becomes much less precise when normal traffic patterns change. Its increased recall comes at the cost of a large number of false alarms.



The gated-hybrid experiment produced the same recorded metrics as XGBoost alone in all tested scenarios. Therefore, the current gating implementation did not demonstrate an improvement.



These stress tests highlight the importance of testing network detection rules under diverse traffic conditions before deployment.



\## 6. Evaluation Limitations



1\. The dataset is synthetic and may not represent the full complexity of actual cloud environments.

2\. Some experiment files use 1,499 test rows, while others use 1,500.

3\. The standalone LSTM and GRU metric files contain slightly different confusion-matrix counts from the common comparison CSV. The common comparison is used for the main model ranking.

4\. Threshold-selection procedures differ between model families.

5\. Stress tests are simulated distribution changes, not comprehensive validation against production telemetry.

6\. The current results do not establish production readiness or guaranteed performance on AWS.



\## 7. Conclusion and Recommendation



CloudPulse compared XGBoost, a hybrid XGBoost and network-rule detector, LSTM, GRU, an adaptive network detector, and ARIMA.



The CloudPulse V3 Hybrid achieved the highest F1 score in the common synthetic test comparison at 96.14%, with 93.33% recall and one false positive. LSTM achieved higher recall at 97.50%, but generated more false positives.



The hybrid is the strongest current baseline for the recorded synthetic dataset. However, stress testing revealed sensitivity to changing normal traffic, so further validation is required before considering deployment.



Future work should include consistent evaluation splits, validation-only threshold tuning, more diverse synthetic scenarios, per-anomaly-type evaluation, and testing against real cloud telemetry collected with appropriate authorization.



\## 8. Suggested Project-Report Summary



CloudPulse is a prototype cloud-resource anomaly detection system that compares tree-based machine learning, recurrent neural networks, statistical forecasting, and a hybrid detection approach. On a common synthetic test set, its hybrid model achieved an F1 score of 96.14% and recall of 93.33%. Stress tests revealed that changes in normal network traffic can increase false alarms, demonstrating the need for broader validation before real-world deployment.



