Certificate-Based Drift Detection Audit for Time-Series Forecasting (Electricity Demand × Weather)

Keywords: drift detection, time-series forecasting, model monitoring, MLOps, audit trail, reproducibility, accountability, electricity demand forecasting

<img src="certificate-drift-audit-flow.png" width="600" alt="Certificate–Ledger–Verifier Flow">

Protocol overview: fixed certificate → append-only ledger → independent verifier (OK / NG).

ai-drift-detector (Ghost Drift Audit v9.9) is a certificate-based audit engine designed to prioritize validity and accountability over mere statistical accuracy. Unlike conventional monitoring that relies on post-hoc threshold tuning, this engine outputs a verifiable certificate and an immutable ledger, allowing any third party to reproduce the exact audit verdict from the same inputs.

Note: Bundled CSVs are reproducibility datasets provided to verify the audit protocol. The system is designed with strict data binding; it will cease execution if the input integrity or logic identity does not match the predefined fingerprints.

---

🔗 Quick Links

* 📂 **Source Code:** [GitHub Repository](https://github.com/GhostDriftTheory/ai-drift-detector)
* 📜 **Main Script:** [ghost_drift_audit_EN.py](https://github.com/GhostDriftTheory/ai-drift-detector/blob/main/ai-drift-detector.py)
* 📦 **Download:** [Project ZIP](https://github.com/GhostDriftTheory/ai-drift-detector/archive/refs/heads/main.zip)
* 📖 **Documentation:** [Online Manual](https://ghostdrifttheory.github.io/ai-drift-detector/) ([⚙️ Jump to Execution Mode](https://ghostdrifttheory.github.io/ai-drift-detector/#profile))
* 🚨 **Support:** [Report Issues](https://github.com/GhostDriftTheory/ai-drift-detector/issues)

---

## 📑 Audit Report (PDF)

- **Report:** [Scientific Audit Report on Structural Integrity of Forecasting Models](./Scientific%20Audit%20Report%20on%20Structural%20Integrity%20of%20Forecasting%20Models.pdf)
- **Verdict:** NG (TAU_CAP_HIT)
- **Protocol:** Ghost Drift Audit v8.0

---

💎 Design Philosophy: From "Accuracy" to "Validity"

To address the “opaque inference” problem in conventional AI operations, this framework shifts the focus from probabilistic estimation to accountable verification.

[!TIP]
Deterministic Audit
The engine generates objectively verifiable evidence for third parties, ensuring that the same data and logic always yield the same verdict.

[!IMPORTANT]
Tamper-evident Certificate
It fixes SHA-256 fingerprints of both input data and the execution logic (Logic Identity Proxy), making any unauthorized modifications mathematically detectable.

[!NOTE]
Operational Accountability
Rather than claiming "perfect prediction," it makes visible the model’s faithful adherence to operational rules, such as structural fluctuation limits and physical constraints.

🛠 Technical Specifications

System Requirements

Language: Python 3.10+

Dependencies: numpy, pandas, matplotlib (Calculations are performed via deterministic FFT/NumPy operations)

Project Structure

.
├── ai-drift-detector.py       # Core Logic & Audit Engine (v9.9)
├── electric_load_weather.csv  # Reproducibility Data: Weather
├── power_usage.csv            # Reproducibility Data: Demand
└── audit_bundle.zip           # Output: Accountability Artifacts (Certificate & Ledger)


<a id="profile"></a>

⚙️ Execution Profiles

Switch the strictness of the audit via the configuration settings in ai-drift-detector.py.

Profile

Use / Target

Strictness

Key Features

demo

Protocol verification

Low

Prioritizes understanding audit flow and evidence

paper

Research / reproducible experiments

Mid

Ensures computational reproducibility via fixed seeds

commercial

Production / High-stakes audit

High

Produces strict gate checks (Logic/Source Identity)

How to Configure

# Configuration within ai-drift-detector.py
# v9.9 is pre-configured to handle Logic Identity and BOM Resilience.
STRICT_AUDIT_MODE = True 


🚀 Deployment & Usage

1. Setup
pip install numpy pandas matplotlib

2. Data Preparation
Place the power_usage.csv and electric_load_weather.csv files in the same directory as the .py script.
[!CAUTION]
No Synthetic Fallback: The v9.9 engine prohibits falling back to dummy data in Strict Mode. Use the provided reproducibility datasets or your own audited datasets with valid headers.

3. Run
python ai-drift-detector.py

4. Verification (Outputs)

📜 audit_record.json: The Certificate. A JSON snapshot of execution conditions and logic fingerprints.
📑 audit_log.jsonl: The Ledger. An append-only hash chain recording the full processing history.
📦 audit_bundle.zip: A self-contained package for independent verification.
⚖️ Scope & Integrity (Non-claims)
🎯 Scope & Limits

Scope: Provides a mathematical framework (including Fejér–Yukawa kernel approaches) to make model behavior and structural shifts observable and verifiable.
Non-claims: Does not guarantee zero future error or absolute "truth"; it guarantees the reproducibility of the audit process itself.

🛡️ Threat Model (Tamper Detection)

Threshold manipulation: Detected via Certificate mismatch and signed Cap records.
Logic manipulation: Detected via Logic Identity Proxy hash change.
Data fabrication: Detected via Source Identity (SHA-256) fingerprinting.

📜 License & Acknowledgments

Code: MIT License
Reproducibility Data: Included for protocol verification.

Patent Notice:
This repository implements techniques related to a pending patent application.
Japanese Patent Application No. 特願2025-182213.
This notice does not restrict use of the open-source code under the MIT License.

From “prediction” to “accountability.”
This repository provides a practical reference implementation for certificate-based drift detection and accountable model monitoring.
Produced by GhostDrift Mathematical Institute (GMI) — Official Website | Online Documentation
