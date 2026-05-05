# Embryo Analyzer

Embryo Analyzer is a Streamlit-based clinical decision-support web application for IVF embryo assessment. It provides:

- Single-frame embryo viability classification as `Viable` or `Non-viable`
- Timelapse embryo developmental stage classification
- Confidence visualization and uncertainty flags
- Grad-CAM heatmap explainability
- Clinical questionnaire inputs for contextual interpretation
- Dashboard analytics and downloadable case summary reports

## Features

- Professional dark-themed clinician-friendly interface
- Tabs for `Viability`, `Stage Classification`, and `Analytics`
- Single or batch embryo image upload
- Timelapse frame upload and stage progression review
- Stage guide with developmental stage descriptions
- Clinical decision-support risk indicators
- PDF-style case summary report generation
- Model information panel for research/demo use

## Supported Stage Labels

The stage classification module supports:

- `tPB2` - Second polar body extrusion
- `tPNa` - Pronuclei appear
- `tPNf` - Pronuclei fading
- `t2` - 2-cell stage
- `t3` - 3-cell stage
- `t4` - 4-cell stage
- `t5` - 5-cell stage
- `t6` - 6-cell stage
- `t7` - 7-cell stage
- `t8` - 8-cell stage
- `t9+` - 9+ cell stage
- `tM` - Morula
- `tSB` - Start of blastulation
- `tB` - Blastocyst
- `tEB` - Expanded blastocyst

## Project Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - Excludes cache files and local model weights

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/heyimsaranya/EMBRYOANALYSER.git
cd EMBRYOANALYSER
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model files

Place your trained model weight files in the project folder:

- `best_embryo_model.pth`
- `best_stage_model.pth`

These files are not included in the repository because they are too large for a standard GitHub push.

### 4. Run the application

```bash
streamlit run app.py
```

The app will open in your browser, usually at:

```text
http://localhost:8501
```

## Clinical Use Note

This application is intended as an assistive clinical decision-support and research tool. It is not a standalone diagnostic system and should be used alongside embryologist expertise, laboratory review, and patient-specific clinical context.

## GitHub Repository

Repository link:

[https://github.com/heyimsaranya/EMBRYOANALYSER](https://github.com/heyimsaranya/EMBRYOANALYSER)
