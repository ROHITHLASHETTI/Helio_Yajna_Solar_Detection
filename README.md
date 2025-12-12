# [PROJECT NAME] - Team_HELIO_YAJNA_Solar_Detection

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Docker](https://img.shields.io/badge/Docker-Enabled-blue) ![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)

## 📌 Project Overview
This project implements an automated machine learning pipeline that identifies specific objects in satellite imagery. It accepts geospatial coordinates (Latitude/Longitude), fetches high-resolution satellite images via the Google Maps API, and processes them using a custom-trained **YOLOv8** model to detect and highlight features.

### Key Features
*   Automated Data Gathering: Fetches static map images based on coordinate inputs.
*   Computer Vision:Uses Ultralytics YOLOv8 for object detection.
*   Dockerized:Fully containerized environment ensuring zero dependency conflicts.
*   Batch Processing:Handles Excel (`.xlsx`) inputs for bulk analysis.

---

## 🛠️ Prerequisites
To run this project, you need:
1.  **Docker Desktop** installed and running.
2.  A valid **Google Maps Static API Key**.

---

## ⚙️ Setup & Configuration

### 1. Clone the Repository
```bash
git clone https://github.com/ROHITHLASHETTI/Helio_Yajna_Solar_Detection
cd team_heli_yajna_eco_ideathon
2. Configure Credentials (Important!)
Create a file named .env in the root directory.
Paste your API key inside it without quotes.
Example .env content:
code
Properties
GOOGLE_API_KEY=Your_Actual_Key_Here
3. Prepare Input Data
Ensure your input Excel file is placed in the input_data folder.
File Path: input_data/input.xlsx
(A sample file is included in the repository for testing).
🚀 How to Run (Evaluation Guide)
We recommend using Docker for the smoothest experience. You can either pull the pre-built image or build it locally.
Option A: Quick Run (Using Pre-built Image)
Use this if you want to test immediately without building.
code
Powershell
docker run --env-file .env `
  -v "${PWD}/input_data:/app/input_data" `
  -v "${PWD}/output_data:/app/output_data" `
  rohithlashetti03/helio_yajna_solar_detection:v1
Option B: Build & Run Locally
Use this if you want to modify the code.
1. Build the Image:
code
Bash
docker build -t helio_yajna_solar_detection.
2. Run the Container (Windows PowerShell):
code
Powershell
docker run --env-file .env `
  -v "${PWD}/input_data:/app/input_data" `
  -v "${PWD}/output_data:/app/output_data" `
  helio_yajna_solar_detection
(Note: If using Mac/Linux, replace ${PWD} with $(pwd) and backticks ` with backslashes \)
📂 Project Structure
code
Text
├── input_data/          # Place input .xlsx files here
├── output_data/         # Results (Images/Logs) appear here after running
├── pipeline/            # Source code
│   └── main.py          # Main execution script
├── weights/             # Trained YOLOv8 models
│   └── best.pt          # Best model weights
├── environment/         # Configuration files
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
📊 Output
Once the script finishes, navigate to the output_data/ folder. You will find:
Detected Images: Satellite images with bounding boxes drawn around detected objects.
Logs: Execution details and any errors encountered during processing.
🐛 Troubleshooting
1. "Missing GOOGLE_API_KEY" Error
Ensure the .env file exists in the root folder.
Ensure you passed --env-file .env in the docker run command.
2. "Input file not found"
Check that input.xlsx is inside the input_data folder.
Ensure the volume mount -v ... path in the docker command is correct.
3. "403 Forbidden" (Google API)
Open your .env file and ensure there are no quotation marks around your key.
Correct: KEY=App... | Incorrect: KEY="Ap..."