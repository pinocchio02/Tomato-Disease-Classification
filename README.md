# 🍅 Tomato Disease Detection App

This project is a full-stack machine learning application that detects tomato plant diseases from images using a trained deep learning model. It includes:

- 🔍 A **FastAPI backend** for local inference
- 🌐 A **React frontend** for user interaction
- ☁️ **Deployment** on Google Cloud Functions
- 🧠 A **trained CNN model** using TensorFlow
- 🧪 Tools and datasets used during **model training**

---

## 📂 Project Structure

## 🚀 How to Run Locally

### 1. Backend (FastAPI)
1. Navigate to the backend folder:

    ```bash
    cd api/
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Run the Flask server:

    ```bash
    flask run
    ```

---
### 2. Frontend (React)

1. Navigate to the frontend folder:

    ```bash
    cd frontend/
    ```

2. Install dependencies:

    ```bash
    npm install
    ```

3. Create a `.env` file and add:

    ```env
    REACT_APP_API_URL=http://localhost:8000/predict
    ```

4. Start the development server:

    ```bash
    npm run start
    ```

---
## 3. Deployment (Google Cloud)

You can deploy the backend using **Google Cloud Functions**:

### 🔧 Deployment Command

```bash
gcloud functions deploy predict \
  --runtime python310 \
  --trigger-http \
  --memory 512MB \
  --timeout 540s \
  --allow-unauthenticated \
  --project <your-project-id>
 ```

## 4. Model Details

- Framework: TensorFlow/Keras
- Architecture: CNN
- Classes:
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites
  - Target Spot
  - Yellow Leaf Curl Virus
  - Mosaic Virus
  - Healthy

---
## 5. Tech_Stack:
  Backend:
    - Flask (Local)
    - Google Cloud Functions (Production)
  Frontend:
    - React (JavaScript)
  Machine_Learning:
    - TensorFlow
    - Pillow
    - NumPy
  Deployment:
    - Google Cloud Functions
  Tools:
    - Git
    - GitHub

## 6. Features:
  - Upload tomato leaf image via UI or Postman
  - Disease class prediction using deep learning
  - Lightweight and modular codebase
  - Cloud-native serverless deployment
  - Clean separation between frontend and backend

## Author:
  Name: Om
  Role: AI Developer & ML Enthusiast
  GitHub: https://github.com/pinocchio02







