# AIH (AM I HUMAN?)

**"Are you a human, or have the aliens taken over?"**
👽 Inspired by the Men in Black series, AIH is a fun and quirky project designed to determine if you're truly human or if you're just a cleverly disguised extraterrestrial. With a mix of modern AI techniques, this tool can assess not only your human status but also your age, gender, whether you're alive, or if you might just be a brilliant spoof. Don't worry, I've got you covered!

---

## ⚙️ **Technologies Used**

* **Transfer Learning (MobileNetV3)**: Using the MobileNetV3 model for efficient image classification.
* **FastAPI**: Backend for handling requests and serving the model efficiently.
* **DVC (Data Version Control)**: For managing model data and tracking changes.
* **MLflow**: To manage the entire machine learning lifecycle including experimentation, model tracking, and deployment.
* **YOLO (You Only Look Once)**: For object detection, especially detecting faces to evaluate human-like features.
* **Dagshub**: Cloud-based model and data management system.
* **Jinja**: For dynamically generating HTML templates to serve results.
* **Uvicorn**: ASGI server for running the FastAPI app.

---

## 🧠 **How It Works**

1. **Model Training**: Fine-tune a MobileNetV3 model using a dataset of human images. The model is trained to classify various features like age, gender, and whether the input appears human.

2. **FastAPI Backend**: FastAPI is used to build a lightweight API that takes an image as input and returns predictions (Human Status, Age, Gender, etc.). It’s fast, lightweight, and super efficient.

3. **Model Management in the Cloud**:

   * The model is completely managed in the cloud via **Dagshub**.
   * Every time the server (FastAPI) restarts, the model is fetched from the cloud and loaded for prediction.
   * **DVC** is used for version control to manage and track data and model changes.
   * **MLflow** is used to track the machine learning lifecycle, from experimentation to deployment.

4. **Prediction Process**: When you upload an image, YOLO detects if the face is human, and the MobileNetV3 model classifies it into human categories. The result is served via a friendly UI built using Jinja templates.
---

## 🎉 **Fun Features**

* **Human Detection**: The AI will tell you if you're human or just pretending to be one!
* **Age & Gender Prediction**: Get an estimate of your age and gender based on the image you provide.
* **Spoof Detection**: Using the techniques to detect spoof faces or avatars.
* **Interactive UI**: Dynamic, fun, and interactive web interface built with Jinja templates.

---

## 💻 **Installation and Setup**

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/AIH.git
cd AIH
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI App

```bash
uvicorn service:app --reload
```

### 4. Start the App

Open your browser and go to `http://127.0.0.1:8000/` to interact with the app!

---

## ⚙️ **Training the Model**

To train the model, follow the steps below:

1. **Run the `model.ipynb`** to train and fine-tune the MobileNetV3 model on your dataset.

   ```bash
   jupyter notebook model.ipynb
   ```

   The model will be trained using the transfer learning approach, leveraging the MobileNetV3 pre-trained weights.

2. **Register the Model on the Server**: Once training is complete, use the `model_registration.ipynb` to register the trained model in the server. This will save the model on **Dagshub** for cloud management.

   ```bash
   jupyter notebook model_registration.ipynb
   ```

   This script will upload the model to the cloud and register it for future use, ensuring that it's available whenever the server restarts.

---

## 🌩️ **MLOps Workflow with Dagshub**

The model and data are managed in the cloud via **Dagshub**, ensuring that:

1. **Version Control**: All models and datasets are versioned using **DVC**.
2. **Tracking**: The entire machine learning workflow, including experimentation, is tracked using **MLflow**.
3. **Cloud-based Deployment**: Whenever the FastAPI server restarts, the model is automatically fetched from **Dagshub** and deployed without any additional setup.

💡 Change Your Dagshub Repo Name
Don’t forget to update the Dagshub remote URL in .dvc/config and .mlflow tracking if you fork this project or rename it:

**Example**:

```bash
dvc remote modify origin url https://dagshub.com/your-username/your-repo-name.git
```

## 🚀 **Deploying the Model**

1. **Run the FastAPI Server**: Once the model is registered, the server can be started using **Uvicorn** to serve the application.

   ```bash
   uvicorn main:app --reload
   ```

2. **Access the API**: Open your browser and go to `http://127.0.0.1:8000/` to interact with the fun human-detection tool.

---

## 🧪 **Contributing**

Since this is a solo project, contributions are welcome but not expected. However, if you think of any fun ideas, bug fixes, or improvements, feel free to fork the repo and submit a pull request!

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Note:**

No alien or spoofed face will get past my detectors! 😉

---