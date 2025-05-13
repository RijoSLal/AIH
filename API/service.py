import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from fastapi import FastAPI,Request,Response,UploadFile,File
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
from ultralytics import YOLO 
from huggingface_hub import hf_hub_download
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from mlflow_models import Server_models
import tensorflow as tf

app = FastAPI()

templates=Jinja2Templates(directory="templates")

repo = "arnabdhar/YOLOv8-Face-Detection"

model_path = hf_hub_download(repo_id=repo, filename="model.pt")

yolo = YOLO(model_path)



models = Server_models() 
age_model = models.model_retriever("age_model",1)
gender_model = models.model_retriever("gender_model",1)
live_model = models.model_retriever("live_model",1)

AGE = ["MIDDLE","OLD","YOUNG"]
GENDER = ["FEMALE","MALE"]
LIVE = ["LIVE","SPOOF"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def frontend_loading(request: Request) -> Response:

    """
    Handles GET requests to the root endpoint and renders the 'AIH.html' frontend template.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        Response: A TemplateResponse object rendering the 'AIH.html' template.
    """
    return templates.TemplateResponse(
        request=request, name="AIH.html"
    )




@app.post("/images")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Handles POST requests to '/images' by reading and processing the uploaded image,
    detecting a face, and running predictions for age, gender, and liveness.

    Args:
        file (UploadFile): The image file uploaded by the client.

    Returns:
        JSONResponse: A JSON response indicating whether a human face was detected and
                      including dummy placeholders for age, gender, and liveness.
    """
    image_data = await file.read()
    np_arr = np.frombuffer(image_data, np.uint8)
 
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    return detect_and_crop_face(image)
    


def detect_and_crop_face(image: np.ndarray) -> JSONResponse:
    """
    Detects a face in the image using a YOLO model, crops it, preprocesses it, and
    runs predictions for age, gender, and liveness using preloaded models.

    Args:
        image (np.ndarray): The decoded input image as a NumPy array.
        save_path (str): Path where the cropped face image will be saved.

    Returns:
        JSONResponse: A JSON response indicating whether a human was detected and
                      predicted values (placeholders shown here) for age, gender, and liveness.
    """
    if image is None:
        raise ValueError("Failed to load the image. Please ensure the file is a valid image format.")
    
    results = yolo.predict(image)

    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.4:
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_face = image[y1:y2, x1:x2]
                image=image_preprocessing(cropped_face)
                age_pred = np.argmax(age_model.predict(image))
                gender_prob = tf.math.sigmoid(gender_model.predict(image)[0][0])
                gender_pred = int(gender_prob > 0.5)
                live_prob = tf.math.sigmoid(live_model.predict(image)[0][0])
                live_pred = int(live_prob > 0.5)
             
                return JSONResponse(content=
                                {"human":"YES",
                                 "age": AGE[age_pred], 
                                 "gender":GENDER[gender_pred],
                                 "real":LIVE[live_pred]})
          
    return JSONResponse(content={"human":"no"}) 


def image_preprocessing(img: np.ndarray) -> tf.Tensor:
    """
    Preprocesses the input image for model prediction. Resizes, normalizes, and
    expands dimensions to fit expected model input.

    Args:
        img (np.ndarray): The cropped face image as a NumPy array.

    Returns:
        tf.Tensor: A preprocessed image tensor ready for prediction.
    """
    img = image.img_to_array(img)                         
    img_resized = tf.image.resize(img, [224, 224])     
    img_ready_exp = tf.expand_dims(img_resized, axis=0)      
    img_ready = preprocess_input(img_ready_exp)  
    return img_ready

