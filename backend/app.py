from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import imutils

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the face detector model
prototxtPath = "facedetector/deploy.prototxt"  # Correct for macOS or Linux
weightsPath = "facedetector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the mask detection model
maskNet = load_model("MSN_maskdetector.keras")


class ImageData(BaseModel):
    image: str  # Base64-encoded image


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


@app.post("/detect-mask/")
async def detect_mask(data: ImageData):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data.image.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Resize and process the frame
        frame = imutils.resize(image, width=400)

        # Get the face locations and predictions
        locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

        results = []
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Draw the bounding box and label on the frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Convert NumPy int64 to native Python int
            results.append({
                # Ensure the box values are native int
                "box": [int(startX), int(startY), int(endX), int(endY)],
                "label": label
            })

        # Convert the processed image back to base64 to return as response
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={"results": results, "image": base64_image})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
