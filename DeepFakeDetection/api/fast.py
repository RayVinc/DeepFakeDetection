from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import numpy as np
from DeepFakeDetection.dl_logic.model import load_model, predict
from PIL import Image
import io


app = FastAPI()

# Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
 )
app.state.model = load_model()


@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):

    # Receiving and decoding the image
    contents = await img.read()
    image = Image.open(io.BytesIO(contents))

    # Resizing the image
    image_size = (128, 128)

    image_resized=image.resize(image_size)
    #print(f'😋{type(image_resized) = }')

    image_np = np.array(image_resized)
    #print(f'🍑{image_np.shape = } {type(image_np) = }')

    image_scaled_down=image_np/255
    #print(f'😃{image_scaled_down.shape = } {type(image_scaled_down) = }')

    image_preproc = np.expand_dims(image_scaled_down, axis=0)
    #print(f'😍{image_preproc.shape = } {type(image_preproc) = }')

    pred = predict(app.state.model, image_preproc)
    print(pred)

    return pred
