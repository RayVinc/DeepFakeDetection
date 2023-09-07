from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
import io
from DeepFakeDetection.dl_logic.model import load_model, predict, compile_model

app = FastAPI()


# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

app.state.model = load_model()


@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    print(cv2_img.shape)
    image_size = (256, 256, 3)

    image= np.resize(cv2_img,image_size)
    #image= np.reshape(cv2_img,image_size)
    print(image.shape)
    #model = compile_model(app.state.model)

    pred = predict(app.state.model, image)
    print(pred)
    ### Do cool stuff with your image.... For example face detection

    ### Encoding and responding with the image
   #im = cv2.imencode('.png', annotated_img)[1] # extension depends on which format is sent from Streamlit
   # return Response(content=im.tobytes(), media_type="image/png")
    return pred
