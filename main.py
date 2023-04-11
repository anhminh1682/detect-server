# FastAPI
from fastapi import FastAPI, File, UploadFile
# FastAI
from fastai.vision.all import *
from PIL import *

# fix path if running on Linux system
import pathlib
plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

app = FastAPI(
    title="detect",
    version=0.1,
    root_path="/"
)
# Declare variables
path = Path("./")
learn_inf = load_learner(path/"export.pkl", cpu=True)
image_path = "skin_type/Test/"


@app.get("/", tags=["FastAPI"])
def root():
    return {"message": "hello world again"}


@app.post("/")
async def root(image: UploadFile = File('')):
    img = PILImage.create(await image.read())
    pred, pred_idx, probs = learn_inf.predict(img)
    rate = probs[pred_idx]
    return {
        "response": {
            "pred": pred,
            "pred_idx": f"{probs[pred_idx]}"
        }
    }


@app.get('/refresh')
def root():
    torch.cuda.empty_cache()
    return {
        "sucess": bool(1)
    }
