import cv2
import numpy as np
from torch.utils.data import Dataset

import torch
from django.conf import settings
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

model = settings.DFU_MODEL
saved_state_dict = torch.load(
    glob.glob(f"{settings.BASE_DIR}/main/*.pth")[0], map_location=torch.device("cpu")
)
model.load_state_dict(saved_state_dict, strict=False)


def preprocess_image(media_path: str):
    img = cv2.imread(f"{settings.BASE_DIR}/{media_path}")
    print(img)
    try:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        max_contour = max(contours, key=cv2.contourArea)
    except:
        img = np.clip(img * (4 / 3), 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        try:
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            max_contour = max(contours, key=cv2.contourArea)
        except:
            resize = cv2.resize(img, (224, 224))
            return resize
    x, y, w, h = cv2.boundingRect(max_contour)
    crop_img = img[y : y + h, x : x + w]
    resize = cv2.resize(crop_img, (224, 224))
    return resize


class TestDataset(Dataset):
    def __init__(self, image_paths, transform=False) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def predict(self):
        for i in range(10):
            if len(glob.glob(self.image_paths)) != 0:
                break
            else:
                print(i)
        image_filepath = self.image_paths
        # if self.transform is not False:
        #     image = cv2.imread(image_filepath)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     image = self.transform(image=image)["image"]
        # else:
        # image = cv2.imread(f"{settings.BASE_DIR}/{self.image_paths}")
        image = preprocess_image(image_filepath)
        image = self.transform(image=image)["image"]
        return image, image_filepath


def getting_model(model_path: str = "main/DFU_pasti.pth"):
    model = settings.DFU_MODEL
    saved_state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    model = model.load_state_dict(saved_state_dict, strict=False)
    return model


def predict(img_file):
    transforms = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.299, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    data = TestDataset(img_file, transforms)
    tensor, _ = data.predict()
    tensor = tensor.to(torch.float32)
    yhat = model(tensor.unsqueeze(0))
    yhat = yhat.clone().detach()
    yhat = torch.argmax(yhat, dim=1)
    return yhat
