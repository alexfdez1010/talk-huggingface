from dataclasses import dataclass
from typing import Tuple, Generator

import streamlit as st
import torch
import torchvision
from PIL import Image, ImageDraw
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import YolosImageProcessor, YolosForObjectDetection


@dataclass
class Model:
    processor: YolosImageProcessor
    detector: YolosForObjectDetection


@dataclass
class Box:
    score: float
    label: str
    box: Tuple[float, float, float, float]


MODEL_ID = "hustvl/yolos-tiny"


def get_model() -> Model:
    """
    Get the model used to detect objects
    :return: the model
    """

    if "model" not in st.session_state:
        yolos = YolosForObjectDetection.from_pretrained(MODEL_ID)
        yolos_image_processor = YolosImageProcessor.from_pretrained(MODEL_ID)
        st.session_state.model = Model(yolos_image_processor, yolos)

    return st.session_state.model


def get_predictions(image: UploadedFile, threshold: float) -> Generator[Box, None, None]:
    """
    Get the predictions for an image
    :param image: the image in bytes
    :param threshold: the threshold
    :return: the boxes
    """
    bytes_data = image.getvalue()
    torch_img = torchvision.io.decode_image(
        torch.frombuffer(bytes_data, dtype=torch.uint8)
    )

    model = get_model()
    torch_img = torch.unsqueeze(torch_img, dim=0)

    tensor = model.processor(torch_img, return_tensors="pt")
    boxes = model.detector(**tensor)

    target_size = torch.tensor([torch_img.shape[2:]])
    # noinspection PyTypeChecker
    results = model.processor.post_process_object_detection(boxes, threshold=threshold, target_sizes=target_size)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = score.item()
        label = model.detector.config.id2label[label.item()]
        box = tuple(box.tolist())

        yield Box(score, label, box)


def draw_boxes(image: UploadedFile, boxes: Generator[Box, None, None]) -> Image:
    """
    Draw rectangles in the image in the corresponding boxes
    :param image: the image
    :param boxes: the boxes
    """
    img = Image.open(image)
    draw_image = ImageDraw.Draw(img)

    for box in boxes:
        draw_image.rectangle(box.box, outline="blue", width=3, fill=None)
        label = f"{box.label} - {box.score*100:.2f}"
        draw_image.text((box.box[0], box.box[1] - 15), label, fill="blue")

    return img


def main():
    st.set_page_config(
        page_title="Object Detection",
        page_icon=":camera:",
        layout="wide"
    )

    st.title("Object Detection")
    st.text("This is an example of a object detection system using a Yolos model.")

    st.header("Image")
    image = st.camera_input("Take a photo")

    st.header("Threshold")
    threshold = st.slider('Select a threshold', 0.0, 1.0, 0.5, 0.01)

    if image is None:
        return

    boxes = get_predictions(image, threshold)
    draw_image = draw_boxes(image, boxes)
    st.image(draw_image)


if __name__ == '__main__':
    main()
