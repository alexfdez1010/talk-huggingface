# Talk of Hugging Face in Python Coruña

This repository contains the slides and the demos of the talk of Hugging Face in Python Coruña.

## Slides

You can find the slides at `presentation.pdf`.

## Demos

There are two demos available:

- `qa.py`: Question Answering demo using Streamlit.
- `object_detection.py`: Object Detection demo using Streamlit.

You can find them already deployed in the following links:

- [Question Answering](https://talk-huggingface-question-and-answers.streamlit.app)
- [Object Detection](https://talk-huggingface-object-detection.streamlit.app)

## Installation

you must have a Python version between 3.9 and 3.11. 
The version 3.12 is not supported yet (PyTorch is not available for this version).

To install the dependencies, you can use `pip`:

```bash
pip install -r requirements.txt
```

## Usage

To run the `qa.py` demo, you can use the following command:

```bash
streamlit run qa.py
```

To run the `object_detection.py` demo, you can use the following command:

```bash
streamlit run object_detection.py
```

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.
