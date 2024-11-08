# ASL Fingerspelling Recognition

This repository contains an implementation of the [winning solution](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485) of the kaggle ASL Fingerspelling Recognition competition. The purpouse is learning and having fun, as always :) .

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Validation](#validation)
- [Data Extraction](#data-extraction)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ASL-Fingerspelling-Recognition.git
    cd ASL-Fingerspelling-Recognition
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the following command:
```bash
python scripts/train.py
```

### Data Extraction

To extract data from the raw dataset, run:
```bash
python scripts/data_extraction.py
```
You will need to [download the data](https://www.kaggle.com/competitions/asl-fingerspelling/data) by yourself and merge train and supplemental csv's.

## Project Structure

```
ASL-Fingerspelling-Recognition/
│
├── data/
│   ├── supplemental_landmarks/
│   ├── train_landmarks/
│   ├── merged.csv
│
├── scripts/
│   ├── model/
│   │   ├── model.py
│   │   ├── encoder.py
│   │   └── decoder.py
│   ├── data.py
│   ├── data_extraction.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

## Model Architecture

The model consists of several components:

### Feature Extraction

The `FeatureExtraction` class in `model.py` is responsible for extracting features from the input data. It uses convolutional layers followed by batch normalization and linear layers.

### Encoder

The `SqueezeformerBlock` class in `encoder.py` implements the Squeezeformer architecture, which includes multi-head attention, feed-forward layers, and convolutional modules.

### Decoder

The `Decoder` class in `decoder.py` uses the `Speech2TextDecoder` from the Hugging Face Transformers library to decode the encoded features.

## Training

The training script (`train.py`) handles the training process. It includes functions for loading data, setting up the model, and training the model over multiple epochs.

### Learning Rate Schedule

The learning rate is adjusted using a cosine annealing schedule with warmup.

### Gradient Clipping

Gradients are clipped to prevent exploding gradients during training.

## Validation

The validation script (`validate.py`) evaluates the model on the validation dataset. It calculates the loss and prints sample predictions.

## Data Extraction

The data extraction script (`data_extraction.py`) processes the raw dataset and saves the processed data in a format suitable for training and validation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.