import json
import codecs

# Read the pipeline.py file so we can split it into cells
with open("src/pipeline.py", "r") as f:
    pipeline_code = f.read()

# Let's split it roughly into:
# 1. Imports
# 2. Class definition (up to run method)
# 3. Class run method
# 4. Main block

# We'll just put the whole class in one cell for simplicity, imports in another, and main block in another.
parts = pipeline_code.split("class TaskEEGPipeline:")
imports_part = parts[0]
rest_of_code = "class TaskEEGPipeline:" + parts[1]

class_and_main = rest_of_code.split("if __name__ == \"__main__\":")
class_part = class_and_main[0]
main_part = "if __name__ == \"__main__\":" + class_and_main[1]

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Detached EEG Pipeline (Standalone)\n",
            "\n",
            "This notebook contains the full pipeline code from `src/pipeline.py` and is ready to be run on Google Colab. Make sure you select a **GPU runtime**."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Mount Google Drive if you have uploaded your dataset here\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Change to the directory where your detachment-eeg code and data live\n",
            "import os\n",
            "os.chdir('/content/drive/MyDrive/detached-eeg')\n",
            "print(\"Working directory:\", os.getcwd())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install Dependencies\n",
            "!pip install mne pyyaml eegdash pyts sktime==0.30.0\n",
            "!pip install git+https://github.com/gon-uri/detach_rocket"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Imports and Setup"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line for line in imports_part.splitlines(True)]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Pipeline Class Definition"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line for line in class_part.splitlines(True)]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Run the Pipeline"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line for line in main_part.splitlines(True)]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "pipeline_standalone.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with codecs.open("notebooks/pipeline_standalone.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Created standalone pipeline notebook: notebooks/pipeline_standalone.ipynb")
