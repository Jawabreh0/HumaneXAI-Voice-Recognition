# HumanexAI Voice Recognition
This is a voice recognition system that uses machine learning techniques to identify the identity of a speaker based on their voice. It is built using Python and the Resemblyzer library, and uses a pretrained deep learning model to extract voice embeddings and a random forest classifier to compare them with registered embeddings.

## Features

* Voice embeddings: The system uses a pre-trained deep learning model from the Resemblyzer library to extract voice embeddings, which are compact numerical representations of the speaker's voice. These embeddings capture the unique characteristics of the speaker's vocal cords and speech patterns, and can be used for identification and verification purposes.

* Random forest classifier: The system uses a random forest classifier to compare the extracted embeddings with a set of registered embeddings, and identify the closest match. The classifier is trained on a set of voice recordings, split into embeddings and validation sets, and optimized for accuracy and performance.

* Flexible registration: The system allows users to register their voice by recording a set of phrases and extracting the corresponding embeddings. These registered embeddings can then be stored in a database or file for later comparison.

## Getting started

To use the voice recognition system, you will need:

* Python 3.6 or higher
* The Resemblyzer library and its dependencies (listed in the requirements.txt file)
* A set of voice recordings for training and validation

To get started, you can follow these steps:

* Clone the voice recognition repository to your local machine.
* Install the necessary Python packages by running pip install -r requirements.txt in your terminal or command prompt.
* Prepare a set of voice recordings for training and validation, and split them into embeddings and validation sets (using the split_dataset.py script provided).
* Register your voice by recording a set of phrases and extracting the embeddings, using the get-embeddings.py script.
* Test the system by running python identify.py and providing a voice recording to identify.
