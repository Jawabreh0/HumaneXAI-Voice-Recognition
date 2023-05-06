# HumaneXAI-Voice-Recognition
This is a voice recognition system that uses machine learning techniques to identify the identity of a speaker based on their voice. It is built using Python and the Resemblyzer library, and uses a pretrained deep learning model to extract voice embeddings and a random forest classifier to compare them with registered embeddings.

## Features

* Voice embeddings: The system uses a pre-trained deep learning model from the Resemblyzer library to extract voice embeddings, which are compact numerical representations of the speaker's voice. These embeddings capture the unique characteristics of the speaker's vocal cords and speech patterns, and can be used for identification and verification purposes.

* Random forest classifier: The system uses a random forest classifier to compare the extracted embeddings with a set of registered embeddings, and identify the closest match. The classifier is trained on a set of voice recordings, split into embeddings and validation sets, and optimized for accuracy and performance.

* Flexible registration: The system allows users to register their voice by recording a set of phrases and extracting the corresponding embeddings. These registered embeddings can then be stored in a database or file for later comparison.

