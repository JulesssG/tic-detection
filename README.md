# tic-detection
This repository contains all the code, presentations and the report related to a semester project supervised by Benjamin Béjar Haro. This project try to leverage activity recognition techniques based on dynamical system modelling to detect tics on video data. Tic detection could help all people suffering from Tourette disorder or other related pathologies to learn how to prevent the tics and this improve their quality of life. This project has been realised in collaboration with ![Joey Ka-Yee Essoe](https://jhucoach.org/about/essoe/) and ![Joseph McGuire](https://jhucoach.org/about/jfmcguire/) from Hopkins university.

This repository serves also as a ![renku](https://datascience.ch/renku/) template.

## File descriptions
- ![report](./report): the directory with the files for the report of the project, explaining the approach and analyses. The report is ![this file](./report/compile/Dynamical_system_modelling_for_tic_activity_recognition_in_Tourette_disorder.pdf)
- ![midterm_presentation](./midterm_presentation): Slides for the midterm presentation
- ![misc](./misc): Miscellaneous files, including the program used to transfer large files to renku (![0x0](./misc/0x0))

- ![autoencoders.py](./autoencoders.py): The different autoencoders used for dimensionality reduction of the signal
- ![custom_pca](./custom_pca.py): A utility class for a PCA model based on sklearn's randomized_svd class
- ![evaluation_classification_JIGSAWS.ipynb](./evaluation_classification_JIGSAWS.ipynb): Evaluation of the method for classification of videos' fragments activity using the ![JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
- ![jigsaws_utils,py](./jigsaws_utils,py): Utility functions for JIGSAWS dataset
- ![testing_clusters.ipynb](./testing_clusters.ipynb): Analysis of the reconstruction errors between models on the JIGSAWS dataset
- ![tic-detection.ipynb](tic-detection.ipynb): Analyses on the dataset from the team at Hopkins university
- ![utils.py](./utils.py): Utility functions
- ![video_loader.py](./video_loader.py): Utility class for working with videos. It is based on the ![opencv library](https://opencv.org/) and allow easy and efficient iteration and manipulation of videos.

- ![synthetic_avglds.py](./synthetic_avglds.py): Script that minimize the Martin distance between multiple synthetic linear dynamical systems

Renku related files:\\
- ![Dockerfile](./Dockerfile)
- ![requirements.txt](./requirements)
- ![environment.yml](./environment)
- ![manifest.yml](./manifest)
