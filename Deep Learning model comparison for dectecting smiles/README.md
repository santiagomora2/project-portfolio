# Deep learning model comparison for detecting smiles [2024]

The effectiveness of two deep learning approaches—Convolutional Neural Networks (CNNs) and autoencoders— was explored for classifying whether a person is smiling in an image. The CNN was designed as a supervised learning model to directly classify smiles, while the autoencoder was employed for anomaly detection by identifying deviations from smiling baselines. 

The autoencoder was trained ion smiling-only data, and the Wasserstein distance was used to measure reconstruction errors in the training set to set a threshold to classify an image as smiling or non-smiling.

In the end, the CNN significantly outperformed the autoencoder, achieving higher accuracy and average recall across all classes. This result highlights the CNN's effectiveness in feature extraction and classification tasks, making it a more reliable solution for smile detection.
