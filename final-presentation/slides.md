---
title: Dynamical system's modelling for tic activity recognition
author: Jules Gottraux
institute: EPFL
date: November 10th, 2020
theme: Boadilla
---
# Introduction
Tic disorders:

- Conditions that induces motor and/or vocal spasms
- Pretty common, especially for young people
- Behavioral therapy as first-line treatment

$\longrightarrow$ Our goal is to facilitate this therapy by automating tic detection

# Introduction
Idea from Joseph F. McGuire and Joey Ka-Yee Essoe. Psychologists specialized in neuropsychiatric conditions such as tic disorder in the childhood.

They contribute by creating the dataset for tic detection.

Creation started in september 2020.

<!-- \vspace{1cm} -->
<!-- \begin{figure} -->
<!--    \includegraphics[width=0.3\textwidth]{figures/joseph.png} -->
<!--    \hspace{2cm} -->
<!--    \includegraphics[width=0.3\textwidth]{figures/joey.png} -->
<!-- \end{figure} -->
\vspace{1cm}
![Setup for dataset](figures/tic-setup.png){width=50%}


# Previous work
Our baseline is an activity recognition framework on pre-segmented dataset using linear dynamical systems. We'll build our methods from there.

\vspace{1cm}
![Screenshot of each task in JIGSAWS dataset](figures/jigsaws-scrsht.png)

# Previous work
Fit a linear dynamical system to the video $\bm Y \in \mathbb{R}^{N\times P},\, \bm Y =\big(y_1,\dots,y_N\big)^T$:

$$y_t = \bm Cx_t$$
$$x_{t+1} = \bm Ax_t$$

The projection is obtained by Principal Component Analysis (PCA) and is fixed. The matrix $\bm A$ is then obtained via:

$$\bm X_- \bm A = \bm X_+$$
$$\Rightarrow \bm A = \bm X_-^\dagger \bm X_+$$

$\bm X$ are the encoded frames and $\bm X_- = \big(x_1,\dots,x_{N-1}\big)^T$, $\bm X_+ = \big(x_2,\dots,x_N\big)^T$

# Outline
Our work consists in improving and adapting this baseline from activity recognition to tic detection and is composed of 4 main sections:

- Baseline extension with non-linear dimensionality reduction
- Baseline extension with joint learning of the dynamical system's components
- Extension of the techniques to online detection
- Test on Hopkins' dataset

All experiments are done on $256\times256$ videos converted to gray scale.

# Non-linear projection
For a video with $N$ frames: $\bm Y \in \mathbb{R}^{N\times P}$. We seek to find a mapping $\bm \Phi_E:\mathbb{R}^P \rightarrow \mathbb{R}^R,\, R \ll P$ such that the frames are \textit{well represented} in the latent space.

We minimize the reconstruction error on a video from Hopkins:

$$\mathcal{L}_{rec} = \frac{1}{N} \sum_{t=1}^{N} \norm{y_t - \bm \Phi_D(\bm \Phi_E(y_t))}_2^2 = \norm{\bm Y-\widehat{\bm Y}}_F^2$$

Where $\widehat{\bm Y}$ is the reconstruction of all frames. And $\bm \Phi_D:\mathbb{R}^R \rightarrow \mathbb{R}^P$ is the decoder, i.e. the inverse mapping.

# Non-linear projection
We compare three models, each one is a neural network with an autoencoder structure:

- ```PCAAE``` (PCA autoencoder): autoencoder using a linear projection
- ```OneHAE``` (one hidden layer autoencoder): autoencoder using a one hidden layer neural network structure
- ```TempConvAE``` (temporal convolutional autoencoder): autoencoder using $3$d (a.k.a. temporal) convolutional layers

# Non-linear projection

![Training error for all temporal convolutional networks, $R=10$](figures/tempconv-rec-errors.png)

# Non-linear projection

![Training error for all models, $R=10$](figures/all-models-rec-errors.png)

# Non-linear projection
Notes:

- Further analysis showed that enforcing the latent dimension with a linear mapping were hurting the model's power
- Better performance could be obtained using known video compression algorithm or network (e.g. U-Net)
- These preliminary results are not conclusive
- We'll stick to linear projection

# Validation and joint learning assessment
Initial goals with this dataset:

- Measure the capability of the method to detect an activity on raw frames
- Assess whether learning $\bm A$ and $\bm C$ jointly helps the algorithm

Instead of fixing the projection and obtaining $\bm A$ as in the baseline, we initialize the models with the parameters from the baseline and minimize:

\begin{align*}
	\mathcal{L}_{pred} &= \frac{1}{N-1} \sum_{t=2}^{N} \norm{y_t - \bm \Phi_D(\bm A\bm \Phi_E(y_{t-1}))}_2^2 \\
		&= \norm{\bm Y_+ - \bm \Phi_D(\bm \Phi_E(\bm Y_-)\bm A)}_F^2
\end{align*}

$\bm \Phi_E$ and $\bm \Phi_D$ as before are the mapping to and from the latent space.

# Validation and joint learning assessment
Evaluation: 

- We have fragments of videos, each with a given activity and fitted dynamical system
- Classification task, the features are the dynamical system's matrices and the labels are the corresponding activity

The classification uses metrics based on the subspace angles between dynamical models. These metrics are the Frobenius and Martin distance.

These metrics tells us how much the dynamics of two different fragments differ.

On top of these metrics, we use K-Nearest Neighbors (KNN) or Support Vector Machine with radial basis function kernel (SVM) for the classification

<!-- For the evaluation, every dynamical system corresponds to a fragment of video with a single activity inside. We want to classify the activity of each fragments using metrics on dynamical systems. The metrics used are distances based on the subspace angles between matrices of dynamical systems. Hence our metrics measure the --> 

# Validation and joint learning assessment

Results for one task (suturing):

\begin{figure}
   \raisebox{-0.5\height}{\includegraphics[width=0.35\textwidth]{figures/jigsaws-eval-separated.png}}
   \raisebox{-0.5\height}{\includegraphics[width=0.45\textwidth]{figures/joint-vs-separated.png}}
\end{figure}

# Validation and joint learning assessment
Notes:

- Joint learning performance is indistinguishable from separated learning. Hence, separated learning is nearly optimal for this framework
- The model's performance are accurate enough for our detection task

# Extension to online detection
We test two approaches for online detection:

- Detection based on the reconstruction error of the prediction of models
- Detection based on the distance between models and a model fitted in a moving window fashion

Experiments are done on a video picked from the JIGSAWS dataset.

# Detection based on reconstruction error
- Pick a particular gesture
- Do the models of this gesture capture the dynamics of the others as well?
- If yes we could use this information to construct an online classifier

\vspace{0.5cm}
![Reconstruction error of the prediction of models from same activity](figures/pred-errors-jigsaws.png){width=70%}

# Detection using Martin distance
- Pick a particular gesture
- Can we detect the occurence of this gesture based on the distance between a moving window model and the fitted models?

\vspace{0.5cm}
![Martin distance between models of the gesture and moving window model](figures/martin-dist.png){width=70%}

# Hopkins dataset
- 3 videos of 15 minutes each
- Very subtle tics
	- Eye blinking tic
	- Nose flare

Two evaluations:

- Binary classification of tic on pre-segmented fragments
- Comparison of tic occurrence with reconstruction error of normal model

# Classification evaluation on Hopkins dataset
Same approach as for JIGSAWS dataset:

- Pre-segmented data
- Using baseline
- KNN or SVM classifier using Martin and Frobenius distance

# Evaluation using reconstruction error
- Fit a dynamical system on all _normal_ frames (no tic)
- Does the reconstruction error and the occurence of tics correlate?

![Reconstruction error of the prediction for normal models](tic-pred-errors.png){width=80%}

# Conclusion
Our experiments indicate that detection of subtle tics is very challenging.

For online detection, creating dynamical models that better captures the general dynamics of the activity is a key ingredient.

Need more data and/or more visible tics to be able to make a proper evaluation.
