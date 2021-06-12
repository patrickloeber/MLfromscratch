# ML algorithms from Scratch!

> Machine Learning algorithm implementations from scratch.

You can find Tutorials with the math and code explanations on my channel: [Here](https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E)

## Algorithms Implemented

- KNN
- Linear Regression
- Logistic Regression
- Naive Bayes
- Perceptron
- SVM
- Decision Tree
- Random Forest
- Principal Component Analysis (PCA)
- K-Means
- AdaBoost
- Linear Discriminant Analysis (LDA)

## Installation and usage.

This project has 2 dependencies.

- `numpy` for the maths implementation and writing the algorithms
- `Scikit-learn` for the data generation and testing.
- `Matplotlib` for the plotting.
- `Pandas` for loading data.

You can install these using the command below!

```sh
# Linux or MacOS
pip3 install -r requirements.txt

# Windows
pip install -r requirements.txt
```

You can run the files as following.

```sh
python -m algorithms.<algorithm-file>
```

with `<algorithm-file>` being the valid filename of the algorithm without the extension.

For example, If I want to run the Linear regression example, I would do 
`python -m mlfromscratch.linear_regression`

**NOTE**: If you want to use the code for any algorithm, that inherits from `BaseAlgorithm` and play with it,
You can do so, without inheriting from it. Just remove `(BaseAlgorithm)` from the class. It is intended just
for Structuring the class and follow a certain rule using `ABCs`.

For example, If you're trying to use Adaboost code standalone, Change the class definition from 
`class Adaboost(BaseAlgorithm):` to `class Adaboost:` and It will work just fine.

## Watch the Playlist

[![Alt text](https://img.youtube.com/vi/ngLyX54e1LU/hqdefault.jpg)](https://www.youtube.com/watch?v=ngLyX54e1LU&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E)
