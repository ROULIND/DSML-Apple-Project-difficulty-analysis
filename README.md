# DSML-Apple-Project-difficulty-analysis



![Image-Groupe-Apple-Presentation.png](https://github.com/ROULIND/DSML-Apple-Project-difficulty-analysis/blob/main/images/Image-Groupe-Apple-Presentation.png)

## 1. Introduction

This project is an academic exploration carried out as part of a Data Science and Machine Learning course. Through the fictional case study of "LingoRank", a startup imagined for the purposes of this project, we immersed ourselves in the real world of data analysis and machine learning applied to linguistics. LingoRank, although a fictional entity, represents an innovative company aiming to transform language learning through technology.

The project simulates the challenge LingoRank could face: developing a model capable of predicting the difficulty of French texts for English-speaking learners. This scenario enabled us to apply theoretical concepts and technical skills in a practical context, while exploring the nuances and challenges of machine learning in the language education sector.


## 2. Objective

The objectives of this project are twofold:

- **Practical Application of Data Science and Machine Learning Skills**: Implement the skills and knowledge acquired in the classroom in a tangible project. This includes data processing and analysis, selecting and training machine learning models, as well as evaluating and interpreting results.
- **Simulation of a Real-World AI Challenge**: Understand and solve a complex artificial intelligence problem - classifying the difficulty of texts based on linguistic proficiency levels. This challenge involves a deep understanding of not just machine learning techniques, but also the linguistic and pedagogical aspects of language learning.

Through this project, we aim to demonstrate our ability to translate theoretical skills into practical solutions, while exploring the possibilities and limitations of artificial intelligence in an educational context. LingoRank, albeit fictional, serves as a backdrop for this educational venture, allowing us to apply our skills in a scenario that mimics real-world business and technology challenges.

## 3. Methodology

### 3.1 Data Analysis

Before diving into the modeling phase, it was essential to thoroughly understand and prepare our dataset. The data provided came in a structured format, comprising features that required careful cleaning and exploration to ensure their suitability for the machine learning models we intended to use.

We commenced by cleansing the data, addressing any missing values, anomalies, or inconsistencies that could potentially skew our analysis and model performance. Our exploratory data analysis (EDA) involved statistical summaries and visual inspections to uncover patterns, trends, and correlations within the data. This step was crucial not only to familiarize ourselves with the dataset's characteristics but also to tailor the preprocessing steps that would follow.

Furthermore, the competition on Kaggle specified a particular format for the submission of predictions. This necessitated an additional layer of data formatting to align our model's output with the expected submission structure. We meticulously formatted our prediction outputs to meet the requirements, ensuring compatibility with Kaggle's evaluation system.

Through this rigorous data processing phase, we established a solid foundation for importing the data into our models and leveraging it effectively to generate accurate predictions.

Here the different data we faced :

**training_data.csv** : data used to train our models
| Id | sentence | difficulty |
|:---------:|:---------:|:---------:|
| 0 | Les coûts kilométriques réels peuvent diverger sensiblement des valeurs moyennes en fonction du moye... | C1 |
| 1 | Le bleu, c'est ma couleur préférée mais je n'aime pas le vert! | A1 |
| ... | ... | ... |


**unlabelled_test_data.csv** : data that are used to predict difficulty
| Id | sentence |
|:---------:|:---------:|
| 0 | Nous dûmes nous excuser des propos que nous eûmes prononcés |
| 1 | Vous ne pouvez pas savoir le plaisir que j'ai de recevoir cette bonne nouvelle. |
| ... | ... |


**submission_sample.csv** : data we use for the kaggle's submissions
| Id | difficulty |
|:---------:|:---------:|
| 0 | A2 |
| 1 | C1 |
| ... | ... |



### 3.2 Model Exploration

With our data meticulously prepared and primed for use, we turned our attention to the core of our project: predictive modeling. We began by experimenting with several fundamental machine learning algorithms to establish a baseline for performance. Among the models we tested were Logistic Regression, Multinomial Naive Bayes, and Neural Networks, each offering unique strengths and suited for text classification tasks.

Our initial trials revealed Logistic Regression as the standout model, providing the best balance of accuracy and complexity for our dataset. Its effectiveness in handling linear relationships made it particularly adept at distinguishing between the varying levels of text difficulty, which was the primary challenge of our Kaggle competition.


| Models    | Logistic Regression | Multinomial NB | NeuralNetwork | GDBT | Random Forest |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Precision | 0.4470  |  0.45014 | 0.4354 | x.xxxx | x.xxxx |
| Recall    | 0.4541  |  0.4125  | 0.4062 | x.xxxx | x.xxxx |
| F1-score | 0.4446 | 0.40452 | 0.40527 | x.xxxx | x.xxxx |
| Accuracy | 0.4541 | 0.4125 | 0.4062 | x.xxxx | x.xxxx |

The exploration into Neural Networks entailed an extensive search for optimal hyperparameters. We conducted numerous experiments to fine-tune the architecture, learning rate, batch size, and other key parameters. Despite our efforts, the results from the Neural Networks were not as promising as we had hoped. Not only were they outperformed by the simpler Logistic Regression model, but they also required significantly more computational resources. The increasing demand for processing power and time rendered the Neural Network approach less feasible for our project's scope.

The journey through model selection taught us valuable lessons about the trade-offs between model complexity and practical constraints such as computational efficiency and time management. Our experience underscored the importance of choosing the right tool for the task at hand, leading to a more focused and resource-conscious approach in future machine learning endeavors.


### 3.3 Advancing to a More Sophisticated Model

Upon recognizing the limitations of our initial models, it became clear that we needed to pivot our strategy towards a more advanced solution, one that was inherently designed for text comprehension and natural language processing tasks. Our quest led us to explore the realm of transformer-based models, which have recently set new benchmarks in the field of language understanding.

It was in this context that we discovered CamemBERT, a state-of-the-art language model specifically trained on a diverse French corpus. Given the nature of our task—predicting the difficulty of French texts—CamemBERT emerged as a particularly well-suited candidate. Its robust training and ability to capture the nuances of the French language made it an ideal choice for enhancing our text classification capabilities.

CamemBERT's transformer architecture, with its attention mechanisms, offered a significant leap forward from traditional machine learning models. It could contextualize words in a sentence more effectively, leading to more nuanced predictions and a deeper understanding of text complexity.

Integrating CamemBERT into our workflow marked a turning point in our project. We adapted our data preprocessing to accommodate the input requirements of the model and fine-tuned CamemBERT on our dataset, eagerly anticipating the impact it would have on our submission's performance in the Kaggle competition.



### 3.4 Data Augmentation

### 3.5 Potential Improvement


## 4. Conclusion

## 5. Team

![](https://github.com/ROULIND/DSML-Apple-Project-difficulty-analysis/blob/main/images/Dimitri-Roulin-PP.jpg)

Dimitri Roulin
Étudiant - Master en Systèmes d'information et Innovation digitale

