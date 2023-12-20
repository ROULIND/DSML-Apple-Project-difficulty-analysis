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


| Models    | Logistic Regression | Multinomial NB | NeuralNetwork | CamenBERT |
|:---------:|:---------:|:---------:|:---------:|:---------:|
| Precision | 0.4470  |  0.45014 | 0.4354 | **0.9520** | 
| Recall    | 0.4541  |  0.4125  | 0.4062 | **0.9522** | 
| F1-score | 0.4446 | 0.40452 | 0.40527 | **0.9520** |
| Accuracy | 0.4541 | 0.4125 | 0.4062 | **0.9518** | 


The exploration into Neural Networks entailed an extensive search for optimal hyperparameters. We conducted numerous experiments to fine-tune the architecture, learning rate, batch size, and other key parameters. Despite our efforts, the results from the Neural Networks were not as promising as we had hoped. Not only were they outperformed by the simpler Logistic Regression model, but they also required significantly more computational resources. The increasing demand for processing power and time rendered the Neural Network approach less feasible for our project's scope.

The journey through model selection taught us valuable lessons about the trade-offs between model complexity and practical constraints such as computational efficiency and time management. Our experience underscored the importance of choosing the right tool for the task at hand, leading to a more focused and resource-conscious approach in future machine learning endeavors.


### 3.3 Advancing to a More Sophisticated Model

Upon recognizing the limitations of our initial models, it became clear that we needed to pivot our strategy towards a more advanced solution, one that was inherently designed for text comprehension and natural language processing tasks. Our quest led us to explore the realm of transformer-based models, which have recently set new benchmarks in the field of language understanding.

It was in this context that we discovered CamemBERT, a state-of-the-art language model specifically trained on a diverse French corpus. Given the nature of our task—predicting the difficulty of French texts—CamemBERT emerged as a particularly well-suited candidate. Its robust training and ability to capture the nuances of the French language made it an ideal choice for enhancing our text classification capabilities.

CamemBERT's transformer architecture, with its attention mechanisms, offered a significant leap forward from traditional machine learning models. It could contextualize words in a sentence more effectively, leading to more nuanced predictions and a deeper understanding of text complexity.

Integrating CamemBERT into our workflow marked a turning point in our project. We adapted our data preprocessing to accommodate the input requirements of the model and fine-tuned CamemBERT on our dataset, eagerly anticipating the impact it would have on our submission's performance in the Kaggle competition.


## 4 Our Best Model

### 4.1 Implementation and Performance

In our journey to find the most effective model for text difficulty classification, we turned to CamemBERT, a transformer-based model renowned for its proficiency in understanding French language nuances. Here's a brief overview of how we implemented and utilized CamemBERT:

- **Data Preparation**: We used a dataset comprising sentences and their corresponding difficulty levels. This dataset was split into training and testing sets, with a focus on balancing the distribution across different difficulty categories.

- **Tokenization**: Utilizing CamembertTokenizer from the transformers library, we tokenized the text data. This process involved truncating and padding the texts to a maximum length, ensuring uniformity for model processing.

- **Model Architecture**: We employed CamembertForSequenceClassification, a variant of CamemBERT fine-tuned for sequence classification tasks. Our model was configured to classify texts into six difficulty levels (from A1 to C2).

- **Training**: The model was trained on the tokenized training data, using the AdamW optimizer with a learning rate of 5e-5. Training was conducted over several epochs, with each batch of data being passed through the model to calculate loss and backpropagate the gradients.

- **Evaluation**: We evaluated the model's performance using a custom evaluate function, calculating accuracy on both the training and testing sets. This provided us with insights into how well the model generalizes to unseen data.

The implementation of CamemBERT proved to be a significant leap forward in our project. It demonstrated remarkable accuracy in classifying the difficulty of French texts, outperforming our initial models. 

| Metrics | CamenBERT | CamenBERT Kaggle Submission |
|:---------:|:---------:|:---------:| 
| Precision  | 0.9520 | |
| Recall     | 0.9522 | |
| F1-score  | 0.9520 | |
| Accuracy | 0.9518 | 0.606 |


A critical aspect of evaluating our CamemBERT model's performance was analyzing its confusion matrix. This analysis revealed key insights into the model's prediction patterns and areas where it most frequently erred. Notably, the most common misclassifications involved:

- **Predicting A2 instead of B1**: A significant number of texts that were actually at the B1 difficulty level were predicted as A2 by our model. This suggests a challenge in distinguishing between the nuances of these intermediate language proficiency levels.
- **Predicting B1 instead of B2**: Similarly, many texts at the B2 level were incorrectly classified as B1. This indicates a similar issue in differentiating between these closely related difficulty levels.

These patterns in the confusion matrix underscored specific challenges in our model's ability to discern subtle differences in text complexity, particularly between adjacent proficiency levels. Understanding these misclassifications provides valuable direction for future improvements, possibly through more targeted data augmentation or fine-tuning of the model to better capture these nuances.

![](https://github.com/ROULIND/DSML-Apple-Project-difficulty-analysis/blob/main/images/CamenBERT-ConfusionMatrix.png)

### 4.2 Overfitting Challenge
Despite achieving impressive scores with our training data using CamemBERT, we encountered a notable challenge when submitting predictions on Kaggle. Our highest score on the Kaggle platform was 0.606, a stark contrast to the high accuracy observed during training. This discrepancy highlighted a critical issue in our model: overfitting.

- Overfitting on Training Data: The high performance on the training set, while initially encouraging, suggested that our model might be too closely attuned to the specifics of the training data. As a result, it struggled to generalize effectively to the diverse range of texts presented in the Kaggle competition.
- Generalization to Unseen Data: The lower score on Kaggle indicated that our model, while powerful, was not adequately generalizing to unseen data. This is a common challenge in machine learning, especially with complex models like transformers.

### 4.3 Data Augmentation Efforts and Challenges

To enhance our model's accuracy, we delved into data augmentation, employing several techniques to expand our training dataset. Our initial approach involved using synonyms to generate new sentences. However, we realized that substituting words with synonyms could inadvertently alter the text difficulty, thus affecting the model's learning process.

We then experimented with generating sentences and their corresponding difficulty levels using ChatGPT, both directly within the prompt and through a Python script. This effort successfully increased our training dataset from 4,800 to over **12,000 entries**, adding more than 7,200 new data points. (**full_augmented_training_data.csv**)

Despite this significant increase in training data, our model's precision on the Kaggle submissions unexpectedly decreased. This **decline in performance** could be attributed to overfitting, particularly as the newly generated data might have varied substantially from the original dataset and the texts used in Kaggle's evaluation. This outcome highlighted the delicate balance required in data augmentation - ensuring new data is beneficial and closely aligned with the task's requirements. (**DSML-Model-CamenBERT-DataAugmentation.ipynb**)


## 5. Conclusion

### 5.1 Reflecting on Our Journey with CamemBERT
In conclusion, we believe CamemBERT has proven to be an excellent model choice for our text classification challenge. Its potential was evident, despite the hurdles we encountered. Given more time and resources, we could have further refined our approach, particularly in producing high-quality augmented data while implementing strategies to significantly reduce overfitting.

Time was a crucial factor in our project. Training CamemBERT, especially with an expanded dataset, required substantial computational resources and time. The augmentation process not only added to the dataset size but also exponentially increased the demand for these resources. If provided with additional time and computational power, we are confident that we could have enhanced the model's accuracy even further, making it more adept at handling the diverse range of texts in the Kaggle competition.

This project has been a testament to the capabilities of advanced NLP models like CamemBERT and the importance of balancing model complexity with data quality and computational practicality.

### 5.2 Kaggle Competition Results

Our foray into the Kaggle competition culminated in a commendable achievement. We secured the 5th position, finishing just 0.023 points behind the winners in terms of precision. This close margin highlights both the competitiveness of the event and the effectiveness of our model.

This result is particularly noteworthy considering the complexity of the task and the high level of skill exhibited by other participants. Finishing in the top five is a testament to the hard work and strategic decisions we made throughout the project, especially our choice to use CamemBERT and our efforts in data processing and model optimization.

![](https://github.com/ROULIND/DSML-Apple-Project-difficulty-analysis/blob/main/images/Kaggle-Leaderboard.jpg)

## 6. Team

Our project was led by Jonatan Gretz and Dimitri Roulin, Master's students specializing in Information Systems and Digital Innovation.



