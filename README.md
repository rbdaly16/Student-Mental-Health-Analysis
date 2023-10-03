# **Student-Mental-Health-Analysis**

## **Authors**:
Bobby Daly, DS

![image](https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/75b180c7-2beb-4ee4-955d-4bb5551841e2)<br>





## **Overview**
For this project I analyzed over 6,000 college student mental health surveys. As mental health strongly affects people's lives, it's imperative that developmental institutions, like colleges, do all they can to support students. The survey consisted of questions regarding various aspects of students' lives including sleep, diet, and more. My goal was to build a predictive model that would be able to identify students that were high and low risk for depression based on their answers to the survey questions. Before modeling I performed a preprocessing approach that included:

- making risk for Depression a binary
- scaling numerical columns
- one-hot encoding categorical columns
- ordinal encoding hierarchical categorical columns

Then I began to build models that aimed to increase the recall and accuracy scores. My final model had a 49% recall score and a 51% accuracy score.

This is more accurate than my dummy model, but the recall score is much lower. While I did have a Logistic Regression model that had an 80% cross validation recall score on training data, the cross validation accuracy score was 46%. My goal was to have above 50% for both scores. While I just barely missed that goal, I did learn a lot in the process. I reached the conclusion that humans may be unreliable in self-assessments and that humans are multi-faceted. In order to make a better model, we need more reliable data. This can be achieved through quantifiable questions given to cohort surveys that follow students over time. 


## **Business Problem**
Mental health affects all of us. The better we are at detecting warning signs, the better we will be at supporting each other. Developmental institutions, like colleges, have a particulary large responsibility to help students in this area as much as possible. In a 2022 article from the [American Psychological Association](https://www.apa.org/monitor/2022/10/mental-health-campus-care), "more than 60% of college students met the criteria for at least one mental health problem." Luckily, this conversation is starting to gain momentum. The article goes on to detail how psychologists are becoming more involved with colleges and how supports are being put in place to help students, including a wellness app. This project seeks to work in tandem with this push and help shed light on which aspects of students' lives correlate with a high risk for depression. With that information, colleges and psychologists can put more tailored supports in place for students.

## **Data Understanding**
![image](https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/f7224b44-78ff-4ff6-bbcf-de124c75c8a6)<br>
The data comes from [kaggle.com](https://www.kaggle.com/datasets/sonia22222/students-mental-health-assessments?select=students_mental_health_survey.csv). It consists of over 6,000 mental health surveys where college students provided information regarding their courses, sleep quality, physical activity, social support, and many more aspects of their lives. In this project I will be converting the Depression_Score to a binary target. Three or higher is Yes, this student has depression, and two or below means no, this student does not have depression. My goal is to find which aspects of a student's life has the greatest impact on their probability of having depression. 

Instructions on accessing data:
Option 1 (kaggle.com):
1. Go to [kaggle.com](https://www.kaggle.com/datasets/sonia22222/students-mental-health-assessments?select=students_mental_health_survey.csv)
2. Scroll down to the students_mental_health_survey.csv and select download.

Option 2 (GitHub repo):
1. Go to the [GitHub repo](https://github.com/rbdaly16/Student-Mental-Health-Analysis/tree/main)
2. Fork and clone or just pull the repo down and you will have access to the data csv located in the data folder.

## **Data Preparation**
Before I started modeling, I wanted to see if there were any obvious correlations between certain column values (such as poor, average, and good quality of sleep) and students being high risk for depression. To do this I created bar charts to show the high risk students for each column and the percentage breakdown of each value. The graphs were quite surprising to me as there did not appear to be many factors highly correlated with a high risk of depression. One strange graph that brings a big question to light is the Substance Use. 80% of students reported that they never used any substances. 80%! This is highly skeptical.<br>

![Screenshot 2023-09-28 at 2 51 39 PM](https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/f790a9bf-6e5a-40ee-855d-82c1947c9842)

To see a better picture of the spread of high risk students by the category in each column, I decided to make a visual of the percentage of each category's high risk students. Many categories above could be misleading because it is based on count instead of percentage. For example, very few students are high risk who live with their family, but living with family is also the smallest subset of students in the dataset. In order to get a better picture of how living with family affected risk of depression, I would prefer to see the risk breakdown of the percentage of students who live at home. The graphs here showed the percentage of students that are high risk for each category in each column. Most column categories have a fairly even spread. For example, the Stress Level graph shows that just under 50% of students with stress levels of 0 are high risk for Depression. However, the graph also shows that for stress levels of 1 through 5, meaning that stress levels are not a good indicator for risk of depression. Surpisingly, it is the same story for Age, Anxiety, Sleep, Physical Activity, Diet, Social Support, Relationship Status, Substance Use, Counseling Service Use, Family History, Chronic Illness, Extracurricular Involvement, Semester Credit Load and Residence Type. <br>

<img width="667" alt="Screenshot 2023-09-29 at 11 42 07 AM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/ca293427-94d7-4fd1-b9f0-9c366ad335e9">



The column that does stand out as having some correlation with high risk is Course. According to the graph, out of all computer science majors who completed this survey, about 70% of them were high risk for depression, while all other courses were around 40%. <br>
<img width="653" alt="Screenshot 2023-09-29 at 11 42 56 AM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/105808bc-e087-4e09-a4ed-4de021fb1227">


In order to prepare the data for preprocessing, I first dropped the 27 rows containing null values. Once I was working with a clean dataset, I then created a binary column for Depression. In the survey, students selected a number between 0 and 5 for their Depression. I created a column titled 'Depression Binary' that had 1 if the depression score was 3 or higher, and 0 if the depression score below 3. I did this in order to create a binary classification model. This type of model simplifies the process for the schools as they can use this to determine whether a student is more or less likely to need additional support. 

After this, the preprocessing steps performed included:

* one-hot encoding categorical columns
* ordinal encoding hierarchical categorical columns
* scaling numerical columns

## **Modeling**
In order to analyze the performance of the models I built, I focused on recall. This model could be used at schools to help determine which students may need additional support in and out of the classroom. I decided it was best to minimize false negatives (saying a student is less likely to have depression when they are actually more likely), because that would be neglecting to offer additional support when it is needed. A false positive on the other hand, would be a small cost to the school, but ultimately would just give more support to a student, which is never a bad thing. 

For this problem I needed to use classification models. There are many options to use and I wasn't sure which would be most appropriate so I created a wide variety. First, I started with a decision tree purposefully trying to overfit the model to confirm that we had the right data in order to build a successful predictive model. After then reducing the overfitting, I then tried Random Forest, Logistic Regression, and finally a Gradient Boosting Classifier.

Note: All of the models created below (except the Dummy Model) utilize a pipeline that includes the preprocessing steps listed earlier in order to avoid data leakage.

## **Evaluation**
In order to evaluate the models consistently, I created a function that would show each model's cross validation and training scores for recall and accuracy. It also showed a confusion matrix for a visual of a breakdown of the model's predictions. While both scores were useful for the interpretation of the model, the tweaking of the models were to improve the recall score. 

#### **Dummy Model**
The first model I made was a Dummy Model. This model predicts 1 every single time and thus, has a perfect recall score.<br>
<img width="416" alt="Screenshot 2023-09-29 at 11 44 53 AM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/84de8218-3886-4f53-be04-dab57d023cd7">

#### **Decision Tree Model**
The first model I created was a decision tree with all default parameters. My goal here was to purposefully create an overfit model in order to confirm it achieves a high recall score. Accomplishing this shows that I am using good enough data in order to predict the target variable. This worked successfully, this model received a recall score of 100% on the training data but below 50% on the cross validation, showing this model is very overfit. <br>
<img width="411" alt="Screenshot 2023-09-29 at 11 46 18 AM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/0477cc2f-8cc8-4b44-ba7f-787688941aa3">

#### **Random Forest Model**
The decision tree certainly held promise, so I decided to use a random forest which would be able to combine the power of many decision trees. I first created a random forest model with default parameters and after a few grid searches I attempting to address the overfitting I ended up with the following scores. <br>
<img width="540" alt="Screenshot 2023-09-29 at 11 48 00 AM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/310b1120-7013-408c-ae65-4afce7c12cae"> <br>

While there was much more tweaking to be done on this model, I decided to use a Gradient Boosting Classifier (shown further down) to continue this plunge because that is also a tree-based model. For now though, I decided to investigate Logistic Regression.

#### **Logistic Regression Model**
After adding polynomial features to increase complexity and tweaking with grid searches to address the overfitting, I ended with a model with the following scores. <br>
<img width="548" alt="Screenshot 2023-09-29 at 11 51 56 AM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/ee98bf4f-ef86-4e84-94af-ed42c5b7db2a"> <br>

While the recall score was excellent, the accuracy score was only about 2% more than the Dummy Model. I decided to create a Logistic Regression model focused on accuracy to see if that could help.

#### **Logistic Regression Model (Accuracy)**
After multiple grid searches, this model received a better accuracy score, but the recall score was much worse. <br>
<img width="551" alt="Screenshot 2023-09-29 at 11 54 08 AM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/feebaf5c-9502-44b0-a76a-389b459d15e9"><br>

Since this model was not providing the desired recall score, I decided to try a Gradient Boosting Classifier. 

#### **Gradient Boosting Classifier Model**
After 10 different iterations of grid searches, I arrived at my best model with both a recall and accuracy score above 50%.<br>
<img width="597" alt="Screenshot 2023-09-29 at 4 58 45 PM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/6fb252e2-0eb1-42c5-86c9-206359e623cc"><br>


While these scores are not great, they were the best I could obtain with the data, so I decided to use it with the test data.

#### **Final Model (Gradient Boosting Classifier)**
The final model was the Gradient Boosting Classifier with Grid Search 10 with the following hyperparameters:
- loss: deviance
- learning_rate: 0.5
- n_estimators: 475
- min_samples_split: 8
- max_depth: 24
- max_features: 25
- subsample: 0.1 <br>

When the final model was used on testing data it received a recall score of 49% and an accuracy score of 51%. <br>

<img width="540" alt="Screenshot 2023-09-29 at 5 01 15 PM" src="https://github.com/rbdaly16/Student-Mental-Health-Analysis/assets/126971652/1b6fb25d-e726-4f73-be6e-733b6a882fea"><br>

## **Conclusions**
When deployed on the testing data, the final model received recall score of 49% and an accuracy score of 51%. This was comparable to the cross validation score and was expected.

The main challenge in this project was attempting to obtain both a high recall and accuracy score. While recall is the most important because we don't want to accidentally ignore a high risk student, many models were simply doing just what the Dummy Model would do. The models struggled with correctly predicting whether a student was high or low risk because none of the columns in the dataset had a strong correlation with depression. This could be due to dishonest responses from students, simplistic snapshots of a student's complex life that doesn't take into consideration their true experiences, or most likely, a combination of both of these.

As the model currently stands, I would not recommend a school implement this model. The recall and accuracy are not high enough yet to be practical. However, there are steps that can be taken to improve the efficacy of the model. 

## **Next Steps**
After this research, I think the best way forward is to obtain honest data through a Cohort Survey with quantifiable questions that follow poarticipants over the course of a semester or school year. 

The quantifiable questions are important because humans tend to lie when talking about hard topics like depression. And even when they're not consciously lying, everyone has a different lens through which they view life. For example, a good night's sleep to someone may be a poor night's sleep to another. To resolve this, we need the survey to include quantifiable questions such as, 
- how many hours of sleep did you get last night? 
- how many hours of physical activity did you get today?
- how many drinks of alcohol did you have last night?
- how many clubs/extracurricular activities are you involved in?

The above questions would help reduce bias and standardize participant responses. But they're still just tiny snapshots into students' lives. Following the participants over the course of a semester or school year is important because human emotions vary greatly from day to day, hour to hour, minute to minute. A student stressed out to the max during an exam week may answer these questions differently than when they have very few assignments due. Having students take daily or weekly surveys throughout the semester or even the year would allow us to average the data and get a more realistic picture of how each individual is truly doing. 


## **Thank You**
Thank you for taking the time to review this research.
I hope this information helps and I look forward to working with you more on the next steps.

Sincerely, <br>
Bobby Daly

## Further Details
Further details are available in the full analysis presented in the [Jupyter Notebook](https://github.com/rbdaly16/Student-Mental-Health-Analysis/blob/main/Current%20Final.ipynb) and the presentation slides can be found [here](). 

## Repository Structure
```
├── data
├── scratch_notebooks
├── README.md
├── LICENSE
├── .gitignore
├── Student Mental Health Analysis Presentation.pdf
└── Student Mental Health Analysis.ipynb
```


