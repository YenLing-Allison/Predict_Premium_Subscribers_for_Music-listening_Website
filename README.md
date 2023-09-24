# Predict Premium Subscribers Projects
echo "# Predictive-Analysis-Projects" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YenLing-Allison/Predictive-Analysis-Projects.git
git push -u origin main

# Business Question
Website XYZ, a music-listening social networking website, follows the “freemium” business model. The website offers basic services for free, and provides a number of additional premium capabilities for a monthly subscription fee.
We are interested in **predicting which people would be likely to convert from free users to premium subscribers in the next 6 month period**, if they are targeted by our promotional campaign. 

# Resrources
Dataset contains 41,540 records (1540 adopters and 40,000 non-adopters), each record representing
a different user of the XYZ website who was targeted in the previous marketing campaign.
Each record is described with 25 attributes.

Dataset: [XYZData.csv](https://github.com/YenLing-Allison/Predict-Premium-Subscribers-for-Music-listening-Website/blob/676ace52e7f0979b2d143101ce2190e462b62ac9/XYZData.csv)

# Analysis Process
1. Data Preprocessing   
-  Data cleaning  
-  Data visulization  
-  Addressed the imbalanced data  
2. Model Development  
-  Feature selection: filter approach
-  Predictive modeling
-  Model tuning
3. Model Performance
-  Confusion Matrix
-  ROC curve
-  AUC

Technical Document: [freemium_to_premium_models.R](https://github.com/YenLing-Allison/Predict_Premium_Subscribers_for_Music-listening_Website/blob/359eb3437bae3fca7f484ad95abb0d3fc42013a3/freemium_to_premium_models.R)  

# Result 
#### Overall, K-NN has the best model performance. 
#### The features of potential premium subscribers are highly related to "Love songs Tracked", "number of friend using XYZ music-listening website", and "songs listened".
#### Most potential premiumn subscribers are around 20-29.  

Executive Summary: [Executive Summary_Predictive.pdf](https://github.com/YenLing-Allison/Predict_Premium_Subscribers_for_Music-listening_Website/blob/15ea47340a78634e3ad3cea27d6d613e1a1eac90/Executive%20Summary_Predictive.pdf)


