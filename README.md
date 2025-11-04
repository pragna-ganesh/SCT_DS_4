# README.md

# US Road Accident Analysis – Clustering & EDA

This project analyzes the **US Accidents Dataset (March 2023)** and applies clustering + visual analytics to identify accident hotspot regions and key factors affecting accidents.

Note: The dataset is very large, so it is NOT included in this repository.  
Download from Kaggle here:  
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

##  What this script does

• Loads the US accidents dataset  
• Takes a 10,000-row sample for faster clustering processing  
• Extracts hour + weekday features  
• Performs **KMeans clustering** on latitude/longitude to detect hotspot zones  
• Creates multiple visualizations such as:

- Hotspot scatter map
- Accident density heatmap (KDE)
- Accidents by hour of the day
- Accidents by weekday
- Top 10 accident cities
- Top 10 weather conditions
- Severity distribution

---

##  Requirements

bash
pip install pandas numpy matplotlib seaborn scikit-learn

---

## Output

Running the script will:

• Print data preview & structure
• Perform KMeans clustering
• Display multiple graphs such as:
    Accident hotspot clusters on map
    Accident heatmap density
    Countplots by hour & weekday
    Bar charts for top cities & weather
    Severity level distribution

These visualizations help identify when & where accidents occur most frequently in the US.
