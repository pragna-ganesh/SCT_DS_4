import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
df = pd.read_csv("/Users/pragnag/task_4/US_Accidents_March23.csv")

# take a sample for faster clustering & plots
sample = df.sample(n=10000, random_state=42)

# ---------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------
sample['Start_Time'] = pd.to_datetime(sample['Start_Time'], errors='coerce', format='mixed')
sample['Hour'] = sample['Start_Time'].dt.hour
sample['Weekday'] = sample['Start_Time'].dt.day_name()

print(df.head())
print(df.info())

# ---------------------------------------------
# CLUSTERING HOTSPOTS USING KMEANS
# ---------------------------------------------
coords = sample[['Start_Lat', 'Start_Lng']].dropna()

kmeans = KMeans(n_clusters=5, random_state=42)
sample.loc[coords.index, 'Cluster'] = kmeans.fit_predict(coords)

# ---------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------

# 1. Hotspots scatter plot
plt.figure(figsize=(12,8))
plt.scatter(sample['Start_Lng'], sample['Start_Lat'],
            c=sample['Cluster'], cmap='Spectral', alpha=0.5, s=6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Accident Hotspots by KMeans Cluster')
plt.show()

# 2. Density heatmap KDE
plt.figure(figsize=(10,8))
sns.kdeplot(
    x=sample['Start_Lng'], 
    y=sample['Start_Lat'], 
    fill=True, 
    cmap='inferno', 
    thresh=0.05, 
    levels=100
)
plt.title('Accident Density Heatmap')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# 3. Accidents by hour
plt.figure(figsize=(10,5))
colors = sns.color_palette('cubehelix', n_colors=24)
sns.countplot(x='Hour', data=sample, palette=colors, legend=False)
plt.title('Accidents by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.show()

# 4. Accidents by weekday
plt.figure(figsize=(10,5))
colors = sns.color_palette('crest', n_colors=7)
sns.countplot(x='Weekday', data=sample, palette=colors, legend=False,
              order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title('Accidents by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 5. Top 10 accident-prone cities
plt.figure(figsize=(10,5))
top_cities = df['City'].value_counts().head(10)
colors = sns.color_palette('flare', n_colors=len(top_cities))
sns.barplot(x=top_cities.index, y=top_cities.values, palette=colors, legend=False)
plt.title("Top 10 Accident-Prone Cities")
plt.ylabel("Number of Accidents")
plt.xlabel("City")
plt.xticks(rotation=45)
plt.show()

# 6. Top 10 weather conditions
plt.figure(figsize=(10,5))
top_weather = df['Weather_Condition'].value_counts().head(10)
colors = sns.color_palette('rocket', n_colors=len(top_weather))
sns.barplot(x=top_weather.index, y=top_weather.values, palette=colors, legend=False)
plt.title("Top 10 Weather Conditions During Accidents")
plt.ylabel("Number of Accidents")
plt.xlabel("Weather Condition")
plt.xticks(rotation=45)
plt.show()

# 7. Severity distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Severity', data=df, palette='light:firebrick', legend=False)
plt.title("Accident Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Number of Accidents")
plt.show()
