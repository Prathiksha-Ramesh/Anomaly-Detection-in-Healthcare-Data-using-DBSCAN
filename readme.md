# Anomaly Detection in Healthcare Data using DBSCAN

This repository contains the source code, dataset, and resources for the **Anomaly Detection in Healthcare Data using DBSCAN** project. The project demonstrates the application of the DBSCAN clustering algorithm for detecting anomalies in a healthcare dataset. Anomalies in this context refer to data points that deviate significantly from the rest of the dataset, which can indicate potential outliers or errors in the data.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Imports](#imports)
  - [Data Loading](#data-loading)
  - [Data Preprocessing](#data-preprocessing)
  - [Anomaly Detection](#anomaly-detection)
  - [Visualization](#visualization)
- [License](#license)
- [Contact](#contact)

## Project Overview

The **Anomaly Detection in Healthcare Data using DBSCAN** project includes the following key steps:

- **Data Loading**: Importing the healthcare dataset from a CSV file.
- **Data Preprocessing**: Scaling the features using `StandardScaler` to normalize the data before applying the clustering algorithm.
- **Anomaly Detection**: Implementing the DBSCAN algorithm to identify and classify anomalies in the dataset.
- **Visualization**: Plotting the results to visualize normal data points and anomalies.

This project is ideal for identifying outliers in healthcare data, which can be crucial for ensuring data quality and detecting unusual patterns that may require further investigation.

## Project Structure

- **notebook.ipynb**: The Jupyter notebook containing the complete code for the analysis, from data loading to anomaly detection and visualization.
- **healthcare.csv**: The dataset used in this project, containing healthcare-related data.
- **LICENSE**: The Apache License 2.0 file that governs the use and distribution of this project's code.
- **requirements.txt**: A file listing all the Python libraries and dependencies required to run the project.
- **.gitignore**: A file specifying which files or directories should be ignored by Git.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
```

2. Navigate to the project directory:
``` bash
cd your-repository-name
```

3. Create a virtual environment (optional but recommended):

``` bash
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

4. Install the required dependencies:
``` bash
 pip install -r requirements.txt
```

5. Run the Jupyter notebook:

``` bash
jupyter notebook notebook.ipynb

```
## Usage

Imports

The notebook begins by importing the necessary libraries:

``` bash

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline

```

These libraries are essential for data manipulation (pandas, numpy), visualization (seaborn, matplotlib), and anomaly detection (DBSCAN from scikit-learn).

## Data Loading
The healthcare dataset is loaded from the healthcare.csv file using pandas:

``` bash
data = pd.read_csv('healthcare.csv')
```

## Data Preprocessing
The features are scaled using StandardScaler to normalize the data before applying DBSCAN:

``` bash
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
```

## Anomaly Detection
The DBSCAN algorithm is applied to the scaled data to detect anomalies:

``` bash
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X_scaled)
labels = dbscan.labels_
```

- Normal Points: Data points with a label of 0 or positive integers are considered normal.
- Anomalies: Data points labeled as -1 are considered anomalies or outliers.

## Visualization
The results of the anomaly detection can be visualized by plotting the data points and coloring them according to their cluster labels:

``` bash
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='plasma')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Detection using DBSCAN on Healthcare Data')
plt.show()

```

This plot will show how DBSCAN has grouped the normal data points and identified anomalies, which are typically represented as noise points (labeled as -1).

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details.
