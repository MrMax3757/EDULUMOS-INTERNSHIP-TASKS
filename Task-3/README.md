# 🛍️ Customer Segmentation Using K-Means

## 📌 Project Overview

**Edulumos Internship — Task 03**

This project focuses on performing **Customer Segmentation** using the **K-Means clustering algorithm** on the Mall Customer Dataset. The goal is to group customers into meaningful segments based on shared characteristics.

This kind of segmentation is frequently used in marketing strategy, e-commerce personalization, retail analytics, and CRM systems to identify:
* High-value customers
* Customers likely to respond to promotions
* Low-engagement or price-sensitive customers
* Potential loyalty program candidates

---

## 📁 Dataset Description

The dataset represents mall/shopping center customers.

| Feature | Description |
| :--- | :--- |
| **Age** | Customer age |
| **Annual Income (k$)** | Yearly spending capacity estimate |
| **Spending Score (1–100)** | Internal metric assigned by the store based on purchasing behavior |

### 🔍 Excluded Columns
The following columns were removed prior to clustering:
* **CustomerID:** Unique identifier (removed as it has no behavioral meaning).
* **Gender:** Categorical data with low correlation to spending behavior; inclusion tends to create artificial clusters in K-Means.

---

## 🧠 Methodology

The complete workflow follows these steps:

1.  **Data Cleaning:** Removal of duplicates and handling missing values.
2.  **Feature Selection:** Focused on `Age`, `Annual Income`, and `Spending Score`.
3.  **Scaling:** Standardization applied using `StandardScaler()` to ensure all features contribute equally.
4.  **Optimal K Finding:**
    * *Elbow Method*
    * *Silhouette Score Analysis*
5.  **Model Training:** K-Means clustering applied with the optimal number of clusters ($k$).
6.  **Dimensionality Reduction:** PCA (Principal Component Analysis) used to visualize the clusters in 2D space.
7.  **Profiling & Saving:** Analyzing average behaviors and exporting the model (`.pkl`) for future use.

---

## 📊 Results & Visualization

The clustering revealed clear customer segments based on spending behavior and income. The project includes the following visualizations:

* ✔ **Elbow Method Plot:** To determine the optimal cluster count.
* ✔ **Silhouette Score Plot:** To validate cluster separation.
* ✔ **PCA 2D Scatter Plot:** To visualize the segmentation.

### 🧩 Business Interpretation (Cluster Profiling)

*Note: Cluster IDs may vary slightly per run.*

| Cluster | Characteristics | Potential Strategy |
| :---: | :--- | :--- |
| **0** | **High Income, High Spending** | VIP loyalty programs, exclusive offers, luxury marketing. |
| **1** | **Low Income, Low Spending** | Budget promotions, discount-driven campaigns. |
| **2** | **High Income, Low Spending** | Cross-selling strategies, brand awareness, win-back campaigns. |
| **3** | **Young, High Spending** | Trend-based marketing, influencer campaigns, seasonal offers. |

---

## 🎯 Key Takeaways

* **K-Means Effectiveness:** Proven effective for segmenting retail customers when utilizing behavioral and demographic data.
* **Feature Importance:** `Age`, `Income`, and `Spending Score` produced the most meaningful segmentation.
* **Interpretability:** Using visualization techniques like PCA makes high-dimensional clusters interpretable for stakeholders.
* **Application:** The final model is ready for targeted marketing, CRM strategy, and recommendation systems.

---

## 💾 Deliverables

This repository contains:
- [x] `CustomerSegmentation.ipynb`: The complete working Jupyter Notebook.
- [x] `Clustered_Customers.csv`: Cleaned dataset with assigned cluster IDs.
- [x] `kmeans_model.pkl` & `scaler.pkl`: Saved models for future prediction.
- [x] Visual charts for business reporting.

---

## 📝 Future Improvements

To make the model more realistic for enterprise use, the following improvements are suggested:
* **Data Enrichment:** Add purchasing history or transaction frequency.
* **Geography:** Incorporate geographic/country segments.
* **Advanced Algorithms:** Experiment with DBSCAN or Gaussian Mixture Models (GMM).
* **Deployment:** Build a UI using Streamlit, Flask, or Power BI.

---

## 🙌 Credits

**Project completed as part of the Edulumos Data Science Internship.**

**Author:** ✨ [Mohammed Naveeduddin]