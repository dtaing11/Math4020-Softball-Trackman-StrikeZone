# 🥎 LSU Softball Strike Zone Prediction Model

This project builds a neural network that predicts the probability a pitch will be **called a strike** in LSU Softball.  
The model uses pitch-location data enhanced with batter and pitcher handedness, swing information, and neural-network transfer learning techniques to capture umpire strike-zone behavior more accurately.

---

## 🎯 Overview

Traditional strike-zone models rely only on pitch location.  
This project expands that idea by incorporating:

- Batter handedness (L/R)
- Pitcher handedness (L/R)
- Swing decision (taken vs swung)
- Full fine-tuning on LSU Softball pitch-tracking data

The result is a **probabilistic strike-zone model** capable of producing detailed heatmaps showing how likely an umpire is to call a pitch a strike under different conditions.

---

## 🧠 Model Architecture

The final Multilayer Perceptron (MLP) has **5 input features**:

1. PlateLocHeight (vertical pitch location)  
2. PlateLocSide (horizontal pitch location)  
3. Swing (0 = taken, 1 = swing)  
4. Batter handedness (0 = R, 1 = L)  
5. Pitcher handedness (0 = R, 1 = L)

Hidden layers:

- Linear (5 → 32) + ReLU  
- Linear (32 → 64) + ReLU  

Output layer:

- Linear (64 → 1) → sigmoid (during inference)

This produces a **single probability** representing the likelihood of a called strike.

---

## 🔁 Training Process & Weight Surgery

### **Stage 1 — Initial Small Dataset**
The project began with a small model trained on a limited dataset with only a few features (primarily pitch location).  
Because the dataset was small, the network was simple and had limited flexibility.

### **Stage 2 — Model Expansion With More Data**
Once a larger and more detailed LSU dataset became available, more variables (batter hand, pitcher hand, swing) could be added.

To avoid discarding the progress made by the original model, we applied **neural network weight surgery**:

- The original **3-input** first-layer weight matrix was copied into a new **5-input** weight matrix.
- The two new feature columns were initialized to zeros.
- This preserved the original model behavior while allowing it to learn the influence of new features during fine-tuning.

### **Stage 3 — Fine-Tuning on LSU Softball Data**
The expanded model was then fine-tuned on the full LSU dataset.  
Because weight surgery preserved previously learned patterns, the model converged faster and avoided overfitting—ideal for a medium-sized dataset.

---

## 🎨 Heatmap Generation

The model can generate **strike-zone probability heatmaps** across the plate area.  
These heatmaps can be conditioned on:

- Swing = 0 or 1  
- Batter Hand = R/L  
- Pitcher Hand = R/L  

This results in **8 distinct scenarios**, each producing a probability map.

Each heatmap also displays:

### **Official Softball Rulebook Strike Zone**
- Horizontal range: **x ∈ [-0.708, 0.708]** (17-inch plate width)  
- Vertical range: **y ∈ [1.5, 3.5]**

This helps visually compare model predictions with the theoretical rulebook zone.

---

## 📊 Interpretation

Heatmaps show how umpire behavior shifts with context:

- Different zones for left-handed vs right-handed batters  
- Inside/outside bias depending on pitcher/batter handedness  
- Effects of whether the batter swings  
- Regions where umpires expand or shrink the zone compared to the rulebook

The model learns empirical tendencies rather than rulebook definitions, enabling performance analysis and scouting insights.

---

## 🏆 Summary

This project combines:

- Neural network modeling  
- Feature engineering  
- Transfer learning with weight surgery  
- Softball strike-zone analytics  
- Visualization of data-driven umpire tendencies  

The final strike-zone predictor provides a detailed, flexible, and explainable probability model suitable for analysis, scouting, and future expansion.

