# 🔥 Prairie Burn Detection – Baseline Machine Learning

This repository contains baseline ML model evaluations for prairie burn detection using stacked multitemporal Landsat TM data.

## 📁 Repository Structure

```
I am still building this structure This is a school project 
Prairie-Burn-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── FullStacked_data.csv
│
├── notebooks/
│   └── Step5_Stacked_BaseLine_ML.ipynb
│
├── results/
│   └── summary_baseline_results.md
│
└── src/
    └── baseline_models.py
``` 

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/prairie-burn-detection.git
    cd prairie-burn-detection
    ```

2. **Place the dataset** in the `/data` folder:

    - Ensure `FullStacked_data.csv` is inside `data/`

3. **Open the Jupyter Notebook**:

    ```bash
    jupyter notebook notebooks/Step5_Stacked_BaseLine_ML.ipynb
    ```

4. **Run all cells** to execute the baseline models.


## 📊 Model Performance Summary

| Model              | Resub Accuracy | CV Avg Accuracy |
|---------------------|----------------|-----------------|
| Logistic Regression | 0.886          | 88.64%          |
| KNN                | 0.933          | 90.79%          |
| Decision Tree      | 0.999          | 87.79%          |
| Linear SVC         | 0.818          | 72.86%          |

📌 **Best Model:** KNN — stable and accurate  
⚠️ **Decision Tree** overfits; **Linear SVC** is inconsistent


## 📝 Notes

- The dataset is randomly shuffled at each run, so results may slightly vary.
- Decision Tree shows signs of overfitting.
- KNN is the best-performing model in terms of stability and accuracy.


## 📌 Future Improvements

- Hyperparameter tuning for KNN and Decision Tree.
- Implementation of **Random Forests** and **Gradient Boosting**.
- Integration with real-time satellite data streams.


## 🙌 Contributing

Feel free to fork this repository, submit issues, and make pull requests. All contributions are welcome!


## 📫 Contact

- **Author:** Jeffrey Appiagyei
- **Email:** [boakyejeff@gmail.com](mailto:boakyejeff@gmail.com)
- - **Email:** [boakyejeff@gmail.com](mailto:boakyejeff@gmail.com)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

