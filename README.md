# Flight Price Prediction: End-to-End Machine Learning Pipeline

## ✈️ Executive Summary

The **Flight Price Prediction** project is a robust, end-to-end machine learning solution designed to estimate flight fares based on various travel factors. By analyzing historical flight data from Bangladesh, this project identifies key drivers of ticket prices—such as airline, route, departure time, and class—and builds predictive models to assist travelers and businesses in making informed decisions.

The project simulates a professional data science workflow, encompassing data ingestion, rigorous cleaning, advanced feature engineering, and the training of multiple regression models to achieve high accuracy.

## ✨ Key Features

- **Comprehensive Data Processing**: Automated cleaning pipeline handling missing values, outliers, and data type standardization.
- **Advanced Feature Engineering**: Creation of derived features like 'Day of Week', 'Journey Month', and seasonality indicators to capture temporal pricing trends.
- **In-Depth EDA**: Visualizations revealing insights into price distributions across airlines, stops, and classes.
- **Multi-Model Training**: Implementation and comparison of Linear Regression, Random Forest, and XGBoost models.
- **Workflow Orchestration**: Design for an Airflow-managed pipeline (conceptually mapped) for automated data flow.
- **Containerization Ready**: Infrastructure design supports Docker for reproducible environments.

## 🏗️ Architecture & Workflow

The pipeline is designed to be modular and scalable. Below is the architectural flow, illustrating how data moves from ingestion to model inference, orchestrated by Airflow and containerized with Docker.

```mermaid
graph TD
    subgraph Infrastructure ["Docker Compose"]
        style Infrastructure fill:#095592,stroke:#333,stroke-width:2px

        subgraph Airflow ["Airflow DAG: flight_price_prediction"]
            direction TB

            T1["<b>Task: ingest_data</b><br/>PythonOperator<br/><i>(Pandas read_csv)</i>"]
            T2["<b>Task: preprocess_data</b><br/>PythonOperator<br/><i>(Clean missing values, Outliers)</i>"]
            T3["<b>Task: feature_engineering</b><br/>PythonOperator<br/><i>(OneHotEncoding, Scaling)</i>"]
            T4["<b>Task: train_models</b><br/>PythonOperator<br/><i>(LinearReg, RandomForest, XGBoost)</i>"]
            T5["<b>Task: evaluate_models</b><br/>PythonOperator<br/><i>(Calculate RMSE, R2, MAE)</i>"]

            T1 --> T2
            T2 --> T3
            T3 --> T4
            T4 --> T5
        end
    end

    %% Data flow annotations
    D1[("Raw CSV")] -.-> T1
    T3 -.-> D2[("Processed Data")]
    T4 -.-> M[("Saved Models")]

    classDef task fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    class T1,T2,T3,T4,T5 task;
```

## 🛠️ Technology Stack

- **Programming Language**: Python 3.12+
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Orchestration**: Apache Airflow
- **Containerization**: Docker, Docker Compose
- **Notebook Environment**: Jupyter / VS Code

## 📊 Dataset Details

The project utilizes the **Flight Price Dataset of Bangladesh**.

- **Source**: Kaggle / Local CSV
- **Size**: ~57,000 records
- **Key Features**:
  - `Airline`: Carrier name (e.g., Biman Bangladesh, US-Bangla).
  - `Source` / `Destination`: Consists of IATA codes (DAC, CXB, etc.).
  - `Departure Date & Time` / `Arrival Date & Time`.
  - `Stops`: Number of layovers (Direct, 1 Stop, etc.).
  - `Class`: Economy, Business.
  - `Total Fare`: Target variable (Price in BDT).

## 🧠 Methodology

### 1. Data Cleaning & Preprocessing

- **Column Standardization**: Renamed columns to snake_case for consistency.
- **Type Conversion**: Converted date/time columns to datetime objects.
- **Handling Nulls/Dupes**: Removed duplicate records and imputed missing values where appropriate.

### 2. Feature Engineering

- **Temporal Features**: Extracted `Day`, `Month`, `Year`, and `Weekday` from timestamps.
- **Duration Calculation**: Computed flight duration in minutes/hours.
- **Categorical Encoding**: Applied One-Hot Encoding for nominal variables (Airline, Source) and Label Encoding for ordinal variables (Stops).

### 3. Exploratory Data Analysis (EDA)

- Analyzed the correlation between **Flight Duration** and **Price**.
- Compared average prices across different **Airlines** and **Classes**.
- Investigated seasonal surges in ticket prices.

### 4. Model Building & Evaluation

We trained multiple regression models to find the best fit:

- **Linear Regression**: Baseline model.
- **Random Forest Regressor**: Captures non-linear relationships; tuned for depth and estimators.
- **XGBoost Regressor**: High-performance gradient boosting model for superior accuracy.

**Metrics Used**:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**

## 📂 Project Structure

```
flight-price-prediction/
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned data for modeling
├── notebooks/
│   └── notebook.ipynb      # Main analysis and modeling notebook
├── src/
│   ├── ingestion.py        # Data loading scripts
│   ├── preprocessing.py    # Cleaning and transformation logic
│   └── train.py            # Model training scripts
├── dags/
│   └── flight_price_dag.py # Airflow DAG definition
├── Dockerfile              # Docker image configuration
├── docker-compose.yaml     # Service orchestration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional, for Airflow)

### Local Setup

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/flight-price-prediction.git
    cd flight-price-prediction
    ```

2.  **Create a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Notebook**:
    ```bash
    jupyter notebook notebooks/notebook.ipynb
    ```

## 📈 Results & Insights

- **Price Determinants**: `Class` (Economy vs. Business) and `Duration` were found to be the strongest predictors of price.
- **Airline Variance**: Specific airlines commanded a premium regardless of route.
- **Model Performance**: XGBoost outperformed Linear Regression by a significant margin, achieving an R² score of ~0.85 (illustrative).

## 🔮 Future Work

- [ ] **Hyperparameter Tuning**: Use Optuna for deeper optimization of XGBoost parameters.
- [ ] **Deployment**: Serve the model via a FastAPI endpoint.
- [ ] **Dashboard**: Build a Streamlit app for users to check predicted prices.

## 🤝 Contributors

- **Richard Elinam Nutsuga** - _Project Lead & Developer_

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
