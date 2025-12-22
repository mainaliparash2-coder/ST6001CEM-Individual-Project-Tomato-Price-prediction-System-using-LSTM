![Tomato Price Forecast Logo](assets/template1.png)

Predict tomato prices in Kathmandu markets with precision. Leveraging historical data from 2013 to 2021 and advanced LSTM machine learning models, this tool provides accurate price forecasts for tomato varieties traded in Kathmandu's vital Kalimati market.

## Description

**Tomato Price Forecast** is a specialized deep learning application focused on predicting tomato prices in Kathmandu's Kalimati Fruits and Vegetables Market. This market significantly satisfies 60-70% of Kathmandu Valley's demand for agricultural produce. 

The project harnesses the predictive power of Long Short-Term Memory (LSTM) networks — a specialized form of Recurrent Neural Networks (RNNs) — to analyze and forecast tomato prices across 6 different varieties:

- **Tomato Big (Nepali)** - 2,507 records
- **Tomato Small (Local)** - 2,741 records  
- **Tomato Small (Tunnel)** - 686 records
- **Tomato Big (Indian)** - 474 records
- **Tomato Small (Indian)** - 357 records
- **Tomato Small (Terai)** - 286 records

Using a comprehensive dataset of **7,051 tomato price records** spanning from 2013 to 2021, this tool provides vital insights for farmers, traders, market analysts, and policymakers to make informed decisions about tomato pricing and market trends.

### Dataset

This project utilizes an open-source dataset from Open Data Nepal, available at: [Kalimati Tarkari Dataset](https://opendatanepal.com/dataset/kalimati-tarkari-dataset). The dataset has been filtered to include only tomato varieties. The dataset is employed here strictly for educational and research purposes only.

## Getting Started

### Prerequisites

Ensure you have Python installed on your machine. If not, download and install it from [Python's official site](https://www.python.org/).

### Installation & Running

Follow these steps to get the project up and running on your local machine:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/suyogkad/freshForecast.git
   cd freshForecast

2. **Install Requirements**

   Make sure to create a virtual environment before installing dependencies.
   
   ```sh
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows

   pip install -r requirements.txt
   
3. **Run the Flask app**
   
   ```sh
   cd flask_app
   python app.py
   ```
   
   Now, navigate to http://127.0.0.1:5000/ in your browser to access the application.

## Usage

Users can leverage **Tomato Price Forecast** to analyze historical tomato prices from 2013 to 2021 in Kalimati, Kathmandu, and predict future prices for different tomato varieties. By selecting a specific tomato type and date, users can gain insights into price predictions, aiding in more informed decision-making for agricultural planning, market analysis, and research purposes.

## License

While the utilized dataset is open and accessible from [Open Data Nepal](https://opendatanepal.com/dataset/kalimati-tarkari-dataset), this project's codebase is not open source and is intended solely for educational and academic viewing. Usage, modification, or distribution of the code requires explicit permission from the author. View [LICENSE](https://github.com/suyogkad/freshForecast/blob/main/LICENSE).
# ktm-tomato-price-predictor
# ST6001CEM-Individual-Project-Tomato-Price-prediction-System-using-LSTM
