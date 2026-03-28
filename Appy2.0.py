from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def encode_plot_to_base64(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.read()).decode('utf-8')

def analyze_cashflow(Pre_year, budget_data, revenue_2016_data):
    # Revenue Comparison Bar Chart
    categories = list(Pre_year.keys())
    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, list(Pre_year.values()), width, label='Previous Year', color='blue')
    plt.bar(x, list(budget_data.values()), width, label='Budget', color='orange')
    plt.bar(x + width, list(revenue_2016_data.values()), width, label='Current Year', color='green')

    plt.ylabel('Revenue (In Million $)')
    plt.title('Revenue Comparison by Category')
    plt.xticks(x, categories)
    plt.ylim(0, max(max(Pre_year.values()), max(budget_data.values()), max(revenue_2016_data.values())) * 1.2)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    revenue_chart = encode_plot_to_base64(plt)

    # Cashflow Simulation and Predictive Model
    np.random.seed(42)
    dates = pd.date_range(start='2015-01-01', end='2016-08-23', freq='D')
    n_days = len(dates)
    inflows = np.random.normal(loc=1000, scale=300, size=n_days)
    outflows = np.random.normal(loc=800, scale=150, size=n_days)
    outflows[dates.day == 25] += np.random.normal(loc=5000, scale=1000, size=len(outflows[dates.day == 25]))
    df = pd.DataFrame({'date': dates, 'inflows': inflows.round(2), 'outflows': outflows.round(2)})
    df['net_cash_flow'] = df['inflows'] - df['outflows']
    df['cash_flow_problem'] = (df['net_cash_flow'] < 0).astype(int)
    df['lag_7_days_problem'] = df['cash_flow_problem'].shift(7).fillna(0).astype(int)
    df = df.dropna()
    X = df[['lag_7_days_problem']]
    y = df['cash_flow_problem']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Problem', 'Problem'], yticklabels=['No Problem', 'Problem'])
    plt.title('Cashflow Prediction Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_matrix_chart = encode_plot_to_base64(plt)

    # ROC Curve Plot
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    roc_curve_chart = encode_plot_to_base64(plt)

    model_coefficient = model.coef_[0][0]
    return revenue_chart, confusion_matrix_chart, roc_curve_chart, model_coefficient

# Flask Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials. Please try again.'
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login Interface</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #71b7e6, #9b59b6);
            color: white;
        }}
        .glass-card {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 2rem;
            width: 90%;
            max-width: 400px;
            text-align: center;
        }}
        .input-group {{
            position: relative;
            margin-bottom: 20px;
        }}
        .input-group:focus-within {{
            color: #e4d836;
        }}
        input {{
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            outline: none;
        }}
        input::placeholder {{
            color: #ddd;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: #5cb85c;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
        }}
        .btn:hover {{
            background-color: #4cae4c;
        }}
    </style>
</head>
<body>
    <div class="glass-card">
        <h2>Login</h2>
        {f'<div style="color: red;">{error}</div>' if error else ''}
        <form method="post">
            <div class="input-group">
                <input type="text" name="username" placeholder="Username" required>
            </div>
            <div class="input-group">
                <input type="password" name="password" placeholder="Password" required>
            </div>
            <button type="submit" class="btn">Login</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    revenue_chart = None
    confusion_matrix_chart = None
    roc_curve_chart = None
    model_coefficient = None

    if request.method == 'POST':
        try:
            revenue_2015_data = {
                'Net Income': float(request.form['net_income_2015']),
                'Revenue': float(request.form['revenue_2015']),
                'Total_Assets': float(request.form['total_assets_2015']),
                'Cashflow': float(request.form['cashflow_2015'])
            }
            budget_data = {
                'Net Income': float(request.form['net_income_budget']),
                'Revenue': float(request.form['revenue_budget']),
                'Total_Assets': float(request.form['total_assets_budget']),
                'Cashflow': float(request.form['cashflow_budget'])
            }
            revenue_2016_data = {
                'Net Income': float(request.form['net_income_2016']),
                'Revenue': float(request.form['revenue_2016']),
                'Total_Assets': float(request.form['total_assets_2016']),
                'Cashflow': float(request.form['cashflow_2016'])
            }
            
            revenue_chart, confusion_matrix_chart, roc_curve_chart, model_coefficient = analyze_cashflow(
                revenue_2015_data, budget_data, revenue_2016_data
            )

            session['revenue_chart'] = revenue_chart
            session['confusion_matrix_chart'] = confusion_matrix_chart
            session['roc_curve_chart'] = roc_curve_chart
            session['model_coefficient'] = model_coefficient
        except (ValueError, KeyError) as e:
            return f"Error: Invalid or missing form data. Please ensure all fields are filled. Error details: {e}", 400
    else:
        revenue_chart = session.get('revenue_chart')
        confusion_matrix_chart = session.get('confusion_matrix_chart')
        roc_curve_chart = session.get('roc_curve_chart')
        model_coefficient = session.get('model_coefficient')

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cashflow Analyzer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #71b7e6, #9b59b6);
            color: white;
            min-height: 100vh;
        }}
        .container {{
            width: 80%;
            margin: auto;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}
        input[type=number] {{
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
        }}
        button {{
            background-color: #5cb85c;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #4cae4c;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin-top: 20px;
            background-color: #fff;
            padding: 10px;
            border-radius: 8px;
        }}
        .chart-container {{
            margin-bottom: 30px;
        }}
        .page {{
            display: none;
        }}
        .page:target {{
            display: block;
        }}
        .navigation {{
            margin-bottom: 20px;
            text-align: center;
        }}
        .navigation a {{
            color: white;
            text-decoration: none;
            padding: 10px 15px;
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            margin: 0 5px;
            transition: background-color 0.3s ease;
        }}
        .navigation a:hover {{
             background-color: rgba(255, 255, 255, 0.4);
        }}
        .analyze-button-container {{
            text-align: center;
            margin-top: 20px;
        }}
        .analyze-button {{
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }}
        .chart-container h2 {{
             color: #fff;
             text-align: center;
             margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="navigation">
            <a href="#page1">Input Data</a> | <a href="#page2">View Analysis</a>
        </div>

        <div id="page1" class="page">
            <h1>Cashflow Analysis</h1>
            <form method="post" action="#page2">
                <h2>Previous Year Data</h2>
                <input type="number" name="net_income_2015" placeholder="Net Income" required>
                <input type="number" name="revenue_2015" placeholder="Revenue" required>
                <input type="number" name="total_assets_2015" placeholder="Total Assets" required>
                <input type="number" name="cashflow_2015" placeholder="Cashflow" required>

                <h2>Budget Data</h2>
                <input type="number" name="net_income_budget" placeholder="Net Income" required>
                <input type="number" name="revenue_budget" placeholder="Revenue" required>
                <input type="number" name="total_assets_budget" placeholder="Total Assets" required>
                <input type="number" name="cashflow_budget" placeholder="Cashflow" required>

                <h2>Current Year Data</h2>
                <input type="number" name="net_income_2016" placeholder="Net Income" required>
                <input type="number" name="revenue_2016" placeholder="Revenue" required>
                <input type="number" name="total_assets_2016" placeholder="Total Assets" required>
                <input type="number" name="cashflow_2016" placeholder="Cashflow" required>

                <div class="analyze-button-container">
                    <button type="submit" class="analyze-button">Analyze</button>
                </div>
            </form>
        </div>

        <div id="page2" class="page">
            {f'<div class="chart-container"><h2>Revenue Comparison</h2><img src="data:image/png;base64,{revenue_chart}" alt="Revenue Comparison Chart"></div>' if revenue_chart else ''}
            
            {f'<div class="chart-container"><h2>Cashflow Prediction Confusion Matrix</h2><img src="data:image/png;base64,{confusion_matrix_chart}" alt="Confusion Matrix Chart"></div>' if confusion_matrix_chart else ''}

            {f'<div class="chart-container"><h2>ROC Curve</h2><img src="data:image/png;base64,{roc_curve_chart}" alt="ROC Curve Chart"></div>' if roc_curve_chart else ''}

            {f'<p style="text-align: center;">Model Coefficient for lag_7_days_problem: {model_coefficient}</p>' if model_coefficient else ''}
        </div>
    </div>
</body>
</html>
"""

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)