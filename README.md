# Coffee Business Dashboard

A **Flask + Pandas web app** to manage my family's coffee business finances.  
It provides an easy way to log expenses and income, edit records, and analyze trends with interactive Chart.js dashboards.

---

## ✨ Features

### Expenses
- Log expenses with:
  - Date, Amount, Type of expense
  - Category (Business/Personal)
  - Recurrence (Fixed/Variable)
  - Payment Method / Card
  - Description
- Editable table with **Edit/Delete** buttons
- Filters:
  - **Date range**
  - **Business/Personal checkboxes**
  - **Card checkboxes** for filtering the *By Type* chart
- Charts (interactive):
  - By Category
  - By Recurrence
  - By Type (Top 12)
  - By Card
  - Monthly Trend

### Income
- Log income in 3 channels:
  - **Cash**
  - **Card**
  - **UberEats**
- Editable table with **Edit/Delete** actions
- Income Dashboard:
  - KPIs: Total Net Income
  - Charts:
    - Net by Channel
    - Monthly Net Trend

---

## 🛠 Tech Stack
- **Backend:** Python, Flask
- **Data Handling:** Pandas
- **Frontend:** Jinja2 templates, Chart.js, custom CSS
- **Storage:** CSV files (`expenses.csv`, `income.csv`)

---

## 🚀 Usage
1. Clone the repo  
   ```bash
   git clone https://github.com/JoseGe96/coffee-dashboard.git
   cd coffee-dashboard
