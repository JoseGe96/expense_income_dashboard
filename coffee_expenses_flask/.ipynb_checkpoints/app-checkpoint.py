
from flask import Flask, render_template, request, redirect, url_for, flash, session
# import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import OrderedDict
import uuid



APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, 'data', 'expenses.csv')
PAYMENT_OPTIONS = ["HEY Negocios D", "HEY Negocios C", "HEY Personal D","HEY Personal C", "Banorte D","Banorte C","Rappi C", "Cash",]
RECURRENCE_OPTIONS = ["Fijo", "Variable"]
# --- Income CSV path ---
INCOME_PATH = os.path.join(APP_DIR, 'data', 'income.csv')

# Fee model: base fee rate + VAT (16%) applied on the fee
INCOME_CHANNELS = ["Card", "UberEats","Cash"]


app = Flask(__name__)
app.secret_key = 'dev-key-change-me'

def load_df():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if 'id' not in df.columns:
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
            save_df(df)
        # normalize dtypes
        if not df.empty:
            # Coerce date and amount
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
            # Fill missing text fields
            for col in ['expense_type','category','recurrence','card','description']:
                if col in df.columns:
                    df[col] = df[col].fillna('')
        else:
            df = pd.DataFrame(columns=['date','amount','expense_type','category','recurrence','card','description'])

        return df
    else:
        return pd.DataFrame(columns=['date','amount','expense_type','category','recurrence','card','description'])


def save_df(df):
    df.to_csv(DATA_PATH, index=False)

def load_income_df():
    if os.path.exists(INCOME_PATH):
        df = pd.read_csv(INCOME_PATH)
    else:
        df = pd.DataFrame(columns=['date','amount','channel','description'])
    
    if 'id' not in df.columns:
        df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        save_income_df(df)
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        for c in ['channel','description']:
            if c in df.columns:
                df[c] = df[c].fillna('').astype(str).str.strip()
    else:
        df = pd.DataFrame(columns=['date','amount','channel','description'])

    return df

def save_income_df(df):
    df.to_csv(INCOME_PATH, index=False)


def existing_expense_types():
    df = load_df()
    if df.empty or 'expense_type' not in df.columns:
        return []
    # unique, non-empty, sorted
    types = (
        df['expense_type']
        .dropna()
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(types, key=lambda s: s.lower())


@app.route('/', methods=['GET'])
def index():
    df = load_df()
    if not df.empty:
        df = df.sort_values(by='date', ascending=False)

    # Default date priority: session > latest in CSV > today
    default_date = session.get('last_date')
    if not default_date:
        if not df.empty and 'date' in df.columns:
            # df['date'] is date-like (string or date); pick the most recent non-empty
            try:
                default_date = pd.to_datetime(df['date'], errors='coerce').max()
                if pd.notna(default_date):
                    default_date = default_date.date().strftime('%Y-%m-%d')
            except Exception:
                default_date = None
    if not default_date:
        from datetime import datetime
        default_date = datetime.today().strftime('%Y-%m-%d')

    return render_template(
        'index.html',
        df=df,
        payment_options=PAYMENT_OPTIONS,       # if you added these earlier
        expense_types=existing_expense_types(),# if you added this earlier
        default_date=default_date,
        recurrence_options=RECURRENCE_OPTIONS,
    )

@app.route('/income', methods=['GET'])
def income():
    df = load_income_df()
    if not df.empty:
        df = df.sort_values(by='date', ascending=False)


    return render_template(
        'income.html',
        df=df,
        income_channels=INCOME_CHANNELS
    )

@app.route('/add_income', methods=['POST'])
def add_income():
    df = load_income_df()

    date_str = request.form.get('date') or ''
    amount_str = request.form.get('amount') or '0'
    channel = (request.form.get('channel') or 'Cash').strip()
    description = (request.form.get('description') or '').strip()

    # validate
    try:
        date_val = pd.to_datetime(date_str, errors='raise').date()
    except Exception:
        flash('Invalid date for income.', 'error')
        return redirect(url_for('income'))
    try:
        amount_val = float(amount_str)
    except Exception:
        flash('Income amount must be a number.', 'error')
        return redirect(url_for('income'))

    new_row = {
        'date': date_val.strftime('%Y-%m-%d'),
        'amount': amount_val,
        'channel': channel if channel in INCOME_CHANNELS else 'Cash',
        'description': description,
        'id': str(uuid.uuid4()),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_income_df(df)
    flash('Income added.', 'success')
    return redirect(url_for('income'))

@app.route('/income_edit/<rid>', methods=['GET'])
def income_edit(rid):
    df = load_income_df()
    mask = (df['id'].astype(str) == rid)
    if df.empty or not mask.any():
        flash('Income record not found.', 'error')
        return redirect(url_for('income'))

    row = df.loc[mask].iloc[0].copy()

    # Prefill date as YYYY-MM-DD
    date_val = pd.to_datetime(row['date'], errors='coerce')
    date_str = date_val.strftime('%Y-%m-%d') if pd.notna(date_val) else ''

    return render_template(
        'income_edit.html',
        rid=rid,
        row=row,
        date_value=date_str,
        income_channels=INCOME_CHANNELS
    )


@app.route('/income_update/<rid>', methods=['POST'])
def income_update(rid):
    df = load_income_df()
    mask = (df['id'].astype(str) == rid)
    if not mask.any():
        flash('Income record not found.', 'error')
        return redirect(url_for('income'))

    # read form
    date_str = request.form.get('date') or ''
    amount_str = request.form.get('amount') or '0'
    channel = (request.form.get('channel') or 'Cash').strip()
    description = (request.form.get('description') or '').strip()

    # validate/coerce
    try:
        date_val = pd.to_datetime(date_str, errors='raise').date()
    except Exception:
        flash('Invalid date.', 'error')
        return redirect(url_for('income_edit', rid=rid))
    try:
        amount_val = float(amount_str)
    except Exception:
        flash('Amount must be a number.', 'error')
        return redirect(url_for('income_edit', rid=rid))

    # update
    df.loc[mask, ['date','amount','channel','description']] = [
        date_val.strftime('%Y-%m-%d'),
        amount_val,
        channel if channel in INCOME_CHANNELS else 'Cash',
        description
    ]
    save_income_df(df)
    flash('Income updated.', 'success')
    return redirect(url_for('income'))


@app.route('/income_delete/<rid>', methods=['POST'])
def income_delete(rid):
    df = load_income_df()
    before = len(df)
    df = df[df['id'].astype(str) != rid]
    if len(df) == before:
        flash('Income record not found.', 'error')
    else:
        save_income_df(df)
        flash('Income deleted.', 'success')
    return redirect(url_for('income'))



@app.route('/add', methods=['POST'])
def add():
    df = load_df()

    date_str = request.form.get('date')
    amount_str = request.form.get('amount')
    expense_type = request.form.get('expense_type', '').strip()
    category = request.form.get('category', 'Business').strip()
    recurrence = request.form.get('recurrence', 'Variable').strip()
    card = request.form.get('card', '').strip()
    description = request.form.get('description', '').strip()

    # Validate
    try:
        date_val = datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else datetime.today().date()
    except ValueError:
        flash('Invalid date format. Use YYYY-MM-DD.', 'error')
        return redirect(url_for('index'))
    try:
        amount_val = float(amount_str)
    except (TypeError, ValueError):
        flash('Amount must be a number.', 'error')
        return redirect(url_for('index'))
    session['last_date'] = date_val.strftime('%Y-%m-%d')
    # Append
    new_row = {
        'date': date_val.strftime('%Y-%m-%d'),
        'amount': amount_val,
        'expense_type': expense_type or 'Uncategorized',
        'category': category or 'Business',
        'recurrence': recurrence or 'Variable',
        'card': card or 'Cash',
        'description': description,
        'id': str(uuid.uuid4()),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_df(df)
    flash('Expense added.', 'success')
    return redirect(url_for('index'))

@app.route('/edit/<rid>', methods=['GET'])
def edit(rid):
    df = load_df()
    if df.empty or 'id' not in df.columns or rid not in set(df['id'].astype(str)):
        flash('Expense not found.', 'error')
        return redirect(url_for('index'))

    row = df.loc[df['id'].astype(str) == rid].iloc[0].copy()

    # Prefill values
    date_val = pd.to_datetime(row['date'], errors='coerce')
    date_str = date_val.date().strftime('%Y-%m-%d') if pd.notna(date_val) else ''

    return render_template(
        'edit.html',
        rid=rid,
        row=row,
        date_value=date_str,
        payment_options=PAYMENT_OPTIONS,          # if you’re using these
        recurrence_options=RECURRENCE_OPTIONS,    # ["Fixed","Variable"]
        expense_types=existing_expense_types()    # if you added this helper
    )
@app.route('/update/<rid>', methods=['POST'])
def update(rid):
    df = load_df()
    mask = (df['id'].astype(str) == rid)
    if not mask.any():
        flash('Expense not found.', 'error')
        return redirect(url_for('index'))

    # read form
    date_str = request.form.get('date') or ''
    amount_str = request.form.get('amount') or '0'
    expense_type = (request.form.get('expense_type') or 'Uncategorized').strip()
    category = (request.form.get('category') or 'Business').strip()
    recurrence = (request.form.get('recurrence') or 'Variable').strip()
    card = (request.form.get('card') or 'Cash').strip()
    description = (request.form.get('description') or '').strip()

    # coerce
    try:
        date_val = pd.to_datetime(date_str, errors='raise').date()
    except Exception:
        flash('Invalid date.', 'error')
        return redirect(url_for('edit', rid=rid))
    try:
        amount_val = float(amount_str)
    except Exception:
        flash('Amount must be a number.', 'error')
        return redirect(url_for('edit', rid=rid))

    # update row
    df.loc[mask, ['date','amount','expense_type','category','recurrence','card','description']] = [
        date_val.strftime('%Y-%m-%d'),
        amount_val,
        expense_type,
        category,
        recurrence,
        card,
        description
    ]
    save_df(df)
    flash('Expense updated.', 'success')
    return redirect(url_for('index'))

@app.route('/delete/<rid>', methods=['POST'])
def delete(rid):
    df = load_df()
    before = len(df)
    df = df[df['id'].astype(str) != rid]
    if len(df) == before:
        flash('Expense not found.', 'error')
    else:
        save_df(df)
        flash('Expense deleted.', 'success')
    return redirect(url_for('index'))

@app.route('/income_dashboard', methods=['GET'])
def income_dashboard():
    from datetime import datetime
    df = load_income_df()

    # --- Date range filter (reuse same pattern) ---
    start_str = request.args.get('start', '')
    end_str   = request.args.get('end', '')
    dft = df.copy()
    dft['date'] = pd.to_datetime(dft['date'], errors='coerce')

    min_date = dft['date'].min()
    max_date = dft['date'].max()

    try:
        start_dt = datetime.strptime(start_str, '%Y-%m-%d').date() if start_str else (min_date.date() if pd.notna(min_date) else None)
    except ValueError:
        start_dt = min_date.date() if pd.notna(min_date) else None
    try:
        end_dt = datetime.strptime(end_str, '%Y-%m-%d').date() if end_str else (max_date.date() if pd.notna(max_date) else None)
    except ValueError:
        end_dt = max_date.date() if pd.notna(max_date) else None
    if start_dt and end_dt and end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    if start_dt:
        df = df[pd.to_datetime(df['date'], errors='coerce') >= pd.Timestamp(start_dt)]
    if end_dt:
        df = df[pd.to_datetime(df['date'], errors='coerce') <= pd.Timestamp(end_dt)]

    start_val = start_dt.strftime('%Y-%m-%d') if start_dt else ''
    end_val   = end_dt.strftime('%Y-%m-%d') if end_dt else ''

    # ---------------- KPIs + aggregations (NET = amount as stored) ----------------
    if df.empty:
        charts = {
            'total_net': 0.0,
            'by_channel_labels': [], 'by_channel_net': [],
            'monthly_labels': [], 'monthly_net': []
        }
        return render_template('income_dashboard.html', charts=charts, start_val=start_val, end_val=end_val)

    tmp = df.copy()
    total_net = float(tmp['amount'].sum())

    by_channel = tmp.groupby('channel')['amount'].sum().sort_values(ascending=False)

    dmt = tmp.copy()
    dmt['date'] = pd.to_datetime(dmt['date'], errors='coerce')
    dmt = dmt.dropna(subset=['date'])
    dmt['year_month'] = dmt['date'].dt.to_period('M').astype(str)
    monthly = dmt.groupby('year_month')['amount'].sum().sort_index()

    charts = {
        'total_net': round(total_net, 2),
        'by_channel_labels': list(by_channel.index.astype(str)),
        'by_channel_net': [round(float(x), 2) for x in by_channel.values],
        'monthly_labels': list(monthly.index.astype(str)),
        'monthly_net': [round(float(x), 2) for x in monthly.values],
    }
    return render_template('income_dashboard.html', charts=charts, start_val=start_val, end_val=end_val)




@app.route('/dashboard', methods=['GET'])
def dashboard():
    df = load_df()

    # --- Date range filter (GET params: start=YYYY-MM-DD, end=YYYY-MM-DD) ---
    start_str = request.args.get('start', '')
    end_str   = request.args.get('end', '')
    
    df_dates = df.copy()
    df_dates['date'] = pd.to_datetime(df_dates['date'], errors='coerce')
    
    # Defaults: full range in the data
    min_date = df_dates['date'].min()
    max_date = df_dates['date'].max()
    
    # Parse inputs (fall back to defaults)
    try:
        start_dt = datetime.strptime(start_str, '%Y-%m-%d').date() if start_str else (min_date.date() if pd.notna(min_date) else None)
    except ValueError:
        start_dt = min_date.date() if pd.notna(min_date) else None
    
    try:
        end_dt = datetime.strptime(end_str, '%Y-%m-%d').date() if end_str else (max_date.date() if pd.notna(max_date) else None)
    except ValueError:
        end_dt = max_date.date() if pd.notna(max_date) else None
    
    # Safety: if swapped, fix
    if start_dt and end_dt and end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    
    # Apply filter (inclusive)
    if start_dt:
        df = df[df_dates['date'] >= pd.Timestamp(start_dt)]
    if end_dt:
        df = df[df_dates['date'] <= pd.Timestamp(end_dt)]
    
    # Keep string values to prefill the form
    start_val = start_dt.strftime('%Y-%m-%d') if start_dt else ''
    end_val   = end_dt.strftime('%Y-%m-%d') if end_dt else ''

    if df.empty:
        # Provide empty structures for Chart.js
        charts = {
            'by_category_labels': [], 'by_category_values': [],
            'by_type_labels': [], 'by_type_values': [],
            'by_card_labels': [], 'by_card_values': [],
            'monthly_labels': [], 'monthly_values': [],
            'total_spend': 0.0,
            'business_spend': 0.0,
            'personal_spend': 0.0
        }
        return render_template('dashboard.html',charts=charts,df=df,recurrence_options=RECURRENCE_OPTIONS,start_val=start_val,end_val=end_val)


    # KPIs
    total_spend = float(df['amount'].sum())
    business_spend = float(df.loc[df['category'].str.lower()=='business','amount'].sum())
    personal_spend = float(df.loc[df['category'].str.lower()=='personal','amount'].sum())

    # Aggregations
    by_category = df.groupby('category', dropna=False)['amount'].sum().sort_values(ascending=False)
    by_type = df.groupby('expense_type', dropna=False)['amount'].sum().sort_values(ascending=False).head(12)
    by_card = df.groupby('card', dropna=False)['amount'].sum().sort_values(ascending=False)
    by_recurrence = df.groupby('recurrence', dropna=False)['amount'].sum().sort_values(ascending=False)


    # Monthly trend
    dft = df.copy()
    dft['date'] = pd.to_datetime(dft['date'], errors='coerce')
    dft = dft.dropna(subset=['date'])
    dft['year_month'] = dft['date'].dt.to_period('M').astype(str)
    monthly = dft.groupby('year_month')['amount'].sum().sort_index()

    charts = {
        'by_category_labels': list(by_category.index.astype(str)),
        'by_category_values': [round(float(x),2) for x in by_category.values],
        'by_type_labels': list(by_type.index.astype(str)),
        'by_type_values': [round(float(x),2) for x in by_type.values],
        'by_card_labels': list(by_card.index.astype(str)),
        'by_card_values': [round(float(x),2) for x in by_card.values],
        'monthly_labels': list(monthly.index.astype(str)),
        'monthly_values': [round(float(x),2) for x in monthly.values],
        'total_spend': round(total_spend,2),
        'business_spend': round(business_spend,2),
        'personal_spend': round(personal_spend,2),
        'by_recurrence_labels': list(by_recurrence.index.astype(str)),
        'by_recurrence_values': [round(float(x),2) for x in by_recurrence.values],
    }

    # ------ Business/Personal series for checkbox filtering ------
    df_b = df[df['category'].str.lower() == 'business'].copy()
    df_p = df[df['category'].str.lower() == 'personal'].copy()
    
    def vals_for(labels, series):
        d = series.to_dict()
        return [round(float(d.get(k, 0.0)), 2) for k in labels]
    
    # By Category (labels are literally Business/Personal)
    cat_labels = ['Business', 'Personal']
    cat_b = df_b.groupby('category')['amount'].sum()
    cat_p = df_p.groupby('category')['amount'].sum()
    
    # By Type (use your existing top-12 labels to keep order)
    type_labels = list(by_type.index.astype(str))
    type_b = df_b.groupby('expense_type')['amount'].sum()
    type_p = df_p.groupby('expense_type')['amount'].sum()
    
    # By Card (use your existing order)
    card_labels = list(by_card.index.astype(str))
    card_b = df_b.groupby('card')['amount'].sum()
    card_p = df_p.groupby('card')['amount'].sum()
    
    # Monthly (align to your existing monthly labels)
    month_labels = charts['monthly_labels']
    def monthly_series(frame):
        tmp = frame.copy()
        tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
        tmp = tmp.dropna(subset=['date'])
        tmp['year_month'] = tmp['date'].dt.to_period('M').astype(str)
        return tmp.groupby('year_month')['amount'].sum()
    
    month_b = monthly_series(df_b)
    month_p = monthly_series(df_p)
    
    charts.update({
        # category
        'cat_labels': cat_labels,
        'cat_business': vals_for(cat_labels, cat_b),
        'cat_personal': vals_for(cat_labels, cat_p),
        # type
        'type_labels': type_labels,
        'type_business': vals_for(type_labels, type_b),
        'type_personal': vals_for(type_labels, type_p),
        # card
        'card_labels': card_labels,
        'card_business': vals_for(card_labels, card_b),
        'card_personal': vals_for(card_labels, card_p),
        # monthly
        'monthly_business': vals_for(month_labels, month_b),
        'monthly_personal': vals_for(month_labels, month_p),
    })
    labels_rec = RECURRENCE_OPTIONS  # or use your RECURRENCE_OPTIONS if you defined them

    def series_to_values(s, labels):
        # s is a Series indexed by recurrence; return values for every label in order
        lookup = s.to_dict()
        return [round(float(lookup.get(l, 0.0)), 2) for l in labels]
    
    s_all = df.groupby('recurrence', dropna=False)['amount'].sum()
    s_bus = df[df['category'].str.lower() == 'business'].groupby('recurrence', dropna=False)['amount'].sum()
    s_per = df[df['category'].str.lower() == 'personal'].groupby('recurrence', dropna=False)['amount'].sum()
    
    charts.update({
        'recurrence_labels': labels_rec,
        'recurrence_all': series_to_values(s_all, labels_rec),
        'recurrence_business': series_to_values(s_bus, labels_rec),
        'recurrence_personal': series_to_values(s_per, labels_rec),
    })

    return render_template('dashboard.html', charts=charts, df=df,recurrence_options=RECURRENCE_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True)
