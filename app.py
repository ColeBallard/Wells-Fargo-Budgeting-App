import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px

# Database connection
DB_FILE = 'budget.db'

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # To return rows as dicts
    return conn

# Create tables if they don't exist
with get_db_connection() as conn:
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    # --- Replace the CREATE TABLE for transactions ---
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            amount REAL NOT NULL,
            vendor TEXT NOT NULL,
            account_type TEXT NOT NULL,
            description TEXT,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
    ''')
    conn.commit()

# Helper functions
def insert_transaction(date, amount, vendor, account_type, description=None):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id FROM transactions
            WHERE date = ? AND amount = ? AND vendor = ? AND account_type = ?
        ''', (date, amount, vendor, account_type))
        if c.fetchone():
            return False
        c.execute('''
            INSERT INTO transactions (date, amount, vendor, account_type, description)
            VALUES (?, ?, ?, ?, ?)
        ''', (date, amount, vendor, account_type, description))
        conn.commit()
        return True

def get_all_transactions():
    with get_db_connection() as conn:
        df = pd.read_sql_query('''
            SELECT t.id, t.date, t.amount, t.vendor, t.account_type,
                t.description, c.name as category
            FROM transactions t
            LEFT JOIN categories c ON t.category_id = c.id
            ORDER BY t.date DESC
        ''', conn)
        df['date'] = pd.to_datetime(df['date'])
    return df

def get_categories():
    with get_db_connection() as conn:
        df = pd.read_sql_query('SELECT id, name FROM categories ORDER BY name', conn)
    return df

def add_category(name):
    with get_db_connection() as conn:
        c = conn.cursor()
        try:
            c.execute('INSERT INTO categories (name) VALUES (?)', (name,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Duplicate name

def delete_category(category_id):
    with get_db_connection() as conn:
        c = conn.cursor()
        # Set transactions to null first
        c.execute('UPDATE transactions SET category_id = NULL WHERE category_id = ?', (category_id,))
        c.execute('DELETE FROM categories WHERE id = ?', (category_id,))
        conn.commit()

def update_transaction_category(transaction_id, category_id):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('UPDATE transactions SET category_id = ? WHERE id = ?', (category_id, transaction_id))
        conn.commit()

# Streamlit App
st.title("Budgeting App")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Upload CSV", "View Transactions", "Manage Categories", "Charts"])

if page == "Upload CSV":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    account_type = st.radio("Account Type", ["checking", "credit"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Assume columns are 0: date, 1: amount, 4: vendor (0-indexed)
        if len(df.columns) < 5:
            st.error("CSV must have at least 5 columns.")
        else:
            df = df.iloc[:, [0, 1, 2, 4]]          # date, amount, col3 (desc), vendor
            df.columns = ['date', 'amount', 'description', 'vendor']
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna()  # Drop invalid rows

            success_count = 0
            for _, row in df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                if insert_transaction(date_str, row['amount'], row['vendor'],
                                    account_type, row['description']):
                    success_count += 1

            st.success(f"Uploaded {success_count} new transactions (skipped duplicates).")

elif page == "View Transactions":
    st.header("View and Edit Transactions")

    # ————— helpers for paging —————
    PAGE_SIZE = 50
    if "txn_page" not in st.session_state:
        st.session_state.txn_page = 0
    def goto_page(p):
        st.session_state.txn_page = max(0, p)

    df = get_all_transactions()
    if df.empty:
        st.info("No transactions yet.")
    else:
        # Categories
        categories = get_categories()
        category_options = {None: 'Uncategorized'}
        category_options.update({row['id']: row['name'] for _, row in categories.iterrows()})
        id_from_name = {v: k for k, v in category_options.items()}

        # Normalize
        df['category'] = df['category'].fillna('Uncategorized')
        df['description'] = df['description'].fillna('')

        # Stable index
        df = df.set_index('id', drop=False)

        # Keep an original copy in session_state for diffing/rendering
        if "orig_txn_df" not in st.session_state:
            st.session_state.orig_txn_df = df.copy()

        # PAGE SLICE
        total = len(st.session_state.orig_txn_df)
        max_page = max(0, (total - 1) // PAGE_SIZE)
        start = st.session_state.txn_page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        page_df = st.session_state.orig_txn_df.iloc[start:end]

        # Pagination controls
        cols = st.columns(4)
        with cols[0]:
            if st.button("⏮️ First", disabled=st.session_state.txn_page == 0):
                goto_page(0); st.rerun()
        with cols[1]:
            if st.button("◀️ Prev", disabled=st.session_state.txn_page == 0):
                goto_page(st.session_state.txn_page - 1); st.rerun()
        with cols[2]:
            if st.button("Next ▶️", disabled=st.session_state.txn_page >= max_page):
                goto_page(st.session_state.txn_page + 1); st.rerun()
        with cols[3]:
            if st.button("Last ⏭️", disabled=st.session_state.txn_page >= max_page):
                goto_page(max_page); st.rerun()
        st.caption(f"Showing rows {start+1}-{end} of {total} • Page {st.session_state.txn_page+1}/{max_page+1}")

        # EDITOR inside a FORM to prevent reruns while typing
        with st.form("txn_edit_form", clear_on_submit=False):
            edited_page_df = st.data_editor(
                page_df[['id','date','amount','vendor','account_type','description','category']],
                column_config={
                    "description": st.column_config.TextColumn("Description", width="large"),
                    "category": st.column_config.SelectboxColumn(
                        "Category", options=list(category_options.values()), required=False, width="medium"
                    ),
                },
                disabled=["id","date","amount","vendor","account_type"],
                key="transaction_editor",
                use_container_width=True,
                hide_index=True,
            )

            # Grab the active cell so we can keep your place after save
            editor_state = st.session_state.get("transaction_editor", {})
            active_row_offset = None
            sel = editor_state.get("selection") or {}
            active = sel.get("active_cell")
            if active and isinstance(active, dict) and "row" in active:
                active_row_offset = int(active["row"])  # row within the page slice

            save = st.form_submit_button("💾 Save changes")
            cancel = st.form_submit_button("↩️ Cancel unsaved edits")

        if cancel:
            # Discard page edits and rerender current page without jumping
            st.session_state.orig_txn_df.iloc[start:end] = df.iloc[start:end]
            st.rerun()

        if save:
            # Diff and persist only changed cells for the visible page
            changed_rows = []
            for rid in edited_page_df.index:  # these are actual ids (index = id)
                before = st.session_state.orig_txn_df.loc[rid]
                after = edited_page_df.loc[rid]
                changes = {}
                if before['description'] != after['description']:
                    changes['description'] = after['description']
                if before['category'] != after['category']:
                    changes['category'] = after['category']
                if changes:
                    changed_rows.append((int(rid), changes))

            if changed_rows:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    for rid, changes in changed_rows:
                        if 'description' in changes:
                            c.execute("UPDATE transactions SET description = ? WHERE id = ?",
                                    (changes['description'] or None, rid))
                        if 'category' in changes:
                            new_cat_id = id_from_name.get(changes['category'])
                            c.execute("UPDATE transactions SET category_id = ? WHERE id = ?",
                                    (new_cat_id, rid))
                    conn.commit()

                # Update in-memory snapshot for only the page slice
                st.session_state.orig_txn_df.iloc[start:end] = edited_page_df
                st.success("Changes saved.")

                # Keep your place:
                # If we know which row you were in, compute its absolute position and page, then go there.
                if active_row_offset is not None:
                    # Translate page-relative row to global id
                    try:
                        current_row_id = edited_page_df.iloc[active_row_offset]['id']
                        abs_pos = st.session_state.orig_txn_df.index.get_loc(current_row_id)
                        same_page = abs_pos // PAGE_SIZE
                        goto_page(same_page)
                    except Exception:
                        pass  # fallback: keep current page

                st.rerun()

        # Manual refresh still available (doesn’t kick you to page 1)
        if st.button("🔄 Refresh from Database"):
            df_fresh = get_all_transactions().set_index('id', drop=False)
            st.session_state.orig_txn_df = df_fresh.copy()
            # Try to keep current page within bounds
            total = len(st.session_state.orig_txn_df)
            max_page = max(0, (total - 1) // PAGE_SIZE)
            st.session_state.txn_page = min(st.session_state.txn_page, max_page)
            st.rerun()

elif page == "Manage Categories":
    st.header("Manage Categories")

    # Add new category
    with st.form("Add Category"):
        new_name = st.text_input("Category Name")
        submit = st.form_submit_button("Add")
        if submit and new_name:
            if add_category(new_name):
                st.success(f"Added category: {new_name}")
            else:
                st.error("Category name already exists.")

    # List categories
    categories = get_categories()
    if not categories.empty:
        for _, row in categories.iterrows():
            col1, col2 = st.columns([3, 1])
            col1.write(row['name'])
            if col2.button("Delete", key=f"del_{row['id']}"):
                delete_category(row['id'])
                st.success(f"Deleted category: {row['name']}")
                st.rerun()  # Refresh
    else:
        st.info("No categories yet.")

elif page == "Charts":
    st.header("Charts and Visualizations")
    df = get_all_transactions()
    if df.empty:
        st.info("No transactions yet.")
    else:
        # Filters
        st.subheader("Filters")
        account_types = st.multiselect("Account Types", ['checking', 'credit'], default=['checking', 'credit'])
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        start_date, end_date = st.date_input("Date Range", [min_date, max_date])
        amount_min, amount_max = st.slider("Amount Range", float(df['amount'].min()), float(df['amount'].max()), (float(df['amount'].min()), float(df['amount'].max())))
        categories = get_categories()['name'].tolist()
        selected_categories = st.multiselect("Categories", categories, default=categories)
        include_uncategorized = st.checkbox("Include Uncategorized", value=True)

        # Normalize category
        df['category'] = df['category'].fillna('Uncategorized')

        # Filter df
        filtered_df = df[
            (df['account_type'].isin(account_types)) &
            (df['date'] >= pd.to_datetime(start_date)) &
            (df['date'] <= pd.to_datetime(end_date)) &
            (df['amount'] >= amount_min) &
            (df['amount'] <= amount_max) &
            (df['category'].isin(selected_categories) | (df['category'] == 'Uncategorized') if include_uncategorized else df['category'].isin(selected_categories))
        ]

        if filtered_df.empty:
            st.info("No data matches filters.")
        else:
            # Add month column for grouping
            filtered_df['month'] = filtered_df['date'].dt.to_period('M').astype(str)

            # Chart 1: Monthly Total Amounts (Line Chart)
            st.subheader("Monthly Total Amounts")
            monthly_totals = filtered_df.groupby('month')['amount'].sum().reset_index()
            fig1 = px.line(monthly_totals, x='month', y='amount', title='Total Amounts by Month')
            st.plotly_chart(fig1)

            # Chart 2: Expenses vs Income (assuming negative=expense, positive=income)
            st.subheader("Expenses vs Income")
            filtered_df['type'] = filtered_df['amount'].apply(lambda x: 'Expense' if x < 0 else 'Income')
            type_totals = filtered_df.groupby(['month', 'type'])['amount'].sum().abs().reset_index()  # Abs for expenses
            fig2 = px.bar(type_totals, x='month', y='amount', color='type', barmode='group', title='Expenses vs Income by Month')
            st.plotly_chart(fig2)

            # Chart 3: Pie Chart by Category (for expenses)
            st.subheader("Expenses by Category (Pie Chart)")
            expenses_df = filtered_df[filtered_df['amount'] < 0].copy()
            expenses_df['amount'] = expenses_df['amount'].abs()  # Make positive for pie
            if not expenses_df.empty:
                category_totals = expenses_df.groupby('category')['amount'].sum().reset_index()
                fig3 = px.pie(category_totals, values='amount', names='category', title='Expenses by Category')
                st.plotly_chart(fig3)
            else:
                st.info("No expenses in filtered data.")

            # Chart 4: Bar Chart by Vendor (top 10 expenses)
            st.subheader("Top 10 Vendors by Expense")
            vendor_totals = expenses_df.groupby('vendor')['amount'].sum().nlargest(10).reset_index()
            fig4 = px.bar(vendor_totals, x='vendor', y='amount', title='Top 10 Vendors by Expense Amount')
            st.plotly_chart(fig4)

            # Chart 5: Scatter Plot of Amounts over Time
            st.subheader("Amounts over Time (Scatter)")
            fig5 = px.scatter(filtered_df, x='date', y='amount', color='account_type', hover_data=['vendor', 'category'], title='Transactions over Time')
            st.plotly_chart(fig5)
