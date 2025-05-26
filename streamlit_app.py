import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials
import numpy as np # For handling potential division by zero with np.nan

# --- 0. Configuration & Page Setup ---
st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("💰 Financial & Operational Insights")
st.markdown("---")

# --- Define Color Scheme for Margin Categories (Color-blind friendly) ---
MARGIN_COLOR_MAP = {
    'Low': '#E69F00',      # Orange
    'Medium': '#56B4E9',   # Sky Blue
    'High': '#0072B2',     # Darker Blue
    'N/A': 'lightgrey',
    pd.NA: 'lightgrey', 
    'Unknown': 'lightgrey' 
}
# Define colors for flags/thresholds (Bright text for dark mode)
FLAG_COLORS = {
    'profit_margin_low': 'color: #FF4B4B; font-weight: bold;', # Bright Red text
    'ehr_low': 'color: #FFDB58; font-weight: bold;',          # Bright Yellow text
    'labor_cost_high': 'color: #FFA500; font-weight: bold;'   # Bright Orange text
}
# Define colors for Quadrant Chart
QUADRANT_COLOR_MAP = {
    "⭐ Gold Accounts (High Margin, Low Cost)": '#FFD700',      # Gold
    "🚫 Rebid/Drop (Low Margin, High Cost)": '#DC143C',      # Crimson
    "🤔 Underpriced? (Low Margin, Low Cost)": '#FFA500',   # Orange
    "🚧 Watch Carefully (High Margin, High Cost)": '#1E90FF', # DodgerBlue
    "Undefined Quadrant": 'grey' # Fallback
}
# Define colors for Stacked Bar Chart
STACKED_BAR_COLORS = {
    'Franchise Fee Amount': '#FFC300', # A distinct yellow/gold
    'Calculated Labor Cost': '#C70039', # A distinct red
    'Absolute Profit': '#3D9970'  # A distinct green (used for Net Profit)
}
# Define KPI Goals
KPI_GOALS = {
    "Avg Profit Margin": 0.30, # 30%
}
UP_ARROW = "⬆️"
DOWN_ARROW = "⬇️"
NEUTRAL_ARROW = "➖" 


# --- 1. Fetch raw data ---
@st.cache_data(ttl=600) 
def load_data(sheet_key):
    try:
        # Check if running in Streamlit Cloud and secrets are available
        if hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = Credentials.from_service_account_info(
                creds_dict,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets.readonly',
                    'https://www.googleapis.com/auth/drive.readonly'
                ]
            )
            # st.sidebar.info("Using credentials from Streamlit Secrets.") # Optional: for debugging deployment
        else:
            # Fallback to local creds.json file
            creds = Credentials.from_service_account_file(
                'creds.json',
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets.readonly',
                    'https://www.googleapis.com/auth/drive.readonly'
                ]
            )
            # st.sidebar.info("Using local creds.json file.") # Optional: for local debugging

        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(sheet_key)
        sheet = spreadsheet.get_worksheet(0)
        data_as_list = sheet.get_all_values()
        if not data_as_list or len(data_as_list) < 2:
            st.error("No data (or only headers) found in the Google Sheet. Please check the sheet.")
            return pd.DataFrame()
        
        headers = [str(header).strip() for header in data_as_list[0]]
        df = pd.DataFrame(data_as_list[1:], columns=headers)
        df.replace('', np.nan, inplace=True) 
        return df

    except FileNotFoundError: # Specifically for local creds.json
        st.error("Error: `creds.json` not found for local execution. If deployed, ensure Streamlit Secrets are set.")
        return pd.DataFrame()
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Error: Spreadsheet with key '{sheet_key}' not found. Please verify the Google Sheet key.")
        return pd.DataFrame()
    except Exception as e: # Catch other potential errors, including auth errors with st.secrets
        st.error(f"An error occurred while fetching/processing data: {e}")
        # If it's an auth error from st.secrets, e might contain a tuple like ('invalid_grant:...')
        if isinstance(e, tuple) and len(e) > 0 and 'invalid_grant' in str(e[0]):
             st.error("This looks like an authentication error with Google. If deployed, please double-check your Streamlit Secrets for the 'gcp_service_account' and ensure they are correctly formatted in TOML. Regenerating your service account key might be necessary.")
        return pd.DataFrame()

SPREADSHEET_KEY = '1UX1sHkfhJiW0jofPl6W8DCQoeMDoBHtmBFy44mHc9mU'
df_original = load_data(SPREADSHEET_KEY) 

if df_original.empty:
    st.warning("Could not load data. Dashboard cannot be displayed.")
    st.stop()

# --- Helper function for identifying columns by aliases ---
def find_alias(aliases, df_columns_list_to_search):
    if not isinstance(df_columns_list_to_search, list) or not all(isinstance(item, str) for item in df_columns_list_to_search):
        return None
    for alias in aliases:
        if alias in df_columns_list_to_search: 
            return alias
    return None

# --- 2. Identify Textual Key Columns BEFORE extensive cleaning ---
initial_df_cols_list = df_original.columns.tolist()
contract_col_name_actual = find_alias(['Contract name'], initial_df_cols_list) 
known_text_column_names = [col for col in [contract_col_name_actual] if col is not None]


# --- 3. Data Cleaning and Numeric Conversion ---
df_processed = df_original.copy()

def sanitize_value(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return value
    if isinstance(value, str):
        cleaned_value = value.replace('$', '').replace(',', '').replace('%', '').strip()
        if not cleaned_value: return np.nan
        return cleaned_value
    return value

for col_name_iter in df_processed.columns: 
    if col_name_iter not in known_text_column_names: 
        df_processed[col_name_iter] = df_processed[col_name_iter].apply(sanitize_value)
        df_processed[col_name_iter].replace(['nan', 'NaN', 'NAN', 'N/A', '#N/A', '<NA>'], np.nan, inplace=True)
        if not pd.api.types.is_numeric_dtype(df_processed[col_name_iter]):
            df_processed[col_name_iter] = pd.to_numeric(df_processed[col_name_iter], errors='coerce')
    elif col_name_iter == contract_col_name_actual: 
        df_processed[col_name_iter] = df_processed[col_name_iter].astype(str).str.strip()
        df_processed[col_name_iter].replace({'nan': np.nan, 'NaN': np.nan, 'NAN': np.nan, '<NA>': np.nan, '':np.nan, 'N/A': np.nan, 'None':np.nan}, inplace=True)


# --- 4. Identify specific columns from Google Sheet for calculations ---
df_cols_list_processed = df_processed.columns.tolist() 

revenue_col_actual = find_alias(['Contract Value After Franchise Fee', 'monthly Contract Value After Franchise Fee'], df_cols_list_processed) 
gross_revenue_col_actual = find_alias(['monthly Contract Value'], df_cols_list_processed) 
franchise_fee_pct_col_actual = find_alias(['Franchise Fee %'], df_cols_list_processed)
profit_target_pct_col_actual = find_alias(['Profit Target per contract after franchise fee %'], df_cols_list_processed)


num_employees_col_actual = find_alias(['Number of Employees'], df_cols_list_processed)
wage_col_actual = find_alias(['hourly wage per employee'], df_cols_list_processed) 
hrs_visit_col_actual = find_alias(['Hrs per Visit'], df_cols_list_processed)
visits_monthly_col_actual = find_alias(['Visits per month'], df_cols_list_processed) 

fee_col_actual = revenue_col_actual 

contract_col = contract_col_name_actual
revenue_col = revenue_col_actual 
visits_col = visits_monthly_col_actual
fee_col = fee_col_actual

# Adjust identified percentage columns to be decimals
if franchise_fee_pct_col_actual and franchise_fee_pct_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[franchise_fee_pct_col_actual]):
    if (df_processed[franchise_fee_pct_col_actual] > 1).any(): 
        df_processed[franchise_fee_pct_col_actual] = df_processed[franchise_fee_pct_col_actual] / 100.0

if profit_target_pct_col_actual and profit_target_pct_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[profit_target_pct_col_actual]):
    if (df_processed[profit_target_pct_col_actual] > 1).any(): 
        df_processed[profit_target_pct_col_actual] = df_processed[profit_target_pct_col_actual] / 100.0


# --- 4a. Calculate Franchise Fee Amount ---
df_processed['Franchise Fee Amount'] = np.nan
if gross_revenue_col_actual and franchise_fee_pct_col_actual and \
   gross_revenue_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[gross_revenue_col_actual]) and \
   franchise_fee_pct_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[franchise_fee_pct_col_actual]):
    df_processed['Franchise Fee Amount'] = df_processed[gross_revenue_col_actual] * df_processed[franchise_fee_pct_col_actual] 
else:
    if not (gross_revenue_col_actual and gross_revenue_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[gross_revenue_col_actual])):
        st.sidebar.warning(f"Gross revenue column ('{gross_revenue_col_actual}') for Franchise Fee calculation is missing or not numeric.")
    if not (franchise_fee_pct_col_actual and franchise_fee_pct_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[franchise_fee_pct_col_actual])):
        st.sidebar.warning(f"Franchise Fee % column ('{franchise_fee_pct_col_actual}') for Franchise Fee calculation is missing or not numeric.")


# --- 4b. Calculate Labor Cost ---
labor_col_calculated_name = "Calculated Labor Cost"
df_processed[labor_col_calculated_name] = np.nan 

components_present = {
    "num_emp": num_employees_col_actual and num_employees_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[num_employees_col_actual]),
    "wage": wage_col_actual and wage_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[wage_col_actual]), 
    "hrs_visit": hrs_visit_col_actual and hrs_visit_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[hrs_visit_col_actual]),
    "visits_month": visits_monthly_col_actual and visits_monthly_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[visits_monthly_col_actual])
}

df_processed[labor_col_calculated_name] = np.nan

if wage_col_actual and wage_col_actual in df_processed.columns: 
    owner_operated_mask = df_processed[wage_col_actual].isna()
    df_processed.loc[owner_operated_mask, labor_col_calculated_name] = 0

    standard_labor_mask = ~owner_operated_mask
    
    can_calculate_standard_labor = (
        components_present["num_emp"] and 
        components_present["wage"] and 
        components_present["hrs_visit"] and 
        components_present["visits_month"]
    )

    if can_calculate_standard_labor:
        df_processed.loc[standard_labor_mask, labor_col_calculated_name] = (
            df_processed.loc[standard_labor_mask, num_employees_col_actual] *
            df_processed.loc[standard_labor_mask, wage_col_actual] * df_processed.loc[standard_labor_mask, hrs_visit_col_actual] *
            df_processed.loc[standard_labor_mask, visits_monthly_col_actual]
        )
    elif standard_labor_mask.any(): 
        pass 
else:
    st.sidebar.warning(f"Input 'hourly wage per employee' ('{wage_col_actual}') not found or not numeric. Labor cost cannot be accurately determined.") 

labor_col = labor_col_calculated_name

# --- Critical Column Check ---
if not revenue_col: 
    st.error(f"Critical Error: Net Revenue column ('{revenue_col_actual}') could not be identified. Please check column names in your Google Sheet and aliases in the script.")
    st.stop()
if not (labor_col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[labor_col])):
    st.error(f"Critical Error: Calculated Labor Cost column ('{labor_col}') is not available or not numeric. Check warnings regarding its inputs.")
    st.stop()


# --- 5. Calculations (including new EHR and LER) ---
is_revenue_numeric = revenue_col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[revenue_col])
is_labor_numeric = labor_col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[labor_col]) 

df_processed['Profit Margin'] = np.nan 
df_processed['Absolute Profit'] = np.nan 
df_processed['Labor Cost %'] = np.nan
df_processed['EHR'] = np.nan 
df_processed['LER'] = np.nan 


if is_revenue_numeric and is_labor_numeric:
    denominator_revenue = df_processed[revenue_col].replace(0, np.nan) 
    df_processed['Absolute Profit'] = df_processed[revenue_col] - df_processed[labor_col] 
    df_processed['Profit Margin'] = df_processed['Absolute Profit'] / denominator_revenue 
    df_processed['Labor Cost %'] = df_processed[labor_col] / denominator_revenue 
    
    mask_zero_labor_positive_profit = (df_processed[labor_col] == 0) & (df_processed['Absolute Profit'] > 0)
    mask_standard_ler = ~( (df_processed[labor_col] == 0) ) 

    df_processed.loc[mask_zero_labor_positive_profit, 'LER_Display'] = "Owner/High Eff." 
    df_processed.loc[mask_standard_ler, 'LER'] = df_processed.loc[mask_standard_ler, 'Absolute Profit'] / df_processed.loc[mask_standard_ler, labor_col].replace(0,np.nan)


is_hrs_visit_numeric = hrs_visit_col_actual and hrs_visit_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[hrs_visit_col_actual])
is_visits_monthly_numeric = visits_monthly_col_actual and visits_monthly_col_actual in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[visits_monthly_col_actual])

if is_revenue_numeric and is_hrs_visit_numeric and is_visits_monthly_numeric:
    total_monthly_hours = df_processed[hrs_visit_col_actual] * df_processed[visits_monthly_col_actual]
    df_processed['EHR'] = df_processed[revenue_col] / total_monthly_hours.replace(0, np.nan) 


if fee_col and fee_col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[fee_col]):
    df_processed['MRR'] = df_processed[fee_col] 
else:
    df_processed['MRR'] = np.nan

is_visits_numeric = visits_col and visits_col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[visits_col])
if is_revenue_numeric and is_visits_numeric:
    denominator_visits = df_processed[visits_col].replace(0, np.nan) 
    df_processed['Revenue per Visit'] = df_processed[revenue_col] / denominator_visits 
else:
    df_processed['Revenue per Visit'] = np.nan

if is_labor_numeric and is_visits_numeric: 
    denominator_visits = df_processed[visits_col].replace(0, np.nan) 
    df_processed['Labor per Visit'] = df_processed[labor_col] / denominator_visits
else:
    df_processed['Labor per Visit'] = np.nan

if 'Revenue per Visit' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['Revenue per Visit']) and \
   'Labor per Visit' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['Labor per Visit']):
    df_processed['Profit per Visit'] = df_processed['Revenue per Visit'] - df_processed['Labor per Visit']
else:
    df_processed['Profit per Visit'] = np.nan

if 'Profit Margin' in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed['Profit Margin']):
    df_processed['Margin Category'] = pd.cut(df_processed['Profit Margin'].fillna(0), 
                                   bins=[-float('inf'), 0.15, 0.30, float('inf')], 
                                   labels=['Low', 'Medium', 'High'], right=False)
else:
    df_processed['Margin Category'] = pd.NA


# --- 6. Sidebar Filters ---
st.sidebar.header("Filters") 
margin_categories_options = []
if 'Margin Category' in df_processed.columns and df_processed['Margin Category'].notna().any():
    margin_categories_options = sorted([str(cat) for cat in df_processed['Margin Category'].dropna().unique().tolist() if str(cat) != '<NA>'])


selected_margin_categories = st.sidebar.multiselect(
    "Filter by Margin Category:", options=margin_categories_options, default=margin_categories_options
)
st.sidebar.markdown("---")
st.sidebar.markdown("##### Margin Category Legend:")
for category_key, color_value in MARGIN_COLOR_MAP.items(): 
    if isinstance(category_key, str) and category_key in ['Low', 'Medium', 'High']: 
        st.sidebar.markdown(f"<span style='color:{color_value}; font-weight:bold;'>◼</span> {category_key}", unsafe_allow_html=True)
st.sidebar.markdown("---")


df_filtered = df_processed.copy() 

if margin_categories_options: 
    if selected_margin_categories: 
        df_filtered_temp = df_processed.copy()
        df_filtered_temp['Margin Category String'] = df_filtered_temp['Margin Category'].astype(str).replace('<NA>', 'N/A').fillna('N/A')
        df_filtered = df_filtered_temp[df_filtered_temp['Margin Category String'].isin(selected_margin_categories)]
    else: 
        df_filtered = pd.DataFrame(columns=df_processed.columns) 


# --- 7. Dashboard UI (uses df_filtered) ---
if df_filtered.empty and selected_margin_categories: 
    st.warning("No data matches the selected filter criteria.")
elif df_filtered.empty and not df_processed.empty and margin_categories_options and not selected_margin_categories:
    st.info("Select margin categories from the sidebar to view data.")
elif df_processed.empty: 
    st.error("Processed data is empty. Cannot display dashboard.")
else: 
    # --- 7a. EHR and LER Explanations ---
    with st.expander("Understanding Key Strategic Metrics: EHR & LER"):
        st.markdown("""
        **Effective Hourly Revenue (EHR):**
        - **Measures:** The net revenue generated for every hour of labor spent on a contract.
        - **Formula:** `Net Revenue / (Hours per Visit × Visits per Month)`
        - **Importance:** EHR tells you how valuable each hour of your team's time is for a specific contract. A higher EHR indicates better ROI on labor time.
        - **Target:** Generally, higher is better. The dashboard flags EHR below $50/hr as needing attention.

        **Labor Efficiency Ratio (LER):**
        - **Measures:** How effectively your labor costs are converted into gross profit.
        - **Formula:** `Gross Profit (Absolute Profit) / Labor Cost`
        - **Importance:** LER shows the profitability of your labor.
            - **LER > 1.0:** You are making more gross profit than you spend on labor for that contract (profitable).
            - **LER = 1.0:** You are breaking even on labor.
            - **LER < 1.0:** You are losing money on labor for that contract.
            - **"Owner/High Eff.":** Indicates $0 labor cost with positive profit (e.g., owner-operated), meaning very high efficiency.
        - **Target:** Aim for LER significantly above 1.0. The higher, the more efficiently labor translates to profit.
        """)
    st.markdown("---")
    
    # --- 7b. KPIs ---
    kpi_metrics_data = []
    avg_profit_margin_val = df_filtered['Profit Margin'].mean() if 'Profit Margin' in df_filtered.columns and df_filtered['Profit Margin'].notna().any() else None
    
    if avg_profit_margin_val is not None:
        goal_pm = KPI_GOALS.get("Avg Profit Margin", 0.30) 
        delta_pm_val = avg_profit_margin_val - goal_pm
        delta_pm_arrow = UP_ARROW if delta_pm_val > 0 else (DOWN_ARROW if delta_pm_val < 0 else NEUTRAL_ARROW)
        delta_display_color = "normal" 
        if delta_pm_val < 0: delta_display_color = "inverse" 
        elif delta_pm_val == 0: delta_display_color = "off" 

        kpi_label_pm = f"Avg Profit Margin {delta_pm_arrow}"
        kpi_help_pm = f"Goal: {goal_pm:.0%}"
        kpi_metrics_data.append((kpi_label_pm, f"{avg_profit_margin_val:.1%}", kpi_help_pm, f"{delta_pm_val*100:.1f}% vs Goal" if delta_pm_val is not None else None, delta_display_color))


    if 'Labor Cost %' in df_filtered.columns and labor_col in df_filtered.columns:
        df_for_avg_labor_cost_pct = df_filtered[df_filtered[labor_col] > 0]
        if not df_for_avg_labor_cost_pct.empty and df_for_avg_labor_cost_pct['Labor Cost %'].notna().any():
            kpi_metrics_data.append(("Avg Labor Cost % (Paid Labor)", f"{df_for_avg_labor_cost_pct['Labor Cost %'].mean():.1%}", None, None, "off"))
    
    if 'EHR' in df_filtered.columns and df_filtered['EHR'].notna().any(): 
        kpi_metrics_data.append(("Avg EHR", f"${df_filtered['EHR'].mean():,.0f}/hr", None, None, "off")) 

    if 'LER' in df_filtered.columns and df_filtered['LER'].notna().any(): 
        df_for_avg_ler = df_filtered[(df_filtered[labor_col] > 0) & (df_filtered['LER'].notna())]
        if not df_for_avg_ler.empty:
             kpi_metrics_data.append(("Avg LER (Paid Labor)", f"{df_for_avg_ler['LER'].mean():.2f}", None, None, "off"))


    if 'MRR' in df_filtered.columns and df_filtered['MRR'].notna().any():
        kpi_metrics_data.append(("Total MRR", f"${df_filtered['MRR'].sum():,.0f}", None, None, "off"))
    if 'Absolute Profit' in df_filtered.columns and df_filtered['Absolute Profit'].notna().any():
        kpi_metrics_data.append(("Total Absolute Profit", f"${df_filtered['Absolute Profit'].sum():,.0f}", None, None, "off"))
    if 'Revenue per Visit' in df_filtered.columns and df_filtered['Revenue per Visit'].notna().any():
        kpi_metrics_data.append(("Avg Rev/Visit", f"${df_filtered['Revenue per Visit'].mean():,.0f}", None, None, "off")) 

    if kpi_metrics_data:
        num_kpis = len(kpi_metrics_data)
        kpi_cols = st.columns(num_kpis)
        for i, (label, val, help_text, delta_text, delta_color_status) in enumerate(kpi_metrics_data): 
            with kpi_cols[i]:
                st.metric(label=label, value=val, delta=delta_text, delta_color=delta_color_status)
                if help_text:
                    st.markdown(f"<span style='font-size: 0.8em; color: grey;'>{help_text}</span>", unsafe_allow_html=True)
    else: st.info("KPI data is currently unavailable for the selected filters.")
    st.markdown("---")

    # Contract Performance by Absolute Profit
    st.subheader("Contract Performance by Absolute Profit")
    abs_profit_available_filt = ('Absolute Profit' in df_filtered.columns and 
                            pd.api.types.is_numeric_dtype(df_filtered['Absolute Profit']) and 
                            df_filtered['Absolute Profit'].notna().any())
    contract_col_available_filt = contract_col and contract_col in df_filtered.columns and df_filtered[contract_col].notna().any()

    if contract_col_available_filt and abs_profit_available_filt:
        num_contracts_to_show = st.number_input("Number of Top/Bottom Contracts:", min_value=1, max_value=len(df_filtered) if len(df_filtered) > 0 else 20, value=5, key="num_abs_profit_tables") 
        
        cols_for_profit_tables = [contract_col, 'Absolute Profit Formatted', 'Profit Margin Formatted', 'Labor Cost % Formatted', 'EHR Formatted', 'LER Formatted'] 
        
        col_top, col_bottom = st.columns(2)

        with col_top:
            st.markdown("##### Top Contracts by Absolute Profit")
            top_n_profit = df_filtered.nlargest(num_contracts_to_show, 'Absolute Profit').copy()
            if not top_n_profit.empty:
                top_n_profit[contract_col] = top_n_profit[contract_col].astype(str).fillna("N/A")
                top_n_profit['Absolute Profit Formatted'] = top_n_profit['Absolute Profit'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                top_n_profit['Profit Margin Formatted'] = top_n_profit['Profit Margin'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                top_n_profit['Labor Cost % Formatted'] = top_n_profit['Labor Cost %'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A") 
                top_n_profit['EHR Formatted'] = top_n_profit['EHR'].apply(lambda x: f"${x:,.0f}/hr" if pd.notna(x) else "N/A") 
                top_n_profit['LER Formatted'] = top_n_profit.apply(lambda row: row['LER_Display'] if pd.notna(row.get('LER_Display')) else (f"{row['LER']:.2f}" if pd.notna(row['LER']) else "N/A"), axis=1)
                st.table(top_n_profit[cols_for_profit_tables].rename(columns={'Absolute Profit Formatted': 'Absolute Profit', 'Profit Margin Formatted': 'Profit Margin (%)', 'Labor Cost % Formatted': 'Labor Cost %', 'EHR Formatted': 'EHR', 'LER Formatted': 'LER'}))
            else: st.info("No Top Contracts data for current filters.")

        with col_bottom:
            st.markdown("##### Bottom Contracts by Absolute Profit")
            bottom_n_profit = df_filtered.nsmallest(num_contracts_to_show, 'Absolute Profit').copy()
            if not bottom_n_profit.empty:
                bottom_n_profit[contract_col] = bottom_n_profit[contract_col].astype(str).fillna("N/A")
                bottom_n_profit['Absolute Profit Formatted'] = bottom_n_profit['Absolute Profit'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                bottom_n_profit['Profit Margin Formatted'] = bottom_n_profit['Profit Margin'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                bottom_n_profit['Labor Cost % Formatted'] = bottom_n_profit['Labor Cost %'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A") 
                bottom_n_profit['EHR Formatted'] = bottom_n_profit['EHR'].apply(lambda x: f"${x:,.0f}/hr" if pd.notna(x) else "N/A") 
                bottom_n_profit['LER Formatted'] = bottom_n_profit.apply(lambda row: row['LER_Display'] if pd.notna(row.get('LER_Display')) else (f"{row['LER']:.2f}" if pd.notna(row['LER']) else "N/A"), axis=1)
                st.table(bottom_n_profit[cols_for_profit_tables].rename(columns={'Absolute Profit Formatted': 'Absolute Profit', 'Profit Margin Formatted': 'Profit Margin (%)', 'Labor Cost % Formatted': 'Labor Cost %', 'EHR Formatted': 'EHR', 'LER Formatted': 'LER'}))
            else: st.info("No Bottom Contracts data for current filters.")
    else: st.info("Absolute Profit or Contract Name data unavailable for this section.")
    st.markdown("---")
    
    # --- New Stacked Bar Chart Section ---
    st.subheader("🧱 Contract Cost & Profit Structure")
    stacked_bar_cols_present = (
        contract_col and contract_col in df_filtered.columns and
        'Franchise Fee Amount' in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered['Franchise Fee Amount']) and
        labor_col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[labor_col]) and 
        'Absolute Profit' in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered['Absolute Profit']) and 
        'Revenue per Visit' in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered['Revenue per Visit'])
    )

    if stacked_bar_cols_present:
        plot_data_stacked = df_filtered.dropna(subset=[contract_col, 'Franchise Fee Amount', labor_col, 'Absolute Profit', 'Revenue per Visit']).copy()
        plot_data_stacked[contract_col] = plot_data_stacked[contract_col].astype(str)
        
        if not plot_data_stacked.empty:
            fig_stacked = go.Figure()
            fig_stacked.add_trace(go.Bar(
                name='Franchise Fee', 
                x=plot_data_stacked[contract_col], 
                y=plot_data_stacked['Franchise Fee Amount'],
                marker_color=STACKED_BAR_COLORS['Franchise Fee Amount']
            ))
            fig_stacked.add_trace(go.Bar(
                name='Labor Cost', 
                x=plot_data_stacked[contract_col], 
                y=plot_data_stacked[labor_col],
                marker_color=STACKED_BAR_COLORS['Calculated Labor Cost']
            ))
            fig_stacked.add_trace(go.Bar(
                name='Gross Profit (Net)', 
                x=plot_data_stacked[contract_col], 
                y=plot_data_stacked['Absolute Profit'], # This is Gross Profit
                marker_color=STACKED_BAR_COLORS['Absolute Profit']
            ))

            fig_stacked.add_trace(go.Scatter(
                name='Revenue per Visit',
                x=plot_data_stacked[contract_col],
                y=plot_data_stacked['Revenue per Visit'],
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='rgba(0,0,255,0.7)'), 
                marker=dict(color='rgba(0,0,255,0.7)')
            ))

            fig_stacked.update_layout(
                barmode='stack',
                xaxis_title=contract_col if contract_col else "Contract",
                yaxis_title='Amount ($)',
                yaxis2=dict(
                    title='Revenue per Visit ($)',
                    overlaying='y',
                    side='right',
                    showgrid=False 
                ),
                legend_title_text='Cost/Profit Component',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) 
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
            st.caption("This chart breaks down each contract's original value. Stacked bars show Franchise Fee, Labor Cost, and Gross Profit (Net Profit after labor and franchise fee). The blue line (right axis) shows Revenue per Visit for comparison.")

        else:
            st.info("Not enough data to display the Contract Cost & Profit Structure chart after filtering.")
    else:
        st.info("One or more required columns (Franchise Fee Amount, Labor Cost, Absolute Profit, Revenue per Visit, Contract Name) are missing or not numeric. Cannot display structure chart.")
    st.markdown("---")


    # Lowest Margin Contracts (by %)
    st.subheader("Lowest Margin Contracts (Top 5 by %)")
    profit_margin_available_filt = ('Profit Margin' in df_filtered.columns and 
                               pd.api.types.is_numeric_dtype(df_filtered['Profit Margin']) and 
                               df_filtered['Profit Margin'].notna().any())
    if contract_col_available_filt and profit_margin_available_filt:
        low_margin_df = df_filtered[[contract_col, 'Profit Margin']].copy()
        low_margin_df[contract_col] = low_margin_df[contract_col].astype(str).fillna("N/A")
        low_margin_df['Profit Margin (%)'] = low_margin_df['Profit Margin'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        st.table(low_margin_df.nsmallest(5, 'Profit Margin')[[contract_col, 'Profit Margin (%)']])
    else: st.info("Low-Margin Contracts data cannot be displayed.")
    st.markdown("---")

    # Revenue vs Profit per Visit
    st.subheader("Revenue & Profit per Visit by Contract")
    rev_per_visit_available_filt = ('Revenue per Visit' in df_filtered.columns and 
                               pd.api.types.is_numeric_dtype(df_filtered['Revenue per Visit']) and 
                               df_filtered['Revenue per Visit'].notna().any())
    profit_per_visit_available_filt = ('Profit per Visit' in df_filtered.columns and 
                                  pd.api.types.is_numeric_dtype(df_filtered['Profit per Visit']) and 
                                  df_filtered['Profit per Visit'].notna().any())
    if contract_col_available_filt and rev_per_visit_available_filt and profit_per_visit_available_filt:
        plot_df = df_filtered.copy()
        plot_df[contract_col] = plot_df[contract_col].astype(str).fillna("Unknown")
        plot_df.dropna(subset=['Revenue per Visit', 'Profit per Visit'], inplace=True)
        if not plot_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=plot_df[contract_col], y=plot_df['Revenue per Visit'], name='Revenue/Visit', marker_color='#1f77b4')) 
            fig.add_trace(go.Bar(x=plot_df[contract_col], y=plot_df['Profit per Visit'], name='Profit/Visit', marker_color='#ff7f0e')) 
            fig.update_layout(barmode='group', xaxis_title=contract_col or "Contract", yaxis_title='Amount ($)')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("This chart compares the average revenue generated per visit (blue) with the average profit made per visit (orange) for each contract. It helps identify contracts that are efficient on a per-visit basis.")
        else: st.info("No data to plot for Revenue & Profit per Visit.")
    else: st.info("Revenue/Profit per Visit or Contract Name data unavailable.")
    st.markdown("---")

    # --- New Bar Chart: Profit Margin by Contract (Sorted) ---
    st.subheader("Profit Margin by Contract (Sorted)")
    if contract_col_available_filt and profit_margin_available_filt:
        bar_chart_df = df_filtered.dropna(subset=[contract_col, 'Profit Margin']).copy()
        bar_chart_df = bar_chart_df.sort_values(by='Profit Margin', ascending=True)
        
        if not bar_chart_df.empty:
            bar_chart_df['Performance'] = bar_chart_df['Profit Margin'].apply(lambda x: 'Above Target' if x >= 0.30 else 'Below Target')
            
            color_discrete_map_bar = {
                'Above Target': MARGIN_COLOR_MAP['High'], 
                'Below Target': '#FF4B4B' 
            }
            
            fig_bar_margin = px.bar(bar_chart_df, 
                                    x='Profit Margin', 
                                    y=contract_col, 
                                    orientation='h',
                                    color='Performance', 
                                    color_discrete_map=color_discrete_map_bar, 
                                    text='Profit Margin'
                                   )
            fig_bar_margin.update_layout(
                xaxis_title="Profit Margin (%)", 
                yaxis_title=contract_col if contract_col else "Contract",
                xaxis_tickformat=".0%"
            )
            fig_bar_margin.update_traces(texttemplate='%{x:.1%}', textposition='outside')
            st.plotly_chart(fig_bar_margin, use_container_width=True)
            st.caption("This chart displays the Profit Margin for each contract, sorted for easy comparison. Bars are colored based on whether they meet the 30% target profit margin (Darker Blue for >=30%, Bright Red for <30%).")
        else:
            st.info("No data to display for Profit Margin by Contract chart.")
    else:
        st.info("Profit Margin or Contract Name data unavailable for this chart.")
    st.markdown("---")


    # --- New Quadrant Chart ---
    st.subheader("Strategic Contract Quadrants: Margin vs. Labor Cost Efficiency")
    labor_cost_per_visit_available = ('Labor per Visit' in df_filtered.columns and 
                                      pd.api.types.is_numeric_dtype(df_filtered['Labor per Visit']) and 
                                      df_filtered['Labor per Visit'].notna().any())
    profit_margin_available_for_quadrant = ('Profit Margin' in df_filtered.columns and 
                                            pd.api.types.is_numeric_dtype(df_filtered['Profit Margin']) and 
                                            df_filtered['Profit Margin'].notna().any())

    if labor_cost_per_visit_available and profit_margin_available_for_quadrant and contract_col_available_filt:
        quadrant_df = df_filtered.dropna(subset=['Labor per Visit', 'Profit Margin', contract_col]).copy()
        
        if not quadrant_df.empty and len(quadrant_df) > 1 : 
            y_threshold_margin_quadrant = 0.30 
            x_threshold_labor_cost = quadrant_df['Labor per Visit'].median()
            
            if pd.notna(x_threshold_labor_cost): 
                def assign_quadrant(row):
                    margin = row['Profit Margin']
                    labor_cost_pv = row['Labor per Visit']
                    if pd.isna(margin) or pd.isna(labor_cost_pv):
                        return "Undefined Quadrant" 
                    if margin >= y_threshold_margin_quadrant and labor_cost_pv < x_threshold_labor_cost:
                        return "⭐ Gold Accounts (High Margin, Low Cost)"
                    elif margin < y_threshold_margin_quadrant and labor_cost_pv >= x_threshold_labor_cost:
                        return "🚫 Rebid/Drop (Low Margin, High Cost)"
                    elif margin < y_threshold_margin_quadrant and labor_cost_pv < x_threshold_labor_cost:
                        return "🤔 Underpriced? (Low Margin, Low Cost)"
                    elif margin >= y_threshold_margin_quadrant and labor_cost_pv >= x_threshold_labor_cost:
                        return "🚧 Watch Carefully (High Margin, High Cost)"
                    return "Undefined Quadrant"

                quadrant_df['Quadrant Category'] = quadrant_df.apply(assign_quadrant, axis=1)

                fig_quadrant = px.scatter(quadrant_df, 
                                          x='Labor per Visit', 
                                          y='Profit Margin', 
                                          color='Quadrant Category',
                                          size=visits_col if visits_col and visits_col in quadrant_df.columns and quadrant_df[visits_col].notna().any() and (quadrant_df[visits_col] > 0).any() else None,
                                          hover_name=contract_col,
                                          color_discrete_map=QUADRANT_COLOR_MAP,
                                          title="Contract Profitability vs. Labor Cost Efficiency Quadrants"
                                          )
                fig_quadrant.update_yaxes(tickformat=".0%") 
                fig_quadrant.add_hline(y=y_threshold_margin_quadrant, line_dash="dash", line_color="grey", annotation_text=f"{y_threshold_margin_quadrant:.0%} Margin Threshold", annotation_position="bottom right")
                fig_quadrant.add_vline(x=x_threshold_labor_cost, line_dash="dash", line_color="grey", annotation_text=f"${x_threshold_labor_cost:,.2f} Labor Cost/Visit Median", annotation_position="top left")

                x_min, x_max = quadrant_df['Labor per Visit'].min(), quadrant_df['Labor per Visit'].max()
                y_min, y_max = quadrant_df['Profit Margin'].min(), quadrant_df['Profit Margin'].max()

                if pd.notna(x_min) and pd.notna(x_max) and pd.notna(y_min) and pd.notna(y_max):
                    annotations = [
                        dict(x=(x_min + x_threshold_labor_cost) / 2 if pd.notna(x_min) else x_threshold_labor_cost * 0.5, y=(y_threshold_margin_quadrant + y_max) / 2 if pd.notna(y_max) else y_threshold_margin_quadrant * 1.5, text="⭐ Gold", showarrow=False, font=dict(size=10, color='black'), bgcolor='rgba(255,215,0,0.7)'),
                        dict(x=(x_threshold_labor_cost + x_max) / 2 if pd.notna(x_max) else x_threshold_labor_cost * 1.5, y=(y_min + y_threshold_margin_quadrant) / 2 if pd.notna(y_min) else y_threshold_margin_quadrant * 0.5, text="🚫 Rebid/Drop", showarrow=False, font=dict(size=10, color='white'), bgcolor='rgba(220,20,60,0.7)'),
                        dict(x=(x_min + x_threshold_labor_cost) / 2 if pd.notna(x_min) else x_threshold_labor_cost * 0.5, y=(y_min + y_threshold_margin_quadrant) / 2 if pd.notna(y_min) else y_threshold_margin_quadrant * 0.5, text="🤔 Underpriced?", showarrow=False, font=dict(size=10, color='black'), bgcolor='rgba(255,165,0,0.7)'),
                        dict(x=(x_threshold_labor_cost + x_max) / 2 if pd.notna(x_max) else x_threshold_labor_cost * 1.5, y=(y_threshold_margin_quadrant + y_max) / 2 if pd.notna(y_max) else y_threshold_margin_quadrant * 1.5, text="🚧 Watch", showarrow=False, font=dict(size=10, color='white'), bgcolor='rgba(30,144,255,0.7)')
                    ]
                    fig_quadrant.update_layout(annotations=annotations)
                
                st.plotly_chart(fig_quadrant, use_container_width=True)
                st.caption("""
                This quadrant chart plots contracts by their **Profit Margin (%)** vs. **Labor Cost per Visit ($)**.
                - The horizontal dashed line represents the **30% target profit margin**.
                - The vertical dashed line represents the **median Labor Cost per Visit** of the currently filtered contracts.
                **Strategic Zones:**
                - **⭐ Gold Accounts (High Margin, Low Cost):** Your most profitable and efficient contracts. Aim to replicate and retain.
                - **🚫 Rebid/Drop (Low Margin, High Cost):** These contracts are costly and unprofitable. Consider renegotiating terms, improving efficiency, or dropping if unfixable.
                - **🤔 Underpriced? (Low Margin, Low Cost):** Efficiently serviced but not yielding enough profit. Review pricing – you might be leaving money on the table.
                - **🚧 Watch Carefully (High Margin, High Cost):** Profitable now, but high labor costs could be a risk if revenue dips or costs rise. Investigate ways to improve labor efficiency.
                Dot size can represent visit volume.
                """)
            else:
                st.info("Median Labor Cost per Visit could not be calculated (likely due to insufficient data after filtering). Quadrant chart cannot be displayed.")
        else:
            st.info("No valid data to display the Quadrant Chart after filtering (need at least 2 contracts).")
    else:
        st.info("Labor Cost per Visit, Profit Margin, or Contract Name data is unavailable for the Quadrant Chart.")
    st.markdown("---")


    # Detailed Data View
    st.subheader("Detailed Data View (Filtered)")
    st.markdown("##### Data Table Legend:")
    legend_html = "<ul>"
    legend_html += f"<li><span style='{FLAG_COLORS['profit_margin_low']}'>Profit Margin &lt; 30%</span></li>" 
    legend_html += f"<li><span style='{FLAG_COLORS['ehr_low']}'>Effective Hourly Revenue (EHR) &lt; $50/hr</span></li>"
    legend_html += f"<li><span style='{FLAG_COLORS['labor_cost_high']}'>Labor Cost % &gt; 70%</span></li>"
    legend_html += "</ul>"
    st.markdown(legend_html, unsafe_allow_html=True)


    df_display_final = df_filtered.copy() 
    
    if contract_col and contract_col in df_display_final.columns:
        df_display_final[contract_col] = df_display_final[contract_col].astype(str).fillna("N/A")
    if 'Margin Category' in df_display_final.columns:
        df_display_final['Margin Category'] = df_display_final['Margin Category'].astype(str).replace('<NA>', "N/A").fillna("N/A")
        
    display_columns_ordered = [col_name for col_name in df_original.columns if col_name in df_display_final.columns] 
    calculated_cols_to_add = ['Calculated Labor Cost', 'Franchise Fee Amount', 'Absolute Profit', 'Profit Margin', 'Labor Cost %', 'MRR', 
                              'Revenue per Visit', 'Labor per Visit', 'Profit per Visit', 'Margin Category', 'EHR', 'LER'] 
    for cc in calculated_cols_to_add:
        if cc in df_display_final.columns and cc not in display_columns_ordered:
            display_columns_ordered.append(cc)
    
    display_columns_ordered = [col_name for col_name in display_columns_ordered if col_name in df_display_final.columns] 

    df_to_show_in_table = df_display_final[display_columns_ordered].copy() 

    cols_to_format_display = {
        'Absolute Profit': "${:,.0f}", 'Profit Margin': "{:.1%}", 'Labor Cost %': "{:.1%}", 'MRR': "${:,.0f}", 
        'Revenue per Visit': "${:,.0f}", 'Labor per Visit': "${:,.0f}", 'Profit per Visit': "${:,.0f}", 
        'Calculated Labor Cost': "${:,.0f}", 'Franchise Fee Amount': "${:,.0f}",
        'EHR': "${:,.0f}/hr" 
    }
    if revenue_col and revenue_col in df_to_show_in_table.columns: 
        cols_to_format_display[revenue_col] = "${:,.0f}"
    
    original_value_cols_from_sheet = [
        find_alias(['monthly Contract Value'], initial_df_cols_list), 
        find_alias(['Franchise Fee %'], initial_df_cols_list),
        find_alias(['Number of Employees'], initial_df_cols_list),
        find_alias(['hourly wage per employee'], initial_df_cols_list),
        find_alias(['Hrs per Visit'], initial_df_cols_list),
        find_alias(['Visits per month'], initial_df_cols_list),
    ]

    for orig_col_name in original_value_cols_from_sheet:
        if orig_col_name and orig_col_name in df_to_show_in_table.columns and \
           orig_col_name in df_original.columns and pd.api.types.is_numeric_dtype(df_original[orig_col_name]): 
            if '%' in orig_col_name: 
                cols_to_format_display[orig_col_name] = "{:.2%}" 
            elif 'Value' in orig_col_name or 'wage' in orig_col_name : 
                 cols_to_format_display[orig_col_name] = "${:,.0f}" 
            else: 
                cols_to_format_display[orig_col_name] = "{:,.0f}"


    for col_name_format, fmt_str in cols_to_format_display.items(): 
        if col_name_format in df_to_show_in_table.columns:
            if col_name_format in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col_name_format]):
                 df_to_show_in_table[col_name_format] = df_filtered[col_name_format].apply(lambda x: fmt_str.format(x) if pd.notna(x) else "N/A")
            else: 
                 df_to_show_in_table[col_name_format] = df_to_show_in_table[col_name_format].astype(str).fillna("N/A")
    
    if 'LER' in df_to_show_in_table.columns:
        if 'LER_Display' in df_to_show_in_table.columns:
            df_to_show_in_table['LER'] = df_to_show_in_table.apply(lambda row: row['LER_Display'] if pd.notna(row.get('LER_Display')) else (f"{row['LER']:.2f}" if pd.notna(row['LER']) else "N/A"), axis=1)
            if 'LER_Display' in df_to_show_in_table.columns.tolist() and 'LER_Display' in display_columns_ordered: 
                new_display_columns_ordered = [c for c in display_columns_ordered if c != 'LER_Display']
                if 'LER' not in new_display_columns_ordered and 'LER' in df_to_show_in_table.columns: 
                    try:
                        idx = display_columns_ordered.index('LER_Display')
                        new_display_columns_ordered.insert(idx, 'LER')
                    except ValueError:
                        if 'LER' not in new_display_columns_ordered: new_display_columns_ordered.append('LER')

                df_to_show_in_table = df_to_show_in_table[new_display_columns_ordered] 
        elif 'LER' in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered['LER']): 
             df_to_show_in_table['LER'] = df_filtered['LER'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        else: 
            df_to_show_in_table['LER'] = df_to_show_in_table['LER'].astype(str).fillna("N/A")


    # --- Styling for Flags ---
    def apply_styling_to_row(row_series_from_df_to_show):
        original_index = row_series_from_df_to_show.name 
        
        if original_index not in df_filtered.index:
            return [''] * len(row_series_from_df_to_show) 

        raw_row = df_filtered.loc[original_index]
        
        styles_for_row = [''] * len(row_series_from_df_to_show) 

        # Profit Margin < 30% (Updated threshold)
        if 'Profit Margin' in raw_row and pd.notna(raw_row['Profit Margin']) and raw_row['Profit Margin'] < 0.30:
            return [FLAG_COLORS['profit_margin_low']] * len(row_series_from_df_to_show)
        
        # EHR < $50/hr
        if 'EHR' in raw_row and pd.notna(raw_row['EHR']):
            if pd.api.types.is_numeric_dtype(raw_row['EHR']) and raw_row['EHR'] < 50: 
                 return [FLAG_COLORS['ehr_low']] * len(row_series_from_df_to_show)

        # Labor Cost % > 70%
        if 'Labor Cost %' in raw_row and pd.notna(raw_row['Labor Cost %']) and raw_row['Labor Cost %'] > 0.70:
            return [FLAG_COLORS['labor_cost_high']] * len(row_series_from_df_to_show)
            
        return styles_for_row 

    st.dataframe(df_to_show_in_table.style.apply(apply_styling_to_row, axis=1))
