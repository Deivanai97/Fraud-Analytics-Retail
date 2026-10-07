import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import numpy as np
from backend import load_and_process_data, train_and_evaluate

# 1. IMPORT FROM BACKEND
from backend import load_and_process_data, train_and_evaluate

# 2. BACKEND INTEGRATION: Caching ensures the ML model doesn't retrain on every click
@st.cache_data
def fetch_data():
    # Calls the data pipeline from backend.py
    return load_and_process_data('retail_fraud_detection_100k.csv')

@st.cache_resource
def fetch_model_and_metrics(_df):
    # Calls the ML pipeline from backend.py
    return train_and_evaluate(_df)

# 3. EXECUTE THE CONNECTION: Runs the backend and stores the results
with st.spinner("Connecting to backend engine..."):
    raw_df, processed_df = fetch_data()
    model, explainer, X_test, y_test, idx_test, metrics, features = fetch_model_and_metrics(processed_df)

# --- Frontend Configuration ---
st.set_page_config(page_title="Fraud Analytics Dashboard", layout="wide")
st.markdown("""
    <style>
    .risk-high {background-color: #ffcccc; padding: 15px; border-radius: 8px; border-left: 5px solid #ff0000; color: #900;}
    .risk-low {background-color: #ccffcc; padding: 15px; border-radius: 8px; border-left: 5px solid #00cc00; color: #060;}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Fraud Analytics System")

# backend integration
@st.cache_data
def fetch_data():
    return load_and_process_data()

@st.cache_resource
def fetch_model_and_metrics(_df):
    return train_and_evaluate(_df)

with st.spinner("Connecting to backend engine..."):
    raw_df, processed_df = fetch_data()
    model, explainer, X_test, y_test, idx_test, metrics, features = fetch_model_and_metrics(processed_df)

# --- Dashboard Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Analysis", 
    "🎯 Model Evaluation", 
    "🧠 Global Risk Drivers", 
    "🔍 Live Inspector"
])

# ==========================================
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab1:
    
    # 1. DATA DICTIONARY
    with st.expander("📖 Data Dictionary (Click to expand)"):
        st.markdown("""
        **Transaction Details:**
        * **transaction_id:** Unique identifier for each transaction.
        * **customer_id:** Unique identifier for the customer making the transaction.
        * **transaction_timestamp:** The date and time the transaction occurred.
        * **transaction_amount:** The total monetary value of the transaction.
        * **payment_method:** The method used to pay (e.g., Credit Card, Debit Card, PayPal, Apple Pay).
        * **device_type:** The device used for the transaction (Mobile, Desktop, Tablet).
        * **location:** The geographic country/region where the transaction originated.
        * **merchant_category:** The retail sector of the purchase (e.g., Electronics, Fashion, Groceries).
        * **is_international:** Indicator (1/0) if the transaction crosses international borders.
        
        **Customer History & Behavior (24h/7d):**
        * **transaction_frequency_24h:** Total number of transactions by this customer in the last 24 hours.
        * **avg_transaction_amount_7d:** The customer's average transaction amount over the past 7 days.
        * **failed_transaction_count_24h:** The number of failed payment attempts for this customer in the prior 24 hours.
        * **account_age_days:** The number of days since the customer registered their account.
        
        **Risk Flags & Indicators:**
        * **previous_fraud_flag:** Indicator (1/0) if this customer has a prior history of fraud.
        * **unusual_amount_flag:** Indicator (1/0) of an abnormally high transaction amount for the user.
        * **unusual_location_flag:** Indicator (1/0) of a transaction from a new or unexpected location.
        * **multiple_transactions_short_time:** Indicator (1/0) triggered by burst activity within a very short timeframe.
        * **high_risk_device_flag:** Indicator (1/0) if the device has been previously associated with risky behavior.
        * **velocity_flag:** Indicator (1/0) triggered by rapid successive transactions overall.
        * **fraud_risk:** A categorical risk label (Low, Medium, High) assigned to the transaction.
        
        **Target & Engineered Features:**
        * **fraud_flag:** The target variable (1 = Fraudulent, 0 = Legitimate).
        * **amount_to_avg_7d_ratio:** *(Calculated)* Current transaction amount divided by the 7-day average. Highlights sudden spending surges.
        * **transaction_hour / day_of_week:** *(Calculated)* Extracted from the timestamp to identify high-risk time windows.
        """)

    # 2. OVERVIEW & SUMMARY STATISTICS
    st.subheader("1. Dataset Overview & Summary Statistics")
    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric("Total Rows (Transactions)", f"{raw_df.shape[0]:,}")
    col_met2.metric("Total Columns (Variables)", f"{raw_df.shape[1]}")
    col_met3.metric("Missing Values", f"{raw_df.isnull().sum().sum()}")
    
    st.write("---")
    
    # Dependent Variable (Isolated with Table and Pie Chart)
    st.markdown("<h4 style='color: #ffb7b2;'>🎯 Dependent Variable (fraud_flag)</h4>", unsafe_allow_html=True)
    
    # Using specific column ratios to keep the pie chart small and neatly tucked
    col_dep1, col_dep2, col_dep3 = st.columns([1.5, 1.5, 1])
    
    with col_dep1:
        dep_counts = raw_df['fraud_flag'].value_counts(normalize=True) * 100
        dep_df = pd.DataFrame({
            "Outcome": ["Legitimate (0)", "Fraudulent (1)"],
            "Percentage": [f"{dep_counts[0]:.2f}%", f"{dep_counts[1]:.2f}%"]
        })
        st.write("") 
        st.dataframe(dep_df, hide_index=True, use_container_width=True)
        
    with col_dep2:
        fig_pie, ax_pie = plt.subplots(figsize=(2.2, 2.2))
        ax_pie.pie(raw_df['fraud_flag'].value_counts(), labels=['Legitimate', 'Fraud'], autopct='%1.1f%%', colors=['#28a745', '#dc3545'], startangle=90, textprops={'fontsize': 9})
        st.pyplot(fig_pie)
        
    # Caption explaining the dependent variable section
    st.caption("Isolates the target variable being predicted. The table and chart highlight the severe class imbalance, showing the AI must learn to catch a very small percentage of fraud needles in a massive haystack of legitimate transactions.")
    
    st.write("---")

    # Continuous Variables
    st.markdown("<h4 style='color: #8ab6d6;'>📊 Continuous Variables</h4>", unsafe_allow_html=True)
    cont_cols = ['transaction_amount', 'avg_transaction_amount_7d', 'account_age_days', 'transaction_frequency_24h']
    
    selected_cont = st.selectbox("Select a continuous variable to view its distribution:", cont_cols, index=0)
    
    # Calculate statistics for the selected variable
    stats_series = raw_df[selected_cont].describe()[['min', '25%', '50%', 'mean', '75%', 'max']]
    cont_df = pd.DataFrame({
        "Statistic": ["Minimum", "25th Percentile", "Median (50%)", "Mean", "75th Percentile", "Maximum"],
        "Value": stats_series.values
    })
    
    # Format values based on whether it is an amount or a count/age
    if 'amount' in selected_cont:
        cont_df["Value"] = cont_df["Value"].apply(lambda x: f"${x:,.2f}")
    else:
        cont_df["Value"] = cont_df["Value"].apply(lambda x: f"{x:,.1f}")
        
    # Split this section internally into table and chart
    c_tab_cont, c_chart_cont = st.columns([1, 2])
    
    with c_tab_cont:
        st.write("")
        st.dataframe(cont_df, use_container_width=True, hide_index=True)
        
    with c_chart_cont:
        # Dynamic histogram with Kernel Density Estimation (KDE)
        fig_cont_hist, ax_cont_hist = plt.subplots(figsize=(6, 3))
        sns.histplot(raw_df[selected_cont], bins=30, kde=True, color='royalblue', ax=ax_cont_hist)
        ax_cont_hist.set_xlabel(selected_cont.replace('_', ' ').title())
        ax_cont_hist.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig_cont_hist)
        
    st.caption("Select a continuous variable to see its mathematical summary and distribution shape. The AI models these spreads to identify extreme outliers, such as massive spending spikes.")
    
    st.write("---")
   
    
    # Categorical Variables (Interactive Table + Risk Chart)
    st.markdown("<h4 style='color: #a8e6cf;'>🗂 Categorical Variables</h4>", unsafe_allow_html=True)
    standard_cat_cols = ['payment_method', 'device_type', 'location', 'merchant_category', 'fraud_risk']
    
    selected_cat = st.selectbox("Select a variable to view its distribution and risk:", standard_cat_cols, index=0)
    
    # Calculate volume percentage 
    pcts = raw_df[selected_cat].value_counts(normalize=True) * 100
    
    # Calculate fraud rate as a percentage
    fraud_rates = (raw_df.groupby(selected_cat)['fraud_flag'].mean() * 100).sort_values(ascending=False)
    
    # Build a combined dataframe so the table and chart are perfectly aligned
    cat_df = pd.DataFrame({
        "Category": fraud_rates.index,
        "Volume (%)": [f"{pcts[cat]:.1f}%" for cat in fraud_rates.index],
        "Fraud Rate (%)": [f"{fraud_rates[cat]:.1f}%" for cat in fraud_rates.index]
    })
    
    # Split this section internally into table and chart
    c_tab, c_chart = st.columns([1, 1])
    
    with c_tab:
        st.write("") 
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
        
    with c_chart:
        # Dynamic horizontal bar chart showing FRAUD RATE
        fig_cat_bar, ax_cat_bar = plt.subplots(figsize=(3, 2))
        sns.barplot(x=fraud_rates.values, y=fraud_rates.index, palette='magma', ax=ax_cat_bar)
        
        # FIX: Remove the weird zoom. Always start at 0, extend just past the max value.
        ax_cat_bar.set_xlim(0, fraud_rates.max() * 1.15)
        ax_cat_bar.set_xlabel('Fraud Rate (%)')
        ax_cat_bar.set_ylabel('')
        
        # Add exact percentages to the bars
        for p in ax_cat_bar.patches:
            ax_cat_bar.text(p.get_width() + 1.0, p.get_y() + p.get_height() / 2, f'{p.get_width():.1f}%', va="center", fontsize=8)
            
        plt.tight_layout()
        st.pyplot(fig_cat_bar)
        
    st.caption("The table shows both the baseline volume and the historical fraud rate for each category. The chart visualizes this Fraud Rate, accurately starting at 0% to prevent misleading perspectives.")

    st.divider()

   

    # 4. BIVARIATE ANALYSIS
    st.subheader("3. Relationship to Fraud (Bivariate Analysis)")
    col_bi1, col_bi2, col_bi3 = st.columns(3)
    
    with col_bi1:
        st.write("**Amount vs. Fraud Outcome**")
        fig_box, ax_box = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=raw_df, x='fraud_flag', y='transaction_amount', palette='pastel', ax=ax_box)
        ax_box.set_xticklabels(['Legitimate (0)', 'Fraud (1)'])
        ax_box.set_ylim(0, 500)
        ax_box.set_xlabel('')
        st.pyplot(fig_box)
        st.caption("Compares median transaction sizes. Because fraudulent charges (1) are typically higher than legitimate ones (0), the AI relies heavily on transaction amount to predict risk.")
        
    with col_bi2:
        st.write("**Fraud Risk by Category**")
        cat_fraud = raw_df.groupby('merchant_category')['fraud_flag'].mean().sort_values(ascending=False).reset_index()
        fig_cat, ax_cat = plt.subplots(figsize=(5, 4))
        sns.barplot(data=cat_fraud, x='fraud_flag', y='merchant_category', palette='magma', ax=ax_cat)
        ax_cat.set_xlim(0.46, 0.49) 
        ax_cat.set_xlabel('Fraud Probability')
        ax_cat.set_ylabel('')
        for p in ax_cat.patches:
            ax_cat.text(p.get_width() + 0.0005, p.get_y() + p.get_height() / 2, f'{p.get_width():.1%}', va="center", fontweight='bold')
        st.pyplot(fig_cat)
        st.caption("Ranks which store types have the highest fraud rates. The model slightly elevates the risk score for transactions in categories like Fashion or Luxury.")
        
    with col_bi3:
        st.write("**Failed Attempts vs. Fraud**")
        fig_fail, ax_fail = plt.subplots(figsize=(5, 4))
        sns.barplot(data=raw_df, x='failed_transaction_count_24h', y='fraud_flag', palette='Reds', ax=ax_fail)
        ax_fail.set_xlabel('Failed Transactions (24h)')
        ax_fail.set_ylabel('Fraud Probability')
        st.pyplot(fig_fail)
        st.caption("A high number of failed transactions often precedes a successful fraudulent charge, acting as an early warning trigger for the AI prediction.")

    st.divider()

    # 5. MULTIVARIATE ANALYSIS
    st.subheader("4. Fraud Risk Patterns (Multivariate Analysis)")
    col_multi1, col_multi2 = st.columns(2)
    
    with col_multi1:
        st.write("**Transaction Amount by Device & Risk Status**")
        fig_vio, ax_vio = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=raw_df, x='device_type', y='transaction_amount', hue='fraud_flag', split=True, palette='muted', ax=ax_vio)
        ax_vio.set_ylim(-100, 600)
        ax_vio.set_xlabel('Device Type')
        ax_vio.set_ylabel('Amount ($)')
        st.pyplot(fig_vio)
        st.caption("Reveals the relationship between device type, transaction size, and fraud. The AI uses this to learn complex rules, such as identifying if high-dollar mobile transactions are riskier than high-dollar desktop transactions.")
        
    with col_multi2:
        st.write("**Spending Surges by Payment Method**")
        fig_strip, ax_strip = plt.subplots(figsize=(6, 4))
        sample_df = processed_df.sample(2000, random_state=42)
        sns.stripplot(data=sample_df, x='payment_method_Credit Card', y='amount_to_avg_7d_ratio', hue='fraud_flag', dodge=True, alpha=0.6, palette='Set1', ax=ax_strip)
        ax_strip.set_xticklabels(['Other Methods', 'Credit Card'])
        ax_strip.set_xlabel('')
        ax_strip.set_ylabel('Surge Ratio (Amount / 7D Avg)')
        st.pyplot(fig_strip)
        st.caption("Illustrates the 'spending surge' feature. The model flags fraud easily when it sees customers spending vastly more than their 7-day average.")

    st.divider()

    # 6. CORRELATION MATRIX
    st.subheader("5. Correlation Matrix & Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
    corr_matrix = raw_df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax_corr, cbar_kws={"shrink": .8})
    st.pyplot(fig_corr)
    st.caption("Maps the mathematical relationships between all numerical variables. Variables with high correlation to 'fraud_flag' (like previous fraud history or international status) become the model's primary predictors.")

# ==========================================
# TAB 2: MODEL EVALUATION
# ==========================================
with tab2:
    st.subheader("Model Evaluation Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.3f}")
    c4.metric("F1-Score", f"{metrics['f1']:.3f}")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c6.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    st.info("""
    **💡 Business Interpretation:**
    * **Recall (Catching Fraud):** This tells us the percentage of actual fraud cases the model successfully caught. A higher recall means less financial loss, but pushing it too high can increase false alarms.
    * **Precision (Customer Friction):** This tells us how often the model is correct when it flags a transaction as fraud. High precision is crucial to avoid declining legitimate transactions, which frustrates good customers.
    * **PR-AUC:** Because fraud is rare (highly imbalanced data), PR-AUC is our most important holistic metric. It measures the model's ability to maximize caught fraud while minimizing false alarms.
    """)
    
    st.divider()
    col_cm, col_pr = st.columns(2)
    with col_cm:
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
        st.pyplot(fig_cm)
        st.caption("Details exactly how many test transactions the AI got right (True Positives/Negatives) versus wrong (False Alarms/Missed Fraud).")
    with col_pr:
        st.write("**Precision-Recall Tradeoff Curve**")
        precision, recall, _ = metrics['pr_curve']
        fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
        plt.xlabel('Recall (Fraud Caught)')
        plt.ylabel('Precision (Accuracy of Alarms)')
        plt.legend(loc="lower left")
        st.pyplot(fig_pr)
        st.caption("Visualizes the model's tradeoff curve. A higher line means the AI successfully catches maximum fraud without annoying legitimate customers with false alarms.")

# ==========================================
# TAB 3: GLOBAL RISK DRIVERS
# ==========================================
with tab3:
    st.subheader("What exactly pulls and predicts the fraud cases?")
    col_fi, col_shap = st.columns(2)
    with col_fi:
        st.write("**Algorithm's Top Variables (Feature Importance)**")
        importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
        fig_fi, ax_fi = plt.subplots(figsize=(6, 5))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='crest', ax=ax_fi)
        plt.xlabel("Number of AI Tree Splits (Importance)")
        plt.ylabel("")
        st.pyplot(fig_fi)
        st.caption("Ranks the exact database columns the AI relies on most when splitting its decision trees to predict fraud.")
        
    with col_shap:
        st.write("**Directional Impact (SHAP Summary)**")
        shap_vals = explainer.shap_values(X_test)
        vals_to_plot = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        fig_shap, ax_shap = plt.subplots(figsize=(6, 5))
        shap.summary_plot(vals_to_plot, X_test, max_display=10, show=False)
        st.pyplot(fig_shap)
        
        st.info("""
        **How to Read This Chart**
        * **What it does:** It reveals the hidden logic of the AI, showing how individual variables push the risk score up or down across thousands of transactions.
        * **Color (The 'What'):** 🔴 **Red** means the transaction had a high value for that feature (e.g., a massive purchase amount). 🔵 **Blue** means it had a low value.
        * **Position (The 'Why'):** Dots stretched to the **Right** are reasons the AI flagged a transaction as *Fraud*. Dots on the **Left** are reasons it considered it *Legitimate*.
        * If you see a cluster of red dots far to the right for `amount_to_avg_7d_ratio`, it means sudden spending surges are a primary reason the AI triggers alarms.
        """)
# ==========================================
# TAB 4: LIVE INSPECTOR
# ==========================================
with tab4:
    st.subheader("Live Transaction Inspector")
    row_idx = st.slider("Select a transaction record from the test set:", 0, len(X_test)-1, 42)
    
    selected_ml = X_test.iloc[[row_idx]]
    selected_raw = raw_df.iloc[idx_test[row_idx]]
    
    pred_class = model.predict(selected_ml)[0]
    pred_prob = model.predict_proba(selected_ml)[0][1]
    
    if pred_class == 1:
        st.markdown(f"<div class='risk-high'><h3>🚨 AI ALERT: {pred_prob:.1%} Fraud Probability</h3></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-low'><h3>✅ CLEARED: {pred_prob:.1%} Fraud Probability</h3></div>", unsafe_allow_html=True)
        
    col_info, col_waterfall = st.columns([1, 1.5])
    with col_info:
        st.write("#### Investigator Notes:")
        st.write(f"**Transaction ID:** {selected_raw['transaction_id']}")
        st.write(f"**Amount:** ${selected_raw['transaction_amount']:.2f}")
        st.write(f"**Failed Trx (24h):** {selected_raw['failed_transaction_count_24h']}")
        
    with col_waterfall:
        shap_val_local = explainer(selected_ml)
        
        renamed_ml = selected_ml.rename(columns=lambda x: x.replace('_', ' ').title())
        renamed_explainer = shap.TreeExplainer(model)
        shap_val_renamed = renamed_explainer(renamed_ml)
        
        fig_water, ax_water = plt.subplots(figsize=(6, 4))
        if len(shap_val_renamed.shape) == 3:
            shap.plots.waterfall(shap_val_renamed[0, :, 1], max_display=6, show=False)
        else:
            shap.plots.waterfall(shap_val_renamed[0], max_display=6, show=False)
        st.pyplot(fig_water)
        st.caption("Breaks down the exact math behind this specific prediction. Red bars show behaviors that pushed the AI toward a fraud alert, while blue bars show behaviors that made the transaction look legitimate.")