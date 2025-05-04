import psycopg2
import psycopg2.extras
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import calendar

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool

import db_connect;

SAVINGS_CATEGORY = 'konverzija eur' # ADAPT THIS

MIN_MONTHS_SAVED_FOR_REGULAR = 3
HISTORICAL_LOOKBACK_MONTHS_SAVINGS = 3

@tool
def execute_sql_query(query: str, params: Optional[tuple] = None) -> list[dict]:
    """Executes parameterized SELECT query, returns list of dicts or error dict."""
    print(f"--- Executing SQL Query ---\nQuery: {query}\nParams: {params}\n--------------------------")
    results = []
    conn = None
    try:
        conn = db_connect.connect_to_db()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            if cur.description:
                 results = cur.fetchall()
    except Exception as e:
        print(f"Database Execution Error: {e}")
        return [{"error": f"Database Execution Error: {e}"}]
    finally:
        if conn:
            conn.close()
    print(f"--- Query Results ({len(results)} rows) ---\n{results[:5]}...\n-----------------------")
    return [dict(row) for row in results]

def is_savings_transaction(tx: Dict[str, Any]) -> bool:
    """Checks if a transaction is likely a savings/EUR purchase."""
    category = str(tx.get('category', '')).lower().strip()

    if category == SAVINGS_CATEGORY:
        return True

    return False

def get_debit_transactions_df_for_period(account_number: str, start_date: date, end_date: date) -> pd.DataFrame:
    """ Fetches DEBIT transactions into a DataFrame for a given period. """
    print(f"--- Fetching Debit Transactions ---")
    print(f"Account: {account_number}, Period: {start_date} to {end_date}")
    query = """
        SELECT
            t.transaction_pk,
            t.transaction_date,
            t.description,
            t.debit_amount,
            t.category
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') <= %s -- Use <= end_date
        AND t.debit_amount > 0;
    """
    df = pd.DataFrame()
    conn = None
    try:
        conn = db_connect.connect_to_db()
        df = pd.read_sql(query, conn, params=(account_number, start_date, end_date), parse_dates=['transaction_date'])
        df['category'] = df['category'].fillna('').astype(str)
        df['description'] = df['description'].fillna('').astype(str)
        df['debit_amount'] = pd.to_numeric(df['debit_amount'], errors='coerce').fillna(0)
    except Exception as e:
        print(f"Error fetching debit transactions: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
    print(f"Fetched {len(df)} debit transactions.")
    return df

def check_historical_savings_regularity(
    historical_df: pd.DataFrame,
    min_months_threshold: int = MIN_MONTHS_SAVED_FOR_REGULAR
) -> Tuple[bool, int]:
    """
    Checks if savings transactions occurred regularly in the historical data.
    Returns (is_regular, months_with_savings_count).
    """
    if historical_df.empty:
        print("No historical data to check savings regularity.")
        return False, 0

    historical_df['is_savings'] = historical_df.apply(is_savings_transaction, axis=1)
    savings_hist = historical_df[historical_df['is_savings']].copy()

    if savings_hist.empty:
        print("No historical savings transactions found.")
        return False, 0

    savings_hist['month_year'] = savings_hist['transaction_date'].dt.to_period('M')
    months_with_savings = savings_hist['month_year'].nunique()

    is_regular = months_with_savings >= min_months_threshold
    print(f"Historical Savings: Found in {months_with_savings} month(s). Regularity threshold ({min_months_threshold}): {'Met' if is_regular else 'Not Met'}")
    return is_regular, months_with_savings

def check_savings_in_analysis_month(analysis_month_df: pd.DataFrame) -> bool:
    """Checks if any savings transaction exists in the analysis month's data."""
    if analysis_month_df.empty:
        return False

    analysis_month_df['is_savings'] = analysis_month_df.apply(is_savings_transaction, axis=1)
    found_savings_this_month = analysis_month_df['is_savings'].any()

    print(f"Savings This Month: {'Found' if found_savings_this_month else 'Not Found'}")
    return found_savings_this_month

llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)

missed_savings_alert_prompt_template = """
You are a financial monitoring assistant. An alert for a potential change in savings behavior was triggered for account {account_number}. The client, who regularly saved (bought EUR) in the previous {historical_lookback_months} months, did not perform a similar transaction in {analysis_month_year}. This might indicate emerging financial difficulty.

Account: {account_number}
Month Checked: {analysis_month_year}
Historical Check Period: Previous {historical_lookback_months} months
Historical Savings Found In: {historical_months_with_savings} out of {historical_lookback_months} months (Regularity threshold: {min_historical_payments} months)
Savings Transaction Found This Month ({analysis_month_year}): {savings_this_month}

Generate a brief alert message (1-2 sentences) notifying about the absence of the usual savings activity this month compared to the recent pattern.
"""
missed_savings_alert_prompt = ChatPromptTemplate.from_template(missed_savings_alert_prompt_template)

def conditionally_summarize_missed_savings(input_data: Dict[str, Any]) -> str:
    """Invokes LLM to summarize if regular savings were missed."""
    was_regular = input_data["was_regular_saver"]
    saved_this_month = input_data["saved_in_analysis_month"]
    trigger_met = was_regular and not saved_this_month

    if trigger_met:
        print("--- Absence of Regular Savings Detected: Invoking LLM ---")
        prompt_input = {
            "account_number": input_data["account_number"],
            "analysis_month_year": input_data["analysis_month_year_str"],
            "historical_lookback_months": input_data["historical_lookback_months"],
            "historical_months_with_savings": input_data["historical_months_count"],
            "min_historical_payments": input_data["min_historical_payments"],
            "savings_this_month": "No"
        }
        summarization_chain = missed_savings_alert_prompt | llm | StrOutputParser()
        return summarization_chain.invoke(prompt_input)
    else:
        print("--- No Absence of Regular Savings Detected ---")
        if not was_regular:
            return f"Savings were not detected regularly (in >= {input_data['min_historical_payments']} of the last {input_data['historical_lookback_months']} months)."
        elif saved_this_month:
             return "Savings transaction detected this month, consistent with or resuming pattern."
        else:
             return "No alert triggered for savings absence." # Fallback


missed_savings_chain = (
    RunnablePassthrough.assign(
        historical_transactions_df=lambda x: get_debit_transactions_df_for_period(
            x["account_number"], x["historical_start_date"], x["historical_end_date"]
        )
    )
    | RunnablePassthrough.assign(
        historical_analysis=lambda x: check_historical_savings_regularity(
            x["historical_transactions_df"], x["min_historical_payments"]
        )
    )
    | RunnablePassthrough.assign(
        was_regular_saver=lambda x: x["historical_analysis"][0],
        historical_months_count=lambda x: x["historical_analysis"][1]
    )
    | RunnablePassthrough.assign(
        analysis_month_transactions_df=lambda x: get_debit_transactions_df_for_period(
            x["account_number"], x["analysis_start_date"], x["analysis_end_date"]
        )
    )
    | RunnablePassthrough.assign(
        saved_in_analysis_month=lambda x: check_savings_in_analysis_month(
            x["analysis_month_transactions_df"]
        )
    )
    | RunnableLambda(conditionally_summarize_missed_savings)
)

if __name__ == "__main__":
    account = "325930050010370593"

    # today = date.today()

    today = date.fromisoformat('2024-09-01')

    first_day_current_month = today.replace(day=1)
    last_day_previous_month = first_day_current_month - timedelta(days=1)
    analysis_month = last_day_previous_month.month
    analysis_year = last_day_previous_month.year

    analysis_start_date = date(analysis_year, analysis_month, 1)
    analysis_end_day = calendar.monthrange(analysis_year, analysis_month)[1]
    analysis_end_date = date(analysis_year, analysis_month, analysis_end_day)

    historical_start_date = (analysis_start_date - relativedelta(months=HISTORICAL_LOOKBACK_MONTHS_SAVINGS)).replace(day=1)
    historical_end_date = analysis_start_date - timedelta(days=1)

    historical_lookback_cfg = HISTORICAL_LOOKBACK_MONTHS_SAVINGS
    min_historical_payments_cfg = MIN_MONTHS_SAVED_FOR_REGULAR

    print(f"--- Running Absence of Savings Check ---")
    print(f"Account: {account}")
    print(f"Analyzing Month: {analysis_year}-{analysis_month:02d}")
    print(f"Historical Period Checked: {historical_start_date} to {historical_end_date} ({historical_lookback_cfg} months)")
    print(f"Regularity Threshold: Saved in >= {min_historical_payments_cfg} of the historical months")
    print(f"Target Savings Category: '{SAVINGS_CATEGORY}'")

    chain_input = {
        "account_number": account,
        "analysis_year": analysis_year,
        "analysis_month": analysis_month,
        "analysis_start_date": analysis_start_date,
        "analysis_end_date": analysis_end_date,
        "historical_start_date": historical_start_date,
        "historical_end_date": historical_end_date,
        "analysis_month_year_str": f"{analysis_year}-{analysis_month:02d}",
        "historical_lookback_months": historical_lookback_cfg,
        "min_historical_payments": min_historical_payments_cfg
    }

    final_response = missed_savings_chain.invoke(chain_input)

    print("\n--- Analysis Result ---")
    print(final_response)