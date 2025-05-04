import psycopg2
import psycopg2.extras
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
from decimal import Decimal
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import re
import calendar

# LangChain specific imports
from langchain.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool

import db_connect;

# --- Configuration ---
TARGET_BANK_PREFIX = "205" # Komercijalna Banka
ACCOUNT_REGEX = re.compile(r'\b(' + re.escape(TARGET_BANK_PREFIX) + r'\d{10,})\b')

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

def extract_target_bank_account(description: str) -> Optional[str]:
    """Extracts account number starting with TARGET_BANK_PREFIX using regex."""
    if not description:
        return None
    match = ACCOUNT_REGEX.search(description)
    return match.group(1) if match else None

def get_debit_transactions_for_period(account_number: str, start_date: date, end_date: date) -> List[Dict]:
    """Fetches DEBIT transactions for a given account and date range."""
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
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') <= %s
        AND t.debit_amount > 0;
        -- Optional: AND lower(t.category) = 'transfer out' -- If category helps filter
    """
    results = execute_sql_query.invoke({"query": query, "params": (account_number, start_date, end_date)})
    if results and isinstance(results[0], dict) and "error" in results[0]:
        print(f"Error fetching debit transactions: {results[0]['error']}")
        return []
    return results

def analyze_external_bank_transfers(
    transactions: List[Dict],
    target_prefix: str = TARGET_BANK_PREFIX
) -> Tuple[Decimal, List[Dict]]:
    """
    Filters transactions for transfers to the target bank prefix and calculates the total amount.
    Returns the total sum (Decimal) and a list of the specific transactions.
    """
    total_amount = Decimal("0.0")
    target_transactions = []
    if not transactions:
        return total_amount, target_transactions

    print(f"Analyzing {len(transactions)} transactions for transfers to prefix '{target_prefix}'...")

    for tx in transactions:
        try:
            description = tx.get('description')
            recipient_account = extract_target_bank_account(description)

            if recipient_account:
                amount = Decimal(str(tx.get('debit_amount', '0.0')))
                if amount > 0:
                    total_amount += amount
                    tx_detail = {
                        "pk": tx.get('transaction_pk'),
                        "date": tx.get('transaction_date'),
                        "amount": amount,
                        "description": description[:150]
                    }
                    target_transactions.append(tx_detail)

        except Exception as e:
            print(f"Error processing transaction {tx.get('transaction_pk')} for external transfer: {e}")
            continue

    print(f"Total amount transferred to prefix '{target_prefix}': {total_amount:.2f}")
    return total_amount, target_transactions


def analyze_historical_regularity(
    account_number: str,
    analysis_start_date: date,
    months_lookback: int,
    target_prefix: str = TARGET_BANK_PREFIX,
    rsd_threshold: Decimal = Decimal("0.0")
) -> Dict[str, Any]:
    """Analyzes historical transfers to the target bank for regularity."""
    historical_summary = {
        "months_analyzed": months_lookback,
        "months_with_transfer": 0,
        "months_above_threshold": 0,
        "average_monthly_transfer": 0.0,
        "is_regular_above_threshold": False
    }
    end_date_hist = analysis_start_date 
    start_date_hist = (end_date_hist - relativedelta(months=months_lookback)).replace(day=1)

    print(f"--- Analyzing Historical Regularity ({start_date_hist} to {end_date_hist}) ---")

    hist_transactions = get_debit_transactions_for_period(account_number, start_date_hist, end_date_hist - timedelta(days=1))
    if not hist_transactions:
        return historical_summary

    try:
        hist_df = pd.DataFrame(hist_transactions)
        if hist_df.empty: return historical_summary

        hist_df['target_account'] = hist_df['description'].apply(extract_target_bank_account)
        target_hist_df = hist_df.dropna(subset=['target_account']).copy()

        if target_hist_df.empty:
            print("No historical transfers to target bank found.")
            return historical_summary

        target_hist_df['transaction_date'] = pd.to_datetime(target_hist_df['transaction_date'])
        target_hist_df['debit_amount'] = pd.to_numeric(target_hist_df['debit_amount'], errors='coerce').fillna(0)
        target_hist_df['month_year'] = target_hist_df['transaction_date'].dt.to_period('M')

        monthly_sums = target_hist_df.groupby('month_year')['debit_amount'].sum()

        if monthly_sums.empty: return historical_summary

        historical_summary["months_with_transfer"] = len(monthly_sums)
        historical_summary["average_monthly_transfer"] = float(monthly_sums.mean())
        if rsd_threshold > 0:
            historical_summary["months_above_threshold"] = int((monthly_sums > float(rsd_threshold)).sum())
            historical_summary["is_regular_above_threshold"] = historical_summary["months_above_threshold"] >= (months_lookback // 2)

    except Exception as e:
         print(f"Error during historical analysis: {e}")

    print(f"Historical Summary: {historical_summary}")
    return historical_summary


llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)

external_transfer_alert_prompt_template = """
You are a financial monitoring assistant. An alert for potentially significant monthly payments to an external bank (Account prefix: {target_bank_prefix} - {target_bank_name}) was triggered for account {account_number} for the month of {analysis_month_year}. This pattern might indicate an external loan payment. Summarize the findings.

Alert Threshold: Total monthly transfer to {target_bank_prefix} accounts > {rsd_threshold:.0f} RSD (~{eur_threshold:.0f} EUR)
Account: {account_number}
Month Checked: {analysis_month_year}

Current Month ({analysis_month_year}):
- Total Transferred to {target_bank_prefix} accounts: {current_month_total:.2f} RSD
- Number of Transfers: {current_month_tx_count}

Historical Pattern (Last {historical_lookback_months} months):
- Average Monthly Transfer to {target_bank_prefix}: {historical_avg_monthly:.2f} RSD
- Months with Transfers > {rsd_threshold:.0f} RSD: {historical_months_above_threshold} out of {historical_months_analyzed}
- Considered Regular Above Threshold: {historical_regularity}

Generate a brief alert message (2-3 sentences) highlighting the significant transfer this month and mentioning the historical pattern as context for the potential external loan suspicion.
"""
external_transfer_alert_prompt = ChatPromptTemplate.from_template(external_transfer_alert_prompt_template)

def conditionally_summarize_external_transfer(input_data: Dict[str, Any]) -> str:
    """Invokes LLM to summarize if current month transfer exceeds threshold."""
    current_total = input_data["current_month_total"]
    rsd_threshold = input_data["rsd_threshold"]
    trigger_met = current_total > rsd_threshold

    if trigger_met:
        print(f"--- Threshold Exceeded ({current_total:.2f} > {rsd_threshold:.2f}): Invoking LLM ---")
        hist_summary = input_data["historical_summary"]
        prompt_input = {
            "target_bank_prefix": input_data["target_bank_prefix"],
            "target_bank_name": "Komercijalna Banka",
            "account_number": input_data["account_number"],
            "analysis_month_year": input_data["analysis_month_year_str"],
            "rsd_threshold": float(input_data["rsd_threshold"]),
            "eur_threshold": input_data["eur_threshold"],
            "current_month_total": float(input_data["current_month_total"]),
            "current_month_tx_count": len(input_data["current_month_target_transactions"]),
            "historical_lookback_months": hist_summary["months_analyzed"],
            "historical_avg_monthly": hist_summary["average_monthly_transfer"],
            "historical_months_above_threshold": hist_summary["months_above_threshold"],
            "historical_months_analyzed": hist_summary["months_analyzed"],
            "historical_regularity": hist_summary["is_regular_above_threshold"],
        }
        summarization_chain = external_transfer_alert_prompt | llm | StrOutputParser()
        return summarization_chain.invoke(prompt_input)
    else:
        print(f"--- Threshold Not Exceeded ({current_total:.2f} <= {rsd_threshold:.2f}) ---")
        return f"Total transfers to accounts starting with {input_data['target_bank_prefix']} this month ({current_total:.2f} RSD) did not exceed the threshold (~{input_data['eur_threshold']:.0f} EUR)."


external_loan_chain = (
    RunnablePassthrough.assign(
        current_month_transactions=lambda x: get_debit_transactions_for_period(
            x["account_number"], x["analysis_start_date"], x["analysis_end_date"]
        )
    )
    | RunnablePassthrough.assign(
        current_analysis=lambda x: analyze_external_bank_transfers(
            x["current_month_transactions"], x["target_bank_prefix"]
        )
    )
    | RunnablePassthrough.assign(
         current_month_total=lambda x: x["current_analysis"][0],
         current_month_target_transactions=lambda x: x["current_analysis"][1]
    )
    | RunnablePassthrough.assign(
        historical_summary=lambda x: analyze_historical_regularity(
            x["account_number"],
            x["analysis_start_date"],
            x["historical_lookback_months"],
            x["target_bank_prefix"],
            x["rsd_threshold"]
        )
    )
    | RunnableLambda(conditionally_summarize_external_transfer)
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

    eur_threshold_value = 200.0
    approx_eur_to_rsd_rate = Decimal("117.3")
    rsd_threshold_value = Decimal(str(eur_threshold_value)) * approx_eur_to_rsd_rate

    historical_lookback_months = 6
    target_bank_prefix_const = TARGET_BANK_PREFIX

    print(f"--- Running External Bank Transfer Check ---")
    print(f"Account: {account}")
    print(f"Analyzing Month: {analysis_year}-{analysis_month:02d}")
    print(f"Target Bank Prefix: {target_bank_prefix_const}")
    print(f"Threshold: > {eur_threshold_value} EUR (~{rsd_threshold_value:.2f} RSD) total per month")
    print(f"Historical Lookback: {historical_lookback_months} months")

    chain_input = {
        "account_number": account,
        "analysis_year": analysis_year,
        "analysis_month": analysis_month,
        "analysis_start_date": analysis_start_date,
        "analysis_end_date": analysis_end_date,
        "analysis_month_year_str": f"{analysis_year}-{analysis_month:02d}",
        "eur_threshold": eur_threshold_value,
        "rsd_threshold": rsd_threshold_value,
        "historical_lookback_months": historical_lookback_months,
        "target_bank_prefix": target_bank_prefix_const,
    }

    final_response = external_loan_chain.invoke(chain_input)

    print("\n--- Analysis Result ---")
    print(final_response)