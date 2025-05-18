import os
import json
import psycopg2
import psycopg2.extras
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from decimal import Decimal
import pandas as pd # For historical data, though less critical here than for weekly aggregation
from typing import Dict, Any, Optional, List

# LangChain specific imports
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool
import db_connect;

# Tool to execute simple queries
@tool
def execute_sql_query(query: str, params: Optional[tuple] = None) -> list[tuple]:
    """
    Executes a parameterized SQL query against the PostgreSQL database
    and returns the results as a list of tuples.
    Handles potential errors. ONLY FOR SELECT.
    """
    print(f"--- Executing SQL Query ---\nQuery: {query}\nParams: {params}\n--------------------------")
    results = []
    conn = None
    try:
        conn = db_connect.connect_to_db()
        with conn.cursor() as cur:
            cur.execute(query)
            if cur.description:
                 results = cur.fetchall()
                 results = [[float(col) if isinstance(col, Decimal) else col for col in row] for row in results]
    except Exception as e:
        print(f"Database Execution Error: {e}")
        return [("QUERY_ERROR", str(e))]
    finally:
        if conn:
            conn.close()
    print(f"--- Query Results ---\n{results}\n-----------------------")
    return results


def get_historical_atm_withdrawal_data(account_number: str, end_date: date, months_lookback: int) -> Dict[str, Any]:
    """
    Fetches historical ATM withdrawal transactions and calculates their average amount.
    """
    print(f"--- Fetching Historical ATM Withdrawal Data ---")
    print(f"Account: {account_number}, End Date: {end_date}, Lookback: {months_lookback} months")
    start_date = end_date - relativedelta(months=months_lookback)

    start_date_obj = end_date - relativedelta(months=months_lookback)
    start_date_str = start_date_obj.strftime('%Y-%m-%d') # For TO_DATE(%s, 'YYYY-MM-DD')
    end_date_str = end_date.strftime('%Y-%m-%d')   

    query = """
        SELECT debit_amount
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = '"""+account_number+"""'
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= TO_DATE('"""+start_date_str+"""', 'YYYY-MM-DD') -- Convert DB text and param
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') < TO_DATE('"""+end_date_str+"""', 'YYYY-MM-DD')  -- Convert DB text and param
        AND t.description LIKE '%ATM%' -- Crucial filter
        AND t.debit_amount > 0;
    """
    avg_atm_withdrawal = 0.0
    transaction_count = 0
    error_message = None

    results = execute_sql_query.invoke({"query": query, "params": (account_number, start_date_str, end_date_str)})

    if results and isinstance(results[0][0], str) and results[0][0] == "QUERY_ERROR":
        error_message = f"DB error fetching historical ATM data: {results[0][1]}"
    elif results:
        withdrawal_amounts = [row[0] for row in results if row[0] is not None]
        transaction_count = len(withdrawal_amounts)
        if transaction_count > 0:
            avg_atm_withdrawal = sum(withdrawal_amounts) / transaction_count
    else: # No results or empty list
        error_message = "No historical ATM withdrawal transactions found."


    print(f"Historical ATM Avg: {avg_atm_withdrawal:.2f} from {transaction_count} transactions. Error: {error_message}")
    return {
        "average_amount": float(avg_atm_withdrawal),
        "transaction_count": transaction_count,
        "error": error_message
    }

def get_current_period_atm_withdrawals(account_number: str, period_start_date: date, period_end_date: date) -> List[Dict[str, Any]]:
    """
    Fetches ATM withdrawal transactions for the current period.
    """
    print(f"--- Fetching Current Period ATM Withdrawals ---")
    print(f"Account: {account_number}, Period: {period_start_date} to {period_end_date}")
    start_date_str = period_start_date.strftime('%Y-%m-%d') # Convert Python date to YYYY-MM-DD for comparison
    end_date_str = period_end_date.strftime('%Y-%m-%d')     # Convert Python date to YYYY-MM-DD for comparison
    query = """
        SELECT transaction_pk, transaction_date, debit_amount, description
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = '"""+account_number+"""'
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= TO_DATE('"""+start_date_str+"""', 'YYYY-MM-DD') -- Convert both sides
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') < TO_DATE('"""+end_date_str+"""', 'YYYY-MM-DD')  -- Convert both sides
        AND t.description LIKE '%ATM%'
        AND t.debit_amount > 0;
    """
    atm_transactions = []
    results = execute_sql_query.invoke({"query": query, "params": (account_number, start_date_str, end_date_str)})

    if results and isinstance(results[0][0], str) and results[0][0] == "QUERY_ERROR":
        print(f"Error fetching current ATM withdrawals: {results[0][1]}")
        return [{"error": results[0][1]}] # Return error indication
    elif results:
        for row in results:
            atm_transactions.append({
                "transaction_pk": row[0],
                "date": row[1],
                "amount": float(row[2]),
                "description": row[3]
            })
    print(f"Found {len(atm_transactions)} ATM withdrawals in current period.")
    return atm_transactions


# --- Analysis Function ---
def analyze_atm_withdrawals_against_average(
    current_atm_transactions: List[Dict[str, Any]],
    historical_avg_data: Dict[str, Any],
    threshold_percentage_increase: float
) -> List[Dict[str, Any]]:
    """
    Compares each current ATM withdrawal transaction amount against the historical average.
    Returns a list of transactions that exceed the threshold.
    """
    alerts = []
    historical_avg_amount = historical_avg_data.get("average_amount", 0.0)

    if historical_avg_data.get("error"):
        print(f"Skipping ATM analysis due to historical data error: {historical_avg_data['error']}")
        # Could add an alert about the data error itself if needed
        return [{"error": f"Cannot perform analysis due to: {historical_avg_data['error']}"}]

    # If no historical average, any withdrawal could be considered an alert,
    # or you might choose to skip alerting. Let's alert if avg is 0 and there's a withdrawal.
    if historical_avg_amount <= 0 and historical_avg_data.get("transaction_count", 0) == 0:
        for tx in current_atm_transactions:
            if tx.get("amount", 0) > 0:
                alerts.append({
                    "transaction_pk": tx.get("transaction_pk"),
                    "transaction_date": str(tx.get("date")),
                    "transaction_amount": tx.get("amount"),
                    "description": tx.get("description"),
                    "historical_avg_amount": historical_avg_amount,
                    "percentage_increase": float('inf'),
                    "alert_reason": "ATM withdrawal occurred with no historical average for comparison."
                })
        if alerts: return alerts

    # Define the threshold multiplier (e.g., 100% higher means 2 times the average)
    multiplier = 1.0 + (threshold_percentage_increase / 100.0)
    significant_threshold_amount = historical_avg_amount * multiplier

    for tx in current_atm_transactions:
        if tx.get("error"): # Skip if there was an error fetching current transactions
            alerts.append(tx)
            continue

        current_amount = tx.get("amount", 0.0)
        if current_amount >= significant_threshold_amount:
            percentage_increase = ((current_amount - historical_avg_amount) / historical_avg_amount) * 100.0 if historical_avg_amount > 0 else float('inf')
            alerts.append({
                "transaction_pk": tx.get("transaction_pk"),
                "transaction_date": str(tx.get("date")),
                "transaction_amount": current_amount,
                "description": tx.get("description"),
                "historical_avg_amount": historical_avg_amount,
                "percentage_increase": round(percentage_increase, 2),
                "alert_reason": f"ATM withdrawal is {threshold_percentage_increase}% or more higher than historical average."
            })
    print(f"Found {len(alerts)} ATM withdrawal alerts.")
    print(f"Found alerts: {alerts}")
    return alerts

# --- LLM and Prompts ---
llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)

atm_alert_prompt_template = """
You are a financial monitoring assistant. An ATM withdrawal has been flagged as significantly higher than the client's historical average.
Generate a concise alert message based on the following details.

Alert Details:
- Transaction Date: {transaction_date}
- Transaction Amount: {transaction_amount:.2f} RSD
- Transaction Description: {description}
- Client's Historical Average ATM Withdrawal Amount: {historical_avg_amount:.2f} RSD
- Percentage Increase Above Average: {percentage_increase:.2f}%
- Reason for Alert: {alert_reason}

Alert Message (concise, 1-2 sentences):
"""
atm_alert_prompt = ChatPromptTemplate.from_template(atm_alert_prompt_template)


# --- LangChain Expression Language (LCEL) Chain ---
atm_withdrawal_alert_chain = (
    # 1. Get historical ATM withdrawal average
    RunnablePassthrough.assign(
        historical_avg_data=lambda x: get_historical_atm_withdrawal_data(
            x["account_number"], x["current_period_end_date"], x["historical_lookback_months"]
        )
    )
    # 2. Get current period ATM withdr.
    | RunnablePassthrough.assign(
        current_atm_transactions=lambda x: get_current_period_atm_withdrawals(
            x["account_number"], x["current_period_start_date"], x["current_period_end_date"]
        )
    )
    # 3. Analyze current withdrawals against historical average
    | RunnablePassthrough.assign(
        triggered_alerts=lambda x: analyze_atm_withdrawals_against_average(
            x["current_atm_transactions"], x["historical_avg_data"], x["alert_threshold_percentage"]
        )
    )
    # 4. For each triggered alert, format and generate an LLM notification
    | RunnableLambda(
        lambda x: [
            (atm_alert_prompt | llm | StrOutputParser()).invoke(alert_detail)
            for alert_detail in x.get("triggered_alerts", [])
            if not alert_detail.get("error")
        ]
        
        if x.get("triggered_alerts") and \
           any(not alert.get("error") for alert in x.get("triggered_alerts", []))
        else ["No significant ATM withdrawals detected or error in data processing."]
    )
)

# --- Main Execution ---
if __name__ == "__main__":
    account = "325930050010370593"

    current_period_start = date(2024, 11, 1)
    current_period_end = date(2024, 11, 30)

    alert_threshold_percentage = 100.0 # 100% higher
    historical_lookback_months = 6

    print(f"--- Running High ATM Withdrawal Analysis ---")
    print(f"Account: {account}")
    print(f"Analyzing Period: {current_period_start} to {current_period_end}")
    print(f"Alert Threshold: >= {alert_threshold_percentage}% increase vs {historical_lookback_months}-month avg ATM transaction.")

    chain_input = {
        "account_number": account,
        "current_period_start_date": current_period_start,
        "current_period_end_date": current_period_end,
        "alert_threshold_percentage": alert_threshold_percentage,
        "historical_lookback_months": historical_lookback_months
    }

    # Invoke the full chain
    print("\n--- Invoking Chain ---")
    print(f"Input: {chain_input}")
    list_of_alerts = atm_withdrawal_alert_chain.invoke(chain_input)

    print("\n--- Generated Alerts ---")
    if isinstance(list_of_alerts, list) and list_of_alerts:
        for i, alert_message in enumerate(list_of_alerts):
            print(f"Alert {i+1}: {alert_message}")
    elif isinstance(list_of_alerts, str):
        print(list_of_alerts)
    else:
        print("No alerts generated or an unexpected result.")