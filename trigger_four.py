import psycopg2
import psycopg2.extras
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
import pandas as pd
from typing import Dict, Any, Optional
import calendar

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool

import db_connect;

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


def get_historical_transactions_df(account_number: str, end_date: date, months_lookback: int) -> pd.DataFrame:
    """ Fetches historical transactions into a DataFrame. """
    print(f"--- Fetching Historical Transactions for ATM Avg ---")
    start_date = (end_date.replace(day=1) - relativedelta(months=months_lookback)).replace(day=1)
    print(f"Account: {account_number}, Hist. Period: {start_date} to {end_date}")

    query = """
        SELECT
            t.transaction_date,
            t.debit_amount,
            t.category
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') < %s -- Use < end_date (which is start of current month)
        AND t.debit_amount > 0;
    """
    df = pd.DataFrame()
    conn = None
    try:
        conn = db_connect.connect_to_db()
        df = pd.read_sql(query, conn, params=(account_number, start_date, end_date), parse_dates=['transaction_date'])
        df['debit_amount'] = pd.to_numeric(df['debit_amount'], errors='coerce').fillna(0)
        df['category'] = df['category'].fillna('').astype(str)
    except Exception as e:
        print(f"Error fetching historical data for ATM avg: {e}")
        return pd.DataFrame(columns=['transaction_date', 'debit_amount', 'category'])
    finally:
        if conn:
            conn.close()
    print(f"Fetched {len(df)} historical transactions for ATM avg.")
    return df

def calculate_historical_avg_monthly_atm(historical_df: pd.DataFrame) -> float:
    """Calculates the average monthly ATM withdrawal amount from historical data."""
    if historical_df.empty:
        print("Warning: No historical data for ATM avg calculation.")
        return 0.0

    atm_df = historical_df[historical_df['category'].str.lower() == 'atm withdrawal'].copy()

    if atm_df.empty:
        print("Warning: No historical 'ATM Withdrawal' transactions found.")
        return 0.0

    atm_df['transaction_date'] = pd.to_datetime(atm_df['transaction_date'])
    atm_df_indexed = atm_df.set_index('transaction_date')

    monthly_atm_sums = atm_df_indexed['debit_amount'].resample('MS').sum()

    if monthly_atm_sums.empty:
        print("Warning: No monthly ATM sums found in historical data.")
        return 0.0

    average = monthly_atm_sums.mean()
    print(f"Calculated historical average monthly ATM withdrawal: {average:.2f}")
    return float(average)


def get_analysis_month_atm_total(account_number: str, year: int, month: int) -> float:
    """Gets the total ATM spending for the specified analysis month."""
    start_date = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day)

    print(f"--- Getting Analysis Month ATM Total ---")
    print(f"Account: {account_number}, Month: {year}-{month:02d} ({start_date} to {end_date})")

    query = """
        SELECT SUM(t.debit_amount) as total_atm
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') <= %s
        AND lower(t.category) = 'atm withdrawal'; -- Case-insensitive category match
    """
    results = execute_sql_query.invoke({"query": query, "params": (account_number, start_date, end_date)})

    if not results or (isinstance(results[0], dict) and "error" in results[0]):
        print(f"Error retrieving ATM total for {year}-{month:02d}.")
        return 0.0

    current_total = results[0].get('total_atm', 0.0)
    current_total = float(current_total) if current_total is not None else 0.0

    print(f"Analysis month ({year}-{month:02d}) ATM total: {current_total:.2f}")
    return current_total


def analyze_atm_increase(current_total: float, historical_avg: float, threshold_multiplier: float = 2.0) -> Dict[str, Any]:
    """Compares current month's ATM total to historical average."""
    analysis = {
        "current_month_atm_total": current_total,
        "historical_avg_monthly_atm": historical_avg,
        "threshold_multiplier": threshold_multiplier,
        "increase_detected": False,
        "analysis_error": None
    }
    print(f"Comparing: Current={current_total:.2f}, HistAvg={historical_avg:.2f}, Multiplier={threshold_multiplier}")

    if historical_avg < 0:
         analysis["analysis_error"] = "Historical average is negative."
         return analysis

    if historical_avg == 0:
        if current_total > 0:
            analysis["increase_detected"] = True
            analysis["analysis_error"] = "Historical average ATM withdrawal is zero. Any withdrawal triggers alert."
        else:
             analysis["analysis_error"] = "Historical average and current ATM withdrawal are both zero." # Optional: Or just return increase_detected=False
        return analysis

    try:
        analysis["increase_detected"] = current_total >= (historical_avg * threshold_multiplier)
    except Exception as e:
        analysis["analysis_error"] = f"Error during comparison: {e}"

    return analysis

llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)
atm_alert_prompt_template = """
You are a financial monitoring assistant. An alert for high ATM withdrawal was triggered for account {account_number} for the month of {analysis_month_year}. Summarize the details concisely.

Analysis Details:
- Total ATM Withdrawal this month ({analysis_month_year}): {current_month_atm_total:.2f} RSD
- Historical Average Monthly ATM Withdrawal (last {historical_lookback_months} months): {historical_avg_monthly_atm:.2f} RSD
- Threshold: Amount is {threshold_multiplier:.0f}x or more than the historical average.
- Alert Triggered: {increase_detected}
- Analysis Error: {analysis_error}

Generate a brief alert message (1-2 sentences) notifying the user about the unusually high ATM withdrawals this month compared to their average.
"""
atm_alert_prompt = ChatPromptTemplate.from_template(atm_alert_prompt_template)


def conditionally_summarize_atm_alert(input_data: Dict[str, Any]) -> str:
    """Invokes LLM to summarize if high ATM withdrawal was detected."""
    should_summarize = input_data["analysis_results"].get("increase_detected", False) and not input_data["analysis_results"].get("analysis_error")

    if should_summarize:
        print("--- High ATM Withdrawal Detected: Invoking LLM for Summarization ---")
        prompt_input = {
            "account_number": input_data["account_number"],
            "analysis_month_year": input_data["analysis_month_year_str"],
            "current_month_atm_total": input_data["analysis_results"]["current_month_atm_total"],
            "historical_lookback_months": input_data["historical_lookback_months"],
            "historical_avg_monthly_atm": input_data["analysis_results"]["historical_avg_monthly_atm"],
            "threshold_multiplier": input_data["analysis_results"]["threshold_multiplier"],
            "increase_detected": input_data["analysis_results"]["increase_detected"],
            "analysis_error": input_data["analysis_results"]["analysis_error"],
        }
        summarization_chain = atm_alert_prompt | llm | StrOutputParser()
        return summarization_chain.invoke(prompt_input)
    else:
        print("--- No Significant ATM Withdrawal Increase Detected ---")
        if input_data["analysis_results"].get("analysis_error"):
             return f"ATM withdrawal analysis could not be completed due to error: {input_data['analysis_results'].get('analysis_error')}"
        else:
             return "ATM withdrawals for the month are within the expected range compared to historical average."

atm_analysis_chain = (
    RunnablePassthrough.assign(
        historical_transactions=lambda x: get_historical_transactions_df(
            x["account_number"], x["analysis_start_date"], x["historical_lookback_months"]
        )
    )
    | RunnablePassthrough.assign(
        historical_avg_monthly_atm=lambda x: calculate_historical_avg_monthly_atm(
            x["historical_transactions"]
        )
    )
    | RunnablePassthrough.assign(
        current_month_atm_total=lambda x: get_analysis_month_atm_total(
            x["account_number"], x["analysis_year"], x["analysis_month"]
        )
    )
    | RunnablePassthrough.assign(
        analysis_results=lambda x: analyze_atm_increase(
            x["current_month_atm_total"],
            x["historical_avg_monthly_atm"],
            x["threshold_multiplier"] # Pass multiplier from input
        )
    )
    | RunnableLambda(conditionally_summarize_atm_alert)
)


if __name__ == "__main__":
    account = "325930050010370593"

    # today = date.today()
    today = date.fromisoformat('2024-09-01')
    first_day_current_month = today.replace(day=1)
    last_day_previous_month = first_day_current_month - timedelta(days=1)
    analysis_month = last_day_previous_month.month
    analysis_year = last_day_previous_month.year

    # analysis_start_date = date(analysis_year, analysis_month, 1)
    analysis_start_date = date.fromisoformat('2024-08-02')

    atm_threshold_multiplier = 2.0
    historical_lookback_months = 6

    print(f"--- Running High ATM Withdrawal Check ---")
    print(f"Account: {account}")
    print(f"Analyzing Month: {analysis_year}-{analysis_month:02d}")
    print(f"Threshold: >= {atm_threshold_multiplier}x historical monthly average (last {historical_lookback_months} months)")

    chain_input = {
        "account_number": account,
        "analysis_year": analysis_year,
        "analysis_month": analysis_month,
        "analysis_start_date": analysis_start_date,
        "analysis_month_year_str": f"{analysis_year}-{analysis_month:02d}",
        "threshold_multiplier": atm_threshold_multiplier,
        "historical_lookback_months": historical_lookback_months
    }

    final_response = atm_analysis_chain.invoke(chain_input)

    print("\n--- Analysis Result ---")
    print(final_response)