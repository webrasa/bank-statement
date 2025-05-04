from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
from decimal import Decimal
import pandas as pd
from typing import Dict, Any, Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool

import db_connect;

@tool
def execute_sql_query(query: str, params: Optional[tuple] = None) -> list[tuple]:
    """
    Executes a parameterized SQL query against the PostgreSQL database
    and returns the results. Handles potential errors. ONLY FOR SELECT.
    """
    print(f"--- Executing SQL Query ---\nQuery: {query}\nParams: {params}\n--------------------------")
    results = []
    conn = None
    try:
        conn = db_connect.connect_to_db()
        with conn.cursor() as cur:
            cur.execute(query, params)
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

def get_historical_transactions(account_number: str, end_date: date, months_lookback: int) -> pd.DataFrame:
    """
    Fetches transactions for a given account over a specified lookback period.
    Returns a pandas DataFrame.
    """
    print(f"--- Fetching Historical Transactions ---")
    print(f"Account: {account_number}, End Date: {end_date}, Lookback: {months_lookback} months")
    start_date = end_date - relativedelta(months=months_lookback)

    query = """
        SELECT
            t.transaction_date,
            t.debit_amount
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') < %s -- Use < end_date for lookback period
        AND t.debit_amount > 0;
    """
    df = pd.DataFrame()
    conn = None
    try:
        conn = db_connect.connect_to_db()
        df = pd.read_sql(query, conn, params=(account_number, start_date, end_date), parse_dates=['transaction_date'])
        df['debit_amount'] = pd.to_numeric(df['debit_amount'], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame(columns=['transaction_date', 'debit_amount'])
    finally:
        if conn:
            conn.close()

    print(f"Fetched {len(df)} historical transactions.")
    return df

def calculate_historical_weekly_avg(historical_df: pd.DataFrame) -> float:
    """Calculates the average weekly spending from historical transaction data."""
    if historical_df.empty:
        print("Warning: No historical data to calculate average.")
        return 0.0

    historical_df['transaction_date'] = pd.to_datetime(historical_df['transaction_date'])

    df_indexed = historical_df.set_index('transaction_date')

    weekly_sums = df_indexed['debit_amount'].resample('W-MON').sum()

    if weekly_sums.empty:
        print("Warning: No weekly spending sums found in historical data.")
        return 0.0

    average = weekly_sums.mean()
    print(f"Calculated historical average weekly spend: {average:.2f}")
    return float(average)

def get_current_week_spending(account_number: str, week_start: date, week_end: date) -> float:
    """Gets the total spending for the specified week."""
    print(f"--- Getting Current Week Spending ---")
    print(f"Account: {account_number}, Week: {week_start} to {week_end}")
    query = """
        SELECT SUM(t.debit_amount)
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') <= %s;
    """
    results = execute_sql_query.invoke({"query": query, "params": (account_number, week_start, week_end)})

    if not results or isinstance(results[0][0], str) and results[0][0] == "QUERY_ERROR":
        print("Error retrieving current week spending.")
        return 0.0

    current_spending = results[0][0] if results and results[0] else 0.0
    current_spending = float(current_spending) if current_spending is not None else 0.0

    print(f"Current week ({week_start} to {week_end}) spending: {current_spending:.2f}")
    return current_spending


def analyze_spending_jump(current_spend: float, historical_avg: float, threshold_percentage: float) -> Dict[str, Any]:
    """Compares current spending to historical average and checks threshold."""
    analysis = {
        "current_week_spend": current_spend,
        "historical_weekly_avg": historical_avg,
        "threshold_percentage": threshold_percentage,
        "jump_detected": False,
        "percentage_increase": 0.0,
        "analysis_error": None
    }

    if historical_avg <= 0:
        if current_spend > 0:
            analysis["jump_detected"] = True
            analysis["percentage_increase"] = float('inf')
            analysis["analysis_error"] = "Historical average is zero or negative; any current spending triggers alert."
        else:
             analysis["analysis_error"] = "Historical average is zero or negative, and no current spending."
        return analysis

    try:
        increase = ((current_spend - historical_avg) / historical_avg) * 100.0
        analysis["percentage_increase"] = round(increase, 2)
        analysis["jump_detected"] = increase >= threshold_percentage
    except Exception as e:
        analysis["analysis_error"] = f"Error during percentage calculation: {e}"

    return analysis

llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)

jump_summary_prompt_template = """
You are a financial monitoring assistant. Based on the spending analysis for the week {week_start_date} to {week_end_date}, provide a concise alert ONLY IF a significant jump was detected.

Analysis Details:
- Current Week Spending: {current_week_spend:.2f} RSD
- Historical Average Weekly Spending (last {historical_lookback_months} months): {historical_weekly_avg:.2f} RSD
- Percentage Increase: {percentage_increase:.2f}%
- Alert Threshold: {threshold_percentage:.1f}%
- Jump Detected: {jump_detected}
- Analysis Error: {analysis_error}

Alert Message (Only if Jump Detected is True and no error):
"""
jump_summary_prompt = ChatPromptTemplate.from_template(jump_summary_prompt_template)

def conditionally_invoke_llm_and_parse(input_data: Dict[str, Any]) -> str:
    """
    Invokes the LLM summarization prompt only if jump_detected is True and no error.
    Otherwise, returns a standard message.
    """
    should_summarize = input_data.get("jump_detected", False) and not input_data.get("analysis_error")

    if should_summarize:
        print("--- Condition Met: Invoking LLM for Summarization ---")
        prompt_input = {
            "week_start_date": input_data["week_start_date"].strftime('%Y-%m-%d'),
            "week_end_date": input_data["week_end_date"].strftime('%Y-%m-%d'),
            "current_week_spend": input_data["analysis_results"]["current_week_spend"],
            "historical_lookback_months": input_data["historical_lookback_months"],
            "historical_weekly_avg": input_data["analysis_results"]["historical_weekly_avg"],
            "percentage_increase": input_data["analysis_results"]["percentage_increase"],
            "threshold_percentage": input_data["analysis_results"]["threshold_percentage"],
            "jump_detected": input_data["analysis_results"]["jump_detected"],
            "analysis_error": input_data["analysis_results"]["analysis_error"],
        }
        summarization_chain = jump_summary_prompt | llm | StrOutputParser()
        return summarization_chain.invoke(prompt_input)
    else:
        print("--- Condition Not Met: Skipping LLM Summarization ---")
        if input_data.get("analysis_error"):
             return f"Analysis could not be completed due to error: {input_data.get('analysis_error')}"
        else:
             return "No significant spending jump detected."


spend_jump_analysis_chain = (
    RunnablePassthrough.assign(
        historical_transactions=lambda x: get_historical_transactions(
            x["account_number"], x["week_start_date"], x["historical_lookback_months"]
        )
    )
    | RunnablePassthrough.assign(
        historical_weekly_avg=lambda x: calculate_historical_weekly_avg(x["historical_transactions"])
    )
    | RunnablePassthrough.assign(
        current_week_spend=lambda x: get_current_week_spending(
            x["account_number"], x["week_start_date"], x["week_end_date"]
        )
    )
    | RunnablePassthrough.assign(
        analysis_results=lambda x: analyze_spending_jump(
            x["current_week_spend"], x["historical_weekly_avg"], x["jump_percentage_threshold"]
        )
    )
    | RunnableLambda(conditionally_invoke_llm_and_parse)
)


if __name__ == "__main__":
    account = "325930050010370593"

    today = date.today()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)

    target_week_start = date.fromisoformat('2024-08-26')
    target_week_end = date.fromisoformat('2024-09-01')

    jump_percentage_threshold = 30.0
    historical_lookback_months = 6

    print(f"--- Running Weekly Spend Jump Analysis ---")
    print(f"Account: {account}")
    print(f"Analyzing Week: {target_week_start} to {target_week_end}")
    print(f"Threshold: >{jump_percentage_threshold}% increase vs {historical_lookback_months}-month avg.")

    chain_input = {
        "account_number": account,
        "week_start_date": target_week_start,
        "week_end_date": target_week_end,
        "jump_percentage_threshold": jump_percentage_threshold,
        "historical_lookback_months": historical_lookback_months
    }

    final_response = spend_jump_analysis_chain.invoke(chain_input)

    print("\n--- Analysis Result ---")
    print(final_response)