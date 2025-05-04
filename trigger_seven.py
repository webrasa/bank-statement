import psycopg2
import psycopg2.extras
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
import pandas as pd
from typing import Dict, Any, Optional, List, Set
import calendar

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool

import db_connect;

UTILITY_PROVIDER_MAP = {
    'utility-infostan': 'Infostan',
    'utility-electricity': 'EPS',
    'utility-electricity (eps)': 'EPS',
    'utility-telecom': 'Telecom',
    'utility-yettel': 'Yettel',
    'utility-sbb': 'SBB',
    'utility-mts': 'MTS',
    'utility-gas': 'Gas Utility',
    'jkp infostan': 'Infostan',
    'eps ad': 'EPS',
    'eps snabdevanje': 'EPS',
    'yettel d.o.o': 'Yettel',
    'sbb ': 'SBB',
    'mts ': 'MTS',
    'telekom srbija': 'MTS',
    'beogradske elektrane': 'Heating Utility',
    'srbijagas': 'Gas Utility',
}
MIN_HISTORICAL_PAYMENTS = 4

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

def get_utility_provider(tx: Dict[str, Any]) -> Optional[str]:
    """Identifies the utility provider based on category or description keywords."""
    category = str(tx.get('category', '')).lower().strip()
    description = str(tx.get('description', '')).lower()

    if category in UTILITY_PROVIDER_MAP:
        return UTILITY_PROVIDER_MAP[category]

    for keyword, provider in UTILITY_PROVIDER_MAP.items():
        if keyword in description:
            return provider

    return None

def get_historical_debit_transactions_df(account_number: str, end_date: date, months_lookback: int) -> pd.DataFrame:
    """ Fetches historical DEBIT transactions into a DataFrame. """
    print(f"--- Fetching Historical Debit Transactions for Utility Check ---")
    start_date = (end_date.replace(day=1) - relativedelta(months=months_lookback)).replace(day=1)
    print(f"Account: {account_number}, Hist. Period: {start_date} to {end_date}")

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
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') < %s -- Use < end_date (start of current month)
        AND t.debit_amount > 0;
    """
    df = pd.DataFrame()
    conn = None
    try:
        conn = db_connect.connect_to_db()
        df = pd.read_sql(query, conn, params=(account_number, start_date, end_date), parse_dates=['transaction_date'])
        df['category'] = df['category'].fillna('').astype(str)
        df['description'] = df['description'].fillna('').astype(str)
    except Exception as e:
        print(f"Error fetching historical debits: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
    print(f"Fetched {len(df)} historical debit transactions.")
    return df

def get_analysis_month_debit_transactions(account_number: str, year: int, month: int) -> List[Dict]:
    """ Gets debit transactions for the specified analysis month. """
    start_date = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day)
    print(f"--- Getting Analysis Month Debit Transactions ---")
    print(f"Account: {account_number}, Month: {year}-{month:02d} ({start_date} to {end_date})")

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
    """
    results = execute_sql_query.invoke({"query": query, "params": (account_number, start_date, end_date)})
    if results and isinstance(results[0], dict) and "error" in results[0]:
        print(f"Error retrieving debits for {year}-{month:02d}.")
        return []
    return results

def identify_regular_utility_payees(historical_df: pd.DataFrame) -> Set[str]:
    """Identifies utility providers paid regularly in the historical data."""
    regular_payees = set()
    if historical_df.empty:
        return regular_payees

    historical_df['provider'] = historical_df.apply(get_utility_provider, axis=1)

    utility_payments = historical_df.dropna(subset=['provider']).copy()
    if utility_payments.empty:
        print("No historical utility payments identified.")
        return regular_payees

    utility_payments['month_year'] = utility_payments['transaction_date'].dt.to_period('M')

    provider_monthly_counts = utility_payments.groupby('provider')['month_year'].nunique()

    regular_payees = set(provider_monthly_counts[provider_monthly_counts >= MIN_HISTORICAL_PAYMENTS].index)

    print(f"Identified Regular Utility Payees ({MIN_HISTORICAL_PAYMENTS}+ months): {regular_payees}")
    return regular_payees


def identify_paid_utilities_this_month(current_month_txs: List[Dict]) -> Set[str]:
    """Identifies utility providers paid in the current month's transactions."""
    paid_this_month = set()
    for tx in current_month_txs:
        provider = get_utility_provider(tx)
        if provider:
            paid_this_month.add(provider)
    print(f"Utilities Paid This Month: {paid_this_month}")
    return paid_this_month


def find_missed_utilities(regular_payees: Set[str], paid_this_month: Set[str]) -> List[str]:
    """Finds utilities that were regularly paid but not paid this month."""
    missed = list(regular_payees - paid_this_month)
    print(f"Missed Regular Utilities: {missed}")
    return missed

llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)

missed_payment_alert_prompt_template = """
You are a financial monitoring assistant. An alert for potentially missed regular utility payments was triggered for account {account_number} for the month of {analysis_month_year}. This might indicate financial difficulty. List the potentially missed payments clearly.

Account: {account_number}
Month Checked: {analysis_month_year}
Lookback Period for Regularity: {historical_lookback_months} months
Minimum Payments in Lookback: {min_historical_payments}

Regularly Paid Utilities NOT Detected This Month:
{missed_utilities_list}

Generate a brief alert message (1-2 sentences) notifying the user about these specific missed payments compared to their usual pattern.
"""
missed_payment_alert_prompt = ChatPromptTemplate.from_template(missed_payment_alert_prompt_template)


def conditionally_summarize_missed_alert(input_data: Dict[str, Any]) -> str:
    """Invokes LLM to summarize if missed utility payments were found."""
    missed_utilities = input_data.get("missed_regular_utilities", [])

    if missed_utilities:
        print("--- Missed Utility Payments Found: Invoking LLM for Summarization ---")
        prompt_input = {
            "account_number": input_data["account_number"],
            "analysis_month_year": input_data["analysis_month_year_str"],
            "historical_lookback_months": input_data["historical_lookback_months"],
            "min_historical_payments": input_data["min_historical_payments"],
            "missed_utilities_list": "- " + "\n- ".join(missed_utilities) if missed_utilities else "None"
        }
        summarization_chain = missed_payment_alert_prompt | llm | StrOutputParser()
        return summarization_chain.invoke(prompt_input)
    else:
        print("--- No Missed Regular Utility Payments Detected ---")
        if not input_data.get("regular_payees_identified"):
             return "Could not identify any regularly paid utilities based on the historical data available."
        else:
             return "All regularly paid utilities appear to have been paid this month."


missed_utility_chain = (
    RunnablePassthrough.assign(
        historical_transactions_df=lambda x: get_historical_debit_transactions_df(
            x["account_number"], x["analysis_start_date"], x["historical_lookback_months"]
        )
    )
    | RunnablePassthrough.assign(
        regular_payees_identified=lambda x: identify_regular_utility_payees(
            x["historical_transactions_df"]
        )
    )
    | RunnablePassthrough.assign(
        current_month_transactions=lambda x: get_analysis_month_debit_transactions(
            x["account_number"], x["analysis_year"], x["analysis_month"]
        )
    )
    | RunnablePassthrough.assign(
        paid_this_month_set=lambda x: identify_paid_utilities_this_month(
            x["current_month_transactions"]
        )
    )
    | RunnablePassthrough.assign(
        missed_regular_utilities=lambda x: find_missed_utilities(
            x["regular_payees_identified"], x["paid_this_month_set"]
        )
    )
    | RunnableLambda(conditionally_summarize_missed_alert)
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

    historical_lookback_months = 6
    min_historical_payments_threshold = MIN_HISTORICAL_PAYMENTS

    print(f"--- Running Missed Utility Payment Check ---")
    print(f"Account: {account}")
    print(f"Analyzing Month: {analysis_year}-{analysis_month:02d}")
    print(f"Historical Lookback: {historical_lookback_months} months")
    print(f"Regularity Threshold: >= {min_historical_payments_threshold} payments in lookback")

    chain_input = {
        "account_number": account,
        "analysis_year": analysis_year,
        "analysis_month": analysis_month,
        "analysis_start_date": analysis_start_date,
        "analysis_month_year_str": f"{analysis_year}-{analysis_month:02d}",
        "historical_lookback_months": historical_lookback_months,
        "min_historical_payments": min_historical_payments_threshold
    }

    final_response = missed_utility_chain.invoke(chain_input)

    print("\n--- Analysis Result ---")
    print(final_response)