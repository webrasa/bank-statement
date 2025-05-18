import psycopg2
import psycopg2.extras
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, Optional, List

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool

import db_connect;
import utils;

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
    # Return list of dictionaries
    return [dict(row) for row in results]

def load_statements_by_month(account_number, period):
    connection = db_connect.connect_to_db()
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = "SELECT * FROM statements WHERE account_number = '"+account_number+"' AND  period_start_date IN ("+period+");"
    cursor.execute(query)
    records = cursor.fetchall()
    # Calculate average salary
    total_salary = float(0.0)
    for row in records:
        # print("row: ", row['total_credits'])
        total_salary += float(row['total_credits'])
    # print("total salary: ", total_salary)
    
    # standard_dicts = [dict(row) for row in records]

    return Decimal(total_salary / len(records))


def get_transactions_for_period(account_number: str, start_date: date, end_date: date) -> List[Dict]:
    """Fetches transactions for a given account and date range."""
    print(f"--- Fetching Transactions for Period ---")
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
        AND t.debit_amount > 0; -- Only fetch debits
    """
    results = execute_sql_query.invoke({"query": query, "params": (account_number, start_date, end_date)})

    if results and isinstance(results[0], dict) and "error" in results[0]:
        print(f"Error fetching transactions: {results[0]['error']}")
        return []

    return results

def find_large_transactions(
    transactions: List[Dict],
    average_salary: float,
    threshold_percentage: float
) -> List[Dict]:
    """
    Filters transactions to find those exceeding the salary threshold,
    excluding specified categories like ATM Withdrawals.
    """
    large_transactions_found = []
    if not transactions:
        return large_transactions_found

    threshold_amount = Decimal(str(average_salary)) * (Decimal(str(threshold_percentage)) / Decimal("100.0"))
    print(f"Calculated threshold amount: {threshold_amount:.2f}")
    excluded_categories = {'atm withdrawal'}

    for tx in transactions:
        try:
            debit_amount = Decimal(str(tx.get('debit_amount', '0.0')))
            category = str(tx.get('category', '')).lower()
            transaction_date_str = tx.get('transaction_date')

            formatted_date = 'N/A'
            if transaction_date_str:
                if isinstance(transaction_date_str, date):
                     formatted_date = transaction_date_str.strftime('%Y-%m-%d')
                elif isinstance(transaction_date_str, str):
                    try:
                         transaction_date_obj = datetime.strptime(transaction_date_str, '%d.%m.%Y').date()
                         formatted_date = transaction_date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                         print(f"Warning: Could not parse date string '{transaction_date_str}' for tx {tx.get('transaction_pk')}. Expected YYYY-MM-DD.")
                         formatted_date = str(transaction_date_str)

            if category not in excluded_categories and debit_amount > threshold_amount:
                print(f"Found large transaction: PK={tx.get('transaction_pk')}, Amount={debit_amount}, Category='{category}'")
                percentage_of_salary = (debit_amount / Decimal(str(average_salary))) * Decimal("100.0")
                tx_info = {
                    "transaction_pk": tx.get('transaction_pk'),
                    "date": formatted_date, 
                    "description": tx.get('description', 'N/A')[:100],
                    "amount": float(debit_amount),
                    "percentage_of_salary": float(percentage_of_salary.quantize(Decimal("0.1")))
                }
                large_transactions_found.append(tx_info)
        except Exception as e:
            print(f"Error processing transaction {tx.get('transaction_pk')}: {e}")
            continue

    return large_transactions_found


llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)

large_tx_summary_prompt_template = """
You are a financial monitoring assistant. A large transaction alert was triggered for account {account_number}. Summarize the details clearly and concisely based on the information provided.

Alert Threshold: {threshold_percentage:.1f}% of average monthly salary ({average_salary:.2f} RSD)
Average Salary: {average_salary:.2f} RSD

Large Transaction(s) Found:
{large_transactions_details}

Generate a brief alert message (1-3 sentences) notifying the user about the large transaction(s). Mention the largest one if there are multiple.
"""
large_tx_summary_prompt = ChatPromptTemplate.from_template(large_tx_summary_prompt_template)

def format_large_tx_details(transactions: List[Dict]) -> str:
    """Formats the list of large transactions for the LLM prompt."""
    if not transactions:
        return "None"
    details = []
    for tx in transactions:
        details.append(
            f"- Date: {tx['date']}, Amount: {tx['amount']:.2f} RSD ({tx['percentage_of_salary']:.1f}% of salary), Desc: {tx['description']}"
        )
    return "\n".join(details)


def conditionally_summarize_large_tx(input_data: Dict[str, Any]) -> str:
    """Invokes LLM to summarize if large transactions were found."""
    large_transactions = input_data.get("large_transactions_found", [])

    if large_transactions:
        print("--- Large Transactions Found: Invoking LLM for Summarization ---")
        prompt_input = {
            "account_number": input_data["account_number"],
            "threshold_percentage": input_data["threshold_percentage"],
            "average_salary": input_data["average_monthly_salary"],
            "large_transactions_details": format_large_tx_details(large_transactions)
        }
        summarization_chain = large_tx_summary_prompt | llm | StrOutputParser()
        return summarization_chain.invoke(prompt_input)
    else:
        print("--- No Large Transactions Found ---")
        return "No single large transactions exceeding the threshold were found in the specified period (excluding ATM withdrawals)."



large_transaction_chain = (
    RunnablePassthrough.assign(
        raw_transactions=lambda x: get_transactions_for_period(
            x["account_number"], x["start_date"], x["end_date"]
        )
    )
    | RunnablePassthrough.assign(
        large_transactions_found=lambda x: find_large_transactions(
            x["raw_transactions"],
            x["average_monthly_salary"],
            x["threshold_percentage"]
        )
    )
    | RunnableLambda(conditionally_summarize_large_tx)
)


if __name__ == "__main__":
    account = "325930050010370593"

    today = date.today()
    # start_date_check = today - timedelta(days=7)
    # end_date_check = today # Check up to today

    month = '02'
    year = '2025'
    start_date_check = date.fromisoformat(year+'-'+month+'-01')
    end_date_check = date.fromisoformat(year+'-'+month+'-28')

    year_month = year+'-'+month+'-01'
    period_query = utils.get_period(year_month, 6)

    average_monthly_salary = load_statements_by_month('325930050010370593', period_query)

    large_tx_threshold_percentage = 30.0

    print(f"--- Running Single Large Transaction Check ---")
    print(f"Account: {account}")
    print(f"Checking Period: {start_date_check} to {end_date_check}")
    print(f"Average Salary: {average_monthly_salary:.2f} RSD")
    print(f"Threshold: >{large_tx_threshold_percentage}% of salary (excluding ATM)")

    chain_input = {
        "account_number": account,
        "start_date": start_date_check,
        "end_date": end_date_check,
        "average_monthly_salary": average_monthly_salary,
        "threshold_percentage": large_tx_threshold_percentage
    }

    final_response = large_transaction_chain.invoke(chain_input)

    print("\n--- Analysis Result ---")
    print(final_response)