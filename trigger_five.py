import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta, date
from decimal import Decimal # Use Decimal for currency
from typing import Dict, Any, Optional, List

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


def get_credit_transactions_for_period(account_number: str, start_date: date, end_date: date) -> List[Dict]:
    """Fetches CREDIT transactions for a given account and date range."""
    print(f"--- Fetching Credit Transactions for Period ---")
    print(f"Account: {account_number}, Period: {start_date} to {end_date}")

    query = """
        SELECT
            t.transaction_pk,
            t.transaction_date,
            t.description,
            t.credit_amount,
            t.category
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        WHERE s.account_number = %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') >= %s
        AND TO_DATE(t.transaction_date, 'DD.MM.YYYY') <= %s
        AND t.credit_amount > 0; -- Only fetch credits
    """
    results = execute_sql_query.invoke({"query": query, "params": (account_number, start_date, end_date)})

    if results and isinstance(results[0], dict) and "error" in results[0]:
        print(f"Error fetching credit transactions: {results[0]['error']}")
        return []
    return results


def find_eur_rsd_conversion_credits(
    transactions: List[Dict],
    rsd_threshold: Decimal, 
    target_category: str = 'currency conversion in (eur->rsd)',
    keywords: List[str] = ['konverzija', 'conversion', 'prodaja dev', 'eur']
) -> List[Dict]:
    """
    Filters credit transactions to find likely EUR->RSD conversions exceeding the threshold.
    """
    found_conversions = []
    if not transactions:
        return found_conversions

    print(f"Filtering for EUR->RSD conversions > {rsd_threshold:.2f} RSD")

    for tx in transactions:
        try:
            credit_amount = Decimal(str(tx.get('credit_amount', '0.0')))
            category = str(tx.get('category', '')).lower().strip()
            description = str(tx.get('description', '')).lower()
            transaction_date_val = tx.get('transaction_date')

            is_potential_conversion = False

            if category == target_category:
                is_potential_conversion = True
            elif any(keyword in description for keyword in keywords):
                 is_potential_conversion = True


            if is_potential_conversion and credit_amount > rsd_threshold:
                print(f"Found potential large conversion: PK={tx.get('transaction_pk')}, Amount={credit_amount:.2f} RSD, Category='{category}'")

                formatted_date = 'N/A'
                if transaction_date_val:
                    if isinstance(transaction_date_val, date):
                        formatted_date = transaction_date_val.strftime('%Y-%m-%d')
                    elif isinstance(transaction_date_val, str):
                        try: formatted_date = datetime.strptime(transaction_date_val, '%Y-%m-%d').strftime('%Y-%m-%d')
                        except: formatted_date = transaction_date_val

                tx_info = {
                    "transaction_pk": tx.get('transaction_pk'),
                    "date": formatted_date,
                    "description": tx.get('description', 'N/A')[:100],
                    "amount_rsd": float(credit_amount),
                }
                found_conversions.append(tx_info)

        except Exception as e:
            print(f"Error processing credit transaction {tx.get('transaction_pk')}: {type(e).__name__} - {e}")
            continue

    return found_conversions



llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0) # Use your model

conversion_alert_prompt_template = """
You are a financial monitoring assistant. An alert for a potentially significant currency conversion (EUR to RSD) credited to account {account_number} was triggered. This can sometimes indicate financial difficulty. Summarize the details concisely.

Alert Threshold: Credit amount likely equivalent to > {eur_threshold:.0f} EUR (~{rsd_threshold:.0f} RSD)
Account: {account_number}
Period Checked: {start_date} to {end_date}

Detected Conversion Credit(s):
{conversion_details}

Generate a brief alert message (1-2 sentences) notifying the user about this specific type of transaction exceeding the threshold, mentioning it's being monitored as a potential indicator.
"""
conversion_alert_prompt = ChatPromptTemplate.from_template(conversion_alert_prompt_template)

def format_conversion_details_for_prompt(transactions: List[Dict]) -> str:
    """Formats the list of found conversions for the LLM prompt."""
    if not transactions:
        return "None detected in the period."
    details = []
    for tx in transactions:
        details.append(
            f"- Date: {tx['date']}, Amount Credited: {tx['amount_rsd']:.2f} RSD, Desc: {tx['description']}"
        )
    return "\n".join(details)


def conditionally_summarize_conversion_alert(input_data: Dict[str, Any]) -> str:
    """Invokes LLM to summarize if significant conversions were found."""
    found_conversions = input_data.get("found_eur_rsd_conversions", [])

    if found_conversions:
        print("--- EUR->RSD Conversion Found: Invoking LLM for Summarization ---")
        prompt_input = {
            "account_number": input_data["account_number"],
            "eur_threshold": input_data["eur_threshold"],
            "rsd_threshold": input_data["rsd_threshold"],
            "start_date": input_data["start_date"].strftime('%Y-%m-%d'),
            "end_date": input_data["end_date"].strftime('%Y-%m-%d'),
            "conversion_details": format_conversion_details_for_prompt(found_conversions)
        }
        summarization_chain = conversion_alert_prompt | llm | StrOutputParser()
        return summarization_chain.invoke(prompt_input)
    else:
        print("--- No Significant EUR->RSD Conversions Found ---")
        return "No significant EUR->RSD conversion credits exceeding the threshold were found in the specified period."



eur_conversion_chain = (
    RunnablePassthrough.assign(
        credit_transactions=lambda x: get_credit_transactions_for_period(
            x["account_number"], x["start_date"], x["end_date"]
        )
    )
    | RunnablePassthrough.assign(
        found_eur_rsd_conversions=lambda x: find_eur_rsd_conversion_credits(
            x["credit_transactions"],
            x["rsd_threshold"]
        )
    )
    | RunnableLambda(conditionally_summarize_conversion_alert)
)


if __name__ == "__main__":
    account = "325930050010370593"

    # today = date.today()
    today = date.fromisoformat('2025-02-01')
    start_date_check = today - timedelta(days=30)
    end_date_check = today

    eur_threshold_value = 200.0
    approx_eur_to_rsd_rate = Decimal("117.3")
    rsd_threshold_value = Decimal(str(eur_threshold_value)) * approx_eur_to_rsd_rate

    print(f"--- Running EUR->RSD Conversion Check ---")
    print(f"Account: {account}")
    print(f"Checking Period: {start_date_check} to {end_date_check}")
    print(f"Threshold: > {eur_threshold_value} EUR (~{rsd_threshold_value:.2f} RSD)")

    chain_input = {
        "account_number": account,
        "start_date": start_date_check,
        "end_date": end_date_check,
        "eur_threshold": eur_threshold_value,
        "rsd_threshold": rsd_threshold_value,
    }

    final_response = eur_conversion_chain.invoke(chain_input)

    print("\n--- Analysis Result ---")
    print(final_response)