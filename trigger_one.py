import psycopg2
import psycopg2.extras
from decimal import Decimal
from typing import Dict, Any
from datetime import date, timedelta
# import pandas as pd
import re

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool

import db_connect;
import utils;

DB_SCHEMA = """
Table: statements
Columns:
    statement_id BIGSERIAL PRIMARY KEY,          -- Unique ID for each statement record in our DB
    account_number TEXT NOT NULL,                -- Bank account number (TEXT handles potential leading zeros/non-digits)
    -- account_holder_name TEXT,                 -- Consider adding if consistently available from parsing
    period_start_date TEXT NOT NULL,             -- Start date of the statement period
    period_end_date TEXT NOT NULL,               -- End date of the statement period
    statement_number INTEGER,                    -- Statement number from the bank (e.g., 40)
    currency VARCHAR(3) NOT NULL,                -- Currency code (e.g., 'RSD')
    previous_balance NUMERIC(15, 2) NOT NULL,    -- Balance at the start of the period
    total_debits NUMERIC(15, 2) NOT NULL,        -- Total amount of debits in the period
    total_credits NUMERIC(15, 2) NOT NULL,       -- Total amount of credits in the period
    new_balance NUMERIC(15, 2) NOT NULL,         -- Balance at the end of the period
    overdraft NUMERIC(15, 2),              -- Allowed overdraft (using different name than JSON for clarity)
    source_filename TEXT,                    -- Optional: track the source PDF file name
    parsed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- When this record was created/parsed

    -- Constraint to prevent inserting the exact same statement period for the same account
    CONSTRAINT unique_statement_period UNIQUE (account_number, period_start_date, period_end_date)

Table: transactions
Columns:
    transaction_pk BIGSERIAL PRIMARY KEY,       -- Unique ID for each transaction row in our DB (pk = primary key)
    statement_id BIGINT NOT NULL,               -- Foreign key linking to the statements table
    statement_transaction_id INTEGER,           -- The transaction ID/sequence number from the statement PDF (e.g., 1, 2, 3...)
    transaction_date TEXT NOT NULL,             -- Date the transaction occurred or was recorded
    value_date TEXT NOT NULL,                   -- Value date of the transaction
    description TEXT NOT NULL,                  -- Raw transaction description from the statement
    debit_amount NUMERIC(15, 2) DEFAULT 0.00,   -- Amount debited (use DEFAULT 0 for easier non-null handling)
    credit_amount NUMERIC(15, 2) DEFAULT 0.00,  -- Amount credited (use DEFAULT 0 for easier non-null handling)

    -- Columns to be populated by your AI/Analysis Agents:
    category VARCHAR(100),                      -- Transaction category (e.g., 'Salary', 'Utility-Electricity', 'ATM Withdrawal', 'Transfer Out')
    category_confidence FLOAT,                  -- Optional: Confidence score from the categorization model
    is_potential_loan_payment BOOLEAN DEFAULT FALSE, -- Flag for trigger 8 suspicion
    is_utility_payment BOOLEAN DEFAULT FALSE,   -- Flag for trigger 7 identification
    is_atm_withdrawal BOOLEAN DEFAULT FALSE,    -- Flag for trigger 4 identification
    is_eur_conversion BOOLEAN DEFAULT FALSE,    -- Flag for trigger 5 identification
    is_eur_savings BOOLEAN DEFAULT FALSE,       -- Flag for trigger 6 identification

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- When this transaction record was created

    -- Foreign key constraint
    CONSTRAINT fk_statement
        FOREIGN KEY(statement_id)
        REFERENCES statements(statement_id)
        ON DELETE CASCADE, -- If a statement is deleted, delete its transactions too

    -- Constraint to prevent inserting the same transaction (within the same statement) twice
    CONSTRAINT unique_statement_transaction_id UNIQUE (statement_id, statement_transaction_id)
"""

def clean_sql_output(sql_string: str) -> str:
    """Cleans potential markdown fences and prefixes from the LLM SQL output."""
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql_string, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = sql_string.strip()

    return cleaned

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

@tool
def execute_sql_query(query: str) -> list[tuple]:
    """
    Executes a SQL query against the PostgreSQL database and returns the results.
    Handles potential errors during execution. ONLY FOR SELECT QUERIES.
    """
    print(f"--- Executing SQL Query --- \n{query}\n--------------------------")
    results = []
    try:
        conn = db_connect.connect_to_db()
        with conn.cursor() as cur:
            cur.execute(query)
            if cur.description:
                 results = cur.fetchall()
                 results = [[float(col) if isinstance(col, Decimal) else col for col in row] for row in results]
            conn.commit()
    except Exception as e:
        print(f"Database Execution Error: {e}")
        return [("QUERY_ERROR", str(e))]
    finally:
        if conn:
            conn.close()
    print(f"--- Query Results --- \n{results}\n-----------------------")
    return results

def perform_debit_analysis(query_results: list[tuple], monthly_salary: float, percentage: float) -> Dict[str, Any]:

    """
    Analyzes the query results to calculate debit percentage vs salary.
    Assumes query_results contains the sum of debits if successful.
    """
    analysis = {
        "total_first_week_debit": 0.0,
        "estimated_salary": monthly_salary,
        "percentage": percentage,
        "threshold_amount": monthly_salary * Decimal(percentage / 100.0),
        "debits_exceed_threshold": False,
        "calculation_error": None,
        "raw_query_result": query_results
    }

    if not query_results:
        analysis["calculation_error"] = "No data returned from the database query."
        return analysis

    if len(query_results[0]) == 2 and query_results[0][0] == "QUERY_ERROR":
         analysis["calculation_error"] = f"Database query failed: {query_results[0][1]}"
         return analysis

    if len(query_results) == 1 and len(query_results[0]) == 1:
        total_debit = query_results[0][0]
        if total_debit is None:
             total_debit = 0.0
        analysis["total_first_week_debit"] = total_debit
        analysis["debits_exceed_threshold"] = analysis["total_first_week_debit"] > analysis["threshold_amount"]
    else:
        analysis["calculation_error"] = f"Unexpected query result format. Expected [[SUM]], got: {query_results}"

    return analysis


llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0)

sql_gen_prompt_template = """
You are a PostgreSQL expert. Given the database schema below and a user question, generate a SINGLE, syntactically correct PostgreSQL query to retrieve the necessary information.

Database Schema:
{schema}

User Question: {question}

Instructions:
- Only output the SQL query. No explanations, no introductory text, just the query.
- Ensure the query filters by the correct account number and the specific date range for the '{analysis_period_description}' of the given month and year.
- The query should calculate the SUM of debit amounts for that period.
- Use the 'transactions' table and potentially join with 'statements' if needed for account filtering, although filtering directly on 'transactions' using account_number (if added) or statement_id might be possible. Assume 'transactions' table has an 'account_number' column for simplicity if needed, otherwise use statement_id join. Target account: '{account_number}'. Target month/year: '{month_year}'.
- For date compare use only transaction_date from transaction table. It's type is TEXT and format is DD.MM.YYYY.
- If you need to extract Month or Year from transaction_date use TO_DATE().
- Do not calculate percentages in the SQL.

SQL Query:
"""

print(sql_gen_prompt_template)

sql_gen_prompt = PromptTemplate(
    input_variables=["question", "schema", "account_number", "month_year", "analysis_period_description"],
    template=sql_gen_prompt_template,
)

summarization_prompt_template = """
You are a helpful financial assistant. Based on the analysis results provided below, answer the original user question in one or two clear sentences.

Original User Question: {question}

Analysis Results:
- Total debits in this period: {total_first_week_debit:.2f} {currency}
- Estimated monthly salary: {estimated_salary:.2f} {currency}
- {percentage}% salary threshold amount: {threshold_amount:.2f} {currency}
- Did debits exceed {percentage}% threshold?: {debits_exceed_threshold}
- Calculation Error (if any): {calculation_error}
- You need to mention the treshold amount in your answer using quotes.
- Display year-month in words.

Answer:
"""
summarization_prompt = ChatPromptTemplate.from_template(summarization_prompt_template)

sql_generation_chain = (
    RunnablePassthrough.assign(schema=lambda x: DB_SCHEMA)
    | sql_gen_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_sql_output)
)

analysis_chain = (
    RunnablePassthrough.assign(query_results=lambda x: execute_sql_query.invoke({"query": x["sql_query"]})) 
    | RunnableLambda(lambda x: perform_debit_analysis(x["query_results"], x["monthly_salary"], x["percentage"]))
)

full_chain = (
    RunnablePassthrough.assign(sql_query=sql_generation_chain)
    | RunnablePassthrough.assign(analysis_results=analysis_chain)
    | (lambda x: {
          "question": x["question"],
          "total_first_week_debit": x["analysis_results"]["total_first_week_debit"],
          "currency": "RSD",
          "estimated_salary": x["analysis_results"]["estimated_salary"],
          "threshold_amount": x["analysis_results"]["threshold_amount"],
          "percentage": x["analysis_results"]["percentage"],
          "debits_exceed_threshold": x["analysis_results"]["debits_exceed_threshold"],
          "calculation_error": x["analysis_results"]["calculation_error"],
       })
    | summarization_prompt
    | llm
    | StrOutputParser()
)


# --- Main Execution ---
if __name__ == "__main__":
    account = "325930050010370593"
    month = '02'
    year = '2025'

    year_month = year+'-'+month+'-01'
    period_query = utils.get_period(year_month, 6)

    # print(','.join(period))

    # print('PERIOD: ', period)    
    
    # period = ["'01.08.2024'", "'01.09.2024'", "'01.10.2024'", "'01.11.2024'", "'01.12.2024'", "'01.01.2025'"]
    # period_query = ','.join(period)
    estimated_salary = load_statements_by_month('325930050010370593', period_query)

    # estimated_salary = statements[0]['total_credits']

    print("ESTIMATED SALARY: ", round(estimated_salary, 2))

    # percentage = 40.0
    # percentage = 60.0
    percentage = 80.0

    # analysis_period_description = 'first week (days 1 to 7)'
    # analysis_period_description = "first two weeks (days 1 to 14)"
    analysis_period_description = "first three weeks (days 1 to 21)"
    

    target_month_year = year+'-'+month 

    user_question = (
        f"Are the total debits for account {account} in the {analysis_period_description} "
        f"of {target_month_year} bigger than {percentage}% "
        f"of the monthly salary ({estimated_salary:.2f} RSD)?"
    )

    print(f"--- Running Analysis ---")
    print(f"Question: {user_question}")

    chain_input = {
        "question": user_question,
        "account_number": account,
        "month_year": target_month_year,
        "monthly_salary": round(estimated_salary, 2),
        "analysis_period_description": analysis_period_description, 
        "percentage": percentage
    }

    final_response = full_chain.invoke(chain_input)

    print("\n--- Final Answer ---")
    print(final_response)