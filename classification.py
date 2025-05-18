import psycopg2
import psycopg2.extras

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import db_connect;


PREDEFINED_CATEGORIES = [
    "Salary/Income",
    "Groceries",
    "Restaurants/Dining",
    "Utilities (Electricity/Gas/Water)",
    "Utilities (Internet/Cable/Phone)",
    "Rent/Mortgage",
    "Loan Repayment",
    "ATM Withdrawal",
    "Transfer Out",
    "Transfer In",
    "Shopping (Clothing)",
    "Shopping (Electronics)",
    "Shopping (Other)",
    "Travel/Transportation",
    "Healthcare/Pharmacy",
    "Entertainment",
    "Bank Fees",
    "Savings/Investments",
    "Cash Deposit",
    "Misdemeanor fine",
    "Other/Unknown"
]

def load_transactions_by_statement_id(statement_id):
    connection = db_connect.connect_to_db()
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = "SELECT * FROM transactions WHERE statement_id = "+statement_id+";"
    cursor.execute(query)
    records = cursor.fetchall()

    return records

def update_transaction_category(transaction_id, category):
    connection = db_connect.connect_to_db()
    cursor = connection.cursor()
    query = "UPDATE transactions SET category = '"+category+"' WHERE transaction_pk = "+transaction_id+";"
    cursor.execute(query)
    connection.commit()
    cursor.close()
    connection.close()

llm = ChatOllama(model="llama3.1:8b-instruct-fp16", temperature=0.0) # Adjust model name if needed

system_message_content = f"""
You are an expert bank transaction classifier. Your task is to categorize the given transaction description into ONE of the following predefined categories.
Output ONLY the category name. 
Only credits can be categorized as income.
Do NOT add any other text, explanations, or markdown.

Predefined Categories:
{', '.join(PREDEFINED_CATEGORIES)}

Examples:
Description: UPLATA PLATE OTP BANKA SRBIJA A.D.
Category: Salary/Income

Description: MAXI 345 R KOD LOK
Category: Groceries

Description: EPS AD DISTRIBUCIJA BEOGRAD
Category: Utilities (Electricity/Gas/Water)

Description: PODIZANJE GOTOVINE BANKOMAT 123
Category: ATM Withdrawal
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message_content),
    HumanMessagePromptTemplate.from_template("Transaction Description: {description}")
])

def validate_and_normalize_category(llm_output: str) -> str:
    """
    Validates the LLM output against predefined categories.
    Normalizes by stripping whitespace and attempting a case-insensitive match.
    """
    cleaned_output = llm_output.strip()

    for category in PREDEFINED_CATEGORIES:
        if category.lower() == cleaned_output.lower():
            return category

    print(f"Warning: LLM output '{cleaned_output}' not an exact predefined category. Falling back to 'Other/Unknown'.")
    return "Other/Unknown"

transaction_classification_chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(validate_and_normalize_category)
)

if __name__ == "__main__":
    test_transactions = load_transactions_by_statement_id('40')

    print("--- Classifying Transactions ---")
    classified_transactions = []
    for tx_data in test_transactions:
        description_to_classify = tx_data["description"]
        amount = tx_data["debit_amount"] if tx_data["debit_amount"] > 0 else tx_data["credit_amount"]
        try:
            predicted_category = transaction_classification_chain.invoke({"description": description_to_classify})
            print(f"Id: \"{tx_data["transaction_pk"]}\" Description: \"{description_to_classify}\"  => Category: \"{predicted_category}\" Amount: \"{amount}\"")
            update_transaction_category(str(tx_data["transaction_pk"]), predicted_category)
            classified_transactions.append({**tx_data, "predicted_category": predicted_category})
        except Exception as e:
            print(f"Error classifying description \"{description_to_classify}\": {e}")
            classified_transactions.append({**tx_data, "predicted_category": "Error", "error_message": str(e)})