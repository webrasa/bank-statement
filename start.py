
import json
import psycopg2.extras
import ollama
from ollama import AsyncClient
import asyncio
import db_connect;

def load_data_from_file(file_path):
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the contents of the file
            file_contents = file.read()
            resp = json.loads(file_contents)
            # print(resp['account_details'])
            # Close the file
            file.close()
            return resp

    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

def load_statements_by_month(month):
    connection = db_connect.connect_to_db()
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    # query = "SELECT st.statement_id, st.period_start_date, st.period_end_date, st.total_debits, st.total_credits, st.new_balance " \
    query = "SELECT * FROM statements WHERE period_start_date = '01.08.2024';"
    cursor.execute(query)
    records = cursor.fetchall()
    standard_dicts = [dict(row) for row in records]

    return standard_dicts

def load_data_by_account_num (account_number):
    print("Parameter: ", account_number['account_number'])
    try:
    # Query the database
        connection = db_connect.connect_to_db()
        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # query = "SELECT st.statement_id, st.period_start_date, st.period_end_date, st.total_debits, st.total_credits, st.new_balance " \
        query = "SELECT tr.transaction_date, tr.description, tr.debit_amount, tr.credit_amount " \
        "FROM statements AS st INNER JOIN transactions AS tr ON st.statement_id = tr.statement_id " \
        "WHERE st.account_number = '"+account_number['account_number']+"' AND tr.transaction_date IN('01.08.2024', '02.08.2024', '03.08.2024', '04.08.2024', '05.08.2024', '06.08.2024', '07.08.2024');"
        cursor.execute(query)
        records = cursor.fetchall()

        # Print the results
        # results = []
        # for record in records:
        #     # print(record)
        #     results.append(record)
        standard_dicts = [dict(row) for row in records]

        return standard_dicts

        # Close the cursor and connection
        # cursor.close()
        # connection.close()

    except Exception as e:
        print("An error occurred:", e)

    finally:
            # Close the connection in the finally block to ensure it happens
            if connection:
                connection.close()
                print("Database connection closed.")

def store_data():
    try:
        # Connect to the database
        connection = db_connect.connect_to_db()
        cursor = connection.cursor()
        file_name = "data/full_transactions_s40.json"

        data = load_data_from_file(file_name)
        statement_id = data['account_details']['statement_number']
        
        insert_query = "INSERT INTO statements " \
        "(statement_id, account_number, period_start_date, period_end_date, statement_number, currency, previous_balance, total_debits, total_credits, new_balance, overdraft, source_filename, parsed_at)" \
        "VALUES" \
        "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

        data_to_insert = (
            statement_id,
            data['account_details']['account_number'],
            data['account_details']['period'].split(" - ")[0],
            data['account_details']['period'].split(" - ")[1],
            data['account_details']['statement_number'],
            data['account_details']['currency'],
            data['account_details']['previous_balance'],
            data['account_details']['total_debits'],
            data['account_details']['total_credits'],
            data['account_details']['new_balance'],
            data['account_details']['overdraft'],
            file_name,
            'NOW()'
        )
        cursor.execute(insert_query, data_to_insert)
        connection.commit()

        for transaction in data['transactions']:
            insert_query = "INSERT INTO transactions " \
            "(statement_id, transaction_date, value_date, description, debit_amount, credit_amount)" \
            "VALUES" \
            "(%s, %s, %s, %s, %s, %s)"
            data_to_insert = (
                statement_id,
                transaction['transaction_date'],
                transaction['value_date'],
                transaction['description'],
                transaction['debit'],
                transaction['credit'],
            )
            cursor.execute(insert_query, data_to_insert)

        print("Data inserted successfully!")

        connection.commit()
        cursor.close()
        connection.close()

    except Exception as e:
        print("An error occurred:", e)

# Start the app
# results = load_data_by_account_num("325930050010370593")

# print(results)

# load_data_from_file("data/full_transactions_s40.json")
# store_data()

async def main():
    client = AsyncClient()

    tools = [
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "load_statements_by_month",
        #         "description": "Fetch data by month given as a parameter",
        #         "parameters": {
        #             "type": "object", # MUST be object
        #             "properties": {
        #                 "month": { # The actual parameter name your Python function expects
        #                     "type": "string",
        #                     "description": "The month and year in MM.YYYY format (e.g., '08.2024')."
        #                 }
        #             },
        #         },
        #     },
        # },
        {
            "type": "function",
            "function": {
                "name": "load_data_by_account_num",
                "description": "Fetch data by account number",
                "parameters": {
                    "type": "object", # MUST be object
                    "properties": {
                        "account_number": { # The actual parameter name your Python function expects
                            "type": "string",
                            "description": "The bank account number as a string."
                        }
                    },
                    "required": ["account_number"] # Correct required parameter name
                },
            },
        },
    ]

    results_month = load_statements_by_month("08.2024")

    # results_month = [str(item) for item in results_month]

    # results = load_data_by_account_num("325930050010370593")

    # Convert the list of dictionaries to a list of strings
    # results_month = [str(item) for item in results_month]


    # categorize_prompt = f"""
    # You are an assistant that categorizes bank statement items.
    # **Instructions:**
    # - Period of time is 7 days.
    # - Amount of credit for this period is '{results_month[0]['total_credits']}'.
    # - Return the result **only** as a valid JSON object.
    # - Do **not** include any explanations, greetings, or additional text.
    # - Use double quotes (`"`) for all strings.
    # - Ensure the JSON is properly formatted.

    # **Question:**
    # Does sum of debit for this period exceeds the 40% of the amount of credit for whole month?

    # **Data**
    # {', '.join(results)}
    # """
    # categorize_prompt = f"""
    # **Instructions:**
    # Can you tell me the name of the person who is the owner of the account based on this bank statement?
    # **Example Format:**
    # John Doe
    # **Bank statement items:**
    # {', '.join(results)}
    # """

    # messages = [{"role": "user", "content": categorize_prompt}]

    # print(messages)

    # response = ollama.chat(
    #     model="deepseek-r1:8b-llama-distill-fp16",
    #     # model="deepseek-r1:32b",
    #     # model="llama3.2:1b",
    #     messages=messages,
    #     stream=True
    # )
    # # print(response.message.content)
    # for chunk in response:
    #     print(chunk["message"]["content"], end="", flush=True)

    # response = ollama.chat(
    #     model="deepseek-r1:8b-llama-distill-fp16",
    #     # model="deepseek-r1:32b",
    #     # model="llama3.2:1b",
    #     messages=messages,
    #     stream=False
    # )
    # print(response.message.content)

    prompt = f"""
        For week fetch statements data, use the 'load_data_by_account_num' function using the following parameter:
    - account_number: '{results_month[0]['account_number']}' as string
    """

    # print(prompt)


    messages = [{"role": "user", "content": prompt}]

    response2 = await client.chat(
        # model="deepseek-r1:8b-llama-distill-fp16",
        model="llama3.2:1b",
        messages=messages,
        tools=tools,
    )

    messages.append(response2["message"])

    if response2["message"].get("tool_calls"):
        print("Function calls made by the model:")

        available_functions = {
            "load_data_by_account_num": load_data_by_account_num,
        }
        item_details = []
        for tool_call in response2["message"]["tool_calls"]:
            # print(tool_call["function"]["name"])
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            function_to_call = available_functions.get(function_name)
            print(f"Function name: {function_to_call}")
            print(f"Arguments: {arguments}")
            if function_to_call:
                result = function_to_call(arguments)
                # Add function response to the conversation
                # messages.append(
                #     {
                #         "role": "tool",
                #         "content": json.dumps(result),
                #     }
                # )
                item_details.append(result)

                print(item_details)
        # Store the details for later use
        item_details = []

    # fetch_prompt = f"""
    # You are an assistant that analyze bank statement items.
    # **Instructions:**
    # - Period of time is 7 days.
    # - Amount of credit for this period is '{results_month[0]['total_credits']}'.
    # - Return the result **only** as a valid JSON object.
    # - Do **not** include any explanations, greetings, or additional text.
    # - Use double quotes (`"`) for all strings.
    # - Ensure the JSON is properly formatted.

    # **Question:**
    # Does sum of debit for this period exceeds the 40% of the amount of credit for whole month?

    # **Data**
    # {', '.join(results)}

    # """

    # messages = [{"role": "user", "content": fetch_prompt}]

    # response2 = await client.chat(
    #     model="deepseek-r1:8b-llama-distill-fp16",
    #     messages=messages,
    #     tools=tools,
    # )

    # print(response2.message.content)

asyncio.run(main())

