-- Ensures script can be run multiple times
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS statements;

-- Table to store summary information for each bank statement
CREATE TABLE statements (
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
);

-- Add index on account number for faster lookups per account
CREATE INDEX idx_statements_account_number ON statements (account_number);


-- Table to store individual transactions from the statements
CREATE TABLE transactions (
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
);

-- Add indexes for faster querying based on common analysis criteria
CREATE INDEX idx_transactions_statement_id ON transactions (statement_id);
CREATE INDEX idx_transactions_transaction_date ON transactions (transaction_date);
CREATE INDEX idx_transactions_category ON transactions (category);


-- Optional: Consider a separate table for categories if they become complex
-- CREATE TABLE transaction_categories (
--    category_id SERIAL PRIMARY KEY,
--    category_name VARCHAR(100) UNIQUE NOT NULL,
--    description TEXT,
--    parent_category_id INTEGER REFERENCES transaction_categories(category_id) -- For hierarchical categories
-- );
-- If using this, change transactions.category to:
-- category_id INTEGER REFERENCES transaction_categories(category_id)