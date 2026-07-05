from api.database import test_connection, create_tables
from sqlalchemy import inspect
from api.database import engine

# Test connection
test_connection()

# Create tables
create_tables()

# Verify all 5 tables exist
tables = inspect(engine).get_table_names()
expected = ['stock_data', 'indicators', 'predictions', 'backtest_results', 'trades']

print()
for t in expected:
    status = '✅' if t in tables else '❌'
    print(f'  {status}  {t}')
