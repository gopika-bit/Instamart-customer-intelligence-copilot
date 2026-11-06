"""Validate the persisted Part A reference checkpoints."""
from pathlib import Path
import sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    c=pd.read_csv(ROOT/'data/clean/customers.csv')
    p=pd.read_csv(ROOT/'data/clean/products.csv')
    o=pd.read_csv(ROOT/'data/clean/orders.csv')
    with sqlite3.connect(ROOT/'data/instamart.db') as con:
        tables=pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'",con).name.tolist()
        counts=pd.read_sql_query('SELECT status, COUNT(*) n FROM orders GROUP BY status',con).set_index('status').n.to_dict()
        revenue=float(pd.read_sql_query("SELECT SUM(order_amount) v FROM orders WHERE status='completed'",con).iloc[0,0])
    checks=[len(c)==1200,len(p)==40,len(o)==12309,c.city.nunique()==6,c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30},
            c.age.notna().all(),c.age.between(18,60).all(),counts=={'cancelled':1199,'completed':11110},round(revenue,2)==2998668.80,
            o.loc[o.status.eq('completed'),'delivery_minutes'].notna().all(),o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().all(),{'customers','products','orders'}.issubset(set(tables))]
    assert all(checks),f'Validation failed: {checks}'
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': main()
