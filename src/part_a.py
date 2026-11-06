from pathlib import Path
import sqlite3, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'; DB=ROOT/'data/instamart.db'
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)
def run():
    customers=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    products=pd.read_csv(RAW/'products.csv',keep_default_na=False)
    orders=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    # Clean customers.
    customers['city']=customers['city'].astype(str).str.strip().str.title()
    customers['gender']=customers['gender'].astype(str).str.strip().str.lower().map({'male':'male','m':'male','female':'female','f':'female'}).fillna('unknown')
    customers['age']=pd.to_numeric(customers['age'],errors='coerce')
    customers.loc[~customers['age'].between(10,100),'age']=np.nan
    customers['age']=customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
    customers['age']=customers['age'].round().astype(int)
    customers['signup_date']=pd.to_datetime(customers['signup_date'])
    # Clean orders in required order.
    orders['order_ts']=pd.to_datetime(orders['order_ts'])
    orders=orders.drop_duplicates().copy()
    orders=orders.sort_values(['order_id','order_ts']).drop_duplicates('order_id',keep='last').copy()
    for c in ['quantity','discount_pct','delivery_minutes','order_amount']:
        orders[c]=pd.to_numeric(orders[c],errors='coerce')
    # Restore price-based amount when missing.
    orders=orders.merge(products[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
    miss=orders['order_amount'].isna()
    orders.loc[miss,'order_amount']=(orders.loc[miss,'price_inr']*orders.loc[miss,'quantity']*(1-orders.loc[miss,'discount_pct']/100)).round(2)
    orders.loc[orders['delivery_minutes']<0,'delivery_minutes']=np.nan
    completed=orders['status'].eq('completed')
    med=orders.loc[completed].groupby('category')['delivery_minutes'].transform('median')
    orders.loc[completed,'delivery_minutes']=orders.loc[completed].groupby('category')['delivery_minutes'].transform(lambda s:s.fillna(s.median()))
    orders.loc[~completed,'delivery_minutes']=np.nan
    orders['order_amount']=orders['order_amount'].round(2)
    products['price_inr']=pd.to_numeric(products['price_inr']).astype(int)
    orders=orders[['order_id','customer_id','product_id','quantity','order_ts','status','discount_pct','order_amount','delivery_minutes']]
    customers.to_csv(CLEAN/'customers.csv',index=False,date_format='%Y-%m-%d')
    products.to_csv(CLEAN/'products.csv',index=False)
    orders.to_csv(CLEAN/'orders.csv',index=False,date_format='%Y-%m-%d %H:%M:%S')
    if DB.exists(): DB.unlink()
    with sqlite3.connect(DB) as con:
        customers.assign(signup_date=customers.signup_date.dt.strftime('%Y-%m-%d')).to_sql('customers',con,index=False)
        products.to_sql('products',con,index=False)
        orders.assign(order_ts=orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')).to_sql('orders',con,index=False)
    # Seven required queries, saved and run.
    sqls=[
    "SELECT customer_id, SUM(order_amount) AS spend FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000 ORDER BY customer_id;",
    "SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;",
    "SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;",
    "SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL ORDER BY c.customer_id;",
    "WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM customers c JOIN orders o ON c.customer_id=o.customer_id WHERE o.status='completed' GROUP BY c.customer_id,c.city), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT spend.customer_id,spend.city,spend.total_spend FROM spend,avg_spend WHERE spend.total_spend>avg_spend.avg_total ORDER BY spend.customer_id;",
    "WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT customer_id,city,ROUND(total_spend,2) total_spend,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rank_in_city,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rank_in_city,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) row_number_in_city FROM spend ORDER BY city,rank_in_city,customer_id;",
    "WITH daily AS (SELECT substr(order_ts,1,10) day,ROUND(SUM(order_amount),2) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,ROUND(SUM(revenue) OVER(ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),2) cumulative_revenue FROM daily ORDER BY day;"
    ]
    (ROOT/'sql/queries.sql').write_text('\n\n'.join(f'-- Q{i}\n{q}' for i,q in enumerate(sqls,1))+'\n',encoding='utf-8')
    # checkpoint validation
    with sqlite3.connect(DB) as con:
        n= pd.read_sql_query('select count(*) n from orders',con).iloc[0,0]
        status=pd.read_sql_query("select status,count(*) n from orders group by status",con).set_index('status').n.to_dict()
        total=pd.read_sql_query("select sum(order_amount) s from orders where status='completed'",con).iloc[0,0]
    checks=[len(customers)==1200, len(products)==40, n==12309, status.get('completed')==11110, status.get('cancelled')==1199,
            customers['city'].nunique()==6, customers['age'].isna().sum()==0, orders.loc[orders.status.eq('completed'),'delivery_minutes'].isna().sum()==0,
            orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().sum()==1199, round(float(total),2)==2998668.80]
    if not all(checks): raise AssertionError(f'Part A checkpoint failed: {checks}; statuses={status}; total={total}')
    # Audit report and SQL/pandas cross-checks are generated reproducibly from source data.
    raw_orders=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    raw_customers=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    raw_products=pd.read_csv(RAW/'products.csv',keep_default_na=False)
    def missing_report(df):
        out={}
        for col in df.columns:
            blank=df[col].astype(str).str.strip().eq('')
            out[col]={'count':int(blank.sum()),'percent':round(float(blank.mean()*100),2)}
        return out
    audit={'raw_tables':{},'specific_counts':{
      'orders_rows':len(raw_orders),'exact_duplicate_rows':int(raw_orders.duplicated().sum()),'pending_rows':int((raw_orders.status=='pending').sum()),
      'distinct_order_ids':int(raw_orders.order_id.nunique()),'nonnumeric_amounts':int(pd.to_numeric(raw_orders.order_amount,errors='coerce').isna().sum()),
      'negative_delivery_minutes':int((pd.to_numeric(raw_orders.delivery_minutes,errors='coerce')<0).sum()),'blank_delivery_minutes':int(raw_orders.delivery_minutes.astype(str).str.strip().eq('').sum()),
      'blank_ages':int(raw_customers.age.astype(str).str.strip().eq('').sum()),'ages_outside_10_100':int(((pd.to_numeric(raw_customers.age,errors='coerce')<10)|(pd.to_numeric(raw_customers.age,errors='coerce')>100)).sum()),
      'distinct_raw_city_spellings':int(raw_customers.city.nunique()),'distinct_nonblank_raw_gender_spellings':int(raw_customers.loc[raw_customers.gender.astype(str).str.strip().ne(''),'gender'].nunique()),'blank_genders':int(raw_customers.gender.astype(str).str.strip().eq('').sum())}}
    for name,df in [('orders',raw_orders),('customers',raw_customers),('products',raw_products)]:
        audit['raw_tables'][name]={'shape':list(df.shape),'dtypes':{k:str(v) for k,v in df.dtypes.items()},'missing':missing_report(df)}
    sql_text=(ROOT/'sql/queries.sql').read_text(encoding='utf-8')
    import re
    query_blocks=re.split(r'-- Q[1-7]\s*\n',sql_text)
    query_blocks=[q.strip() for q in query_blocks if q.strip()]
    qframes=[]
    with sqlite3.connect(DB) as con:
        for q in query_blocks: qframes.append(pd.read_sql_query(q,con))
    revenue_q2=qframes[1]
    q2top=revenue_q2.sort_values('revenue',ascending=False).iloc[0].to_dict()
    q5avg=pd.read_sql_query("SELECT AVG(total_spend) avg FROM (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id)",sqlite3.connect(DB)).iloc[0,0]
    q6=qframes[5]
    daily=qframes[6]
    analysis_orders=orders.merge(customers[['customer_id','city']],on='customer_id',how='left').merge(products[['product_id','category']],on='product_id',how='left')
    comp=analysis_orders[analysis_orders.status.eq('completed')].copy()
    q1={'rows':len(qframes[0])}; q2={'rows':len(revenue_q2),'total_revenue':round(float(revenue_q2.revenue.sum()),2),'largest_cell':q2top}
    q3={'rows':len(qframes[2])}; q4={'rows':len(qframes[3])}; q5={'rows':len(qframes[4]),'average_completed_spend':round(float(q5avg),2)}
    topcity=q6.sort_values('rank_in_city').groupby('city',as_index=False).first()[['city','customer_id','total_spend']].to_dict(orient='records')
    q6summary={'rows':len(q6),'rank_differs_from_row_number':int((q6.rank_in_city!=q6.row_number_in_city).sum()),'top_customer_by_city':topcity}
    q7={'rows':len(daily),'last_cumulative_revenue':round(float(daily.cumulative_revenue.iloc[-1]),2)}
    pivot=pd.pivot_table(comp,index='city',columns='category',values='order_amount',aggfunc='sum',margins=True,margins_name='All',fill_value=0)
    citystats=comp.groupby('city').agg(sum=('order_amount','sum'),mean=('order_amount','mean'),count=('order_amount','count')).reset_index()
    vals=comp.order_amount.astype(float); q1v,q3v=vals.quantile(.25),vals.quantile(.75); iqr=q3v-q1v; low=q1v-1.5*iqr; high=q3v+1.5*iqr
    corr=comp[['order_amount','discount_pct','quantity','delivery_minutes']].corr().to_dict()
    audit['cleaned']={'city_counts':customers.city.value_counts().to_dict(),'gender_counts':customers.gender.value_counts().to_dict(),'age_missing':int(customers.age.isna().sum()),'age_min':int(customers.age.min()),'age_max':int(customers.age.max()),'order_rows':len(orders),'status_counts':orders.status.value_counts().to_dict(),'completed_amount_sum':round(float(comp.order_amount.sum()),2),'completed_delivery_missing':int(comp.delivery_minutes.isna().sum()),'cancelled_delivery_missing':int(orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().sum())}
    audit['sql']={'Q1':q1,'Q2':q2,'Q3':q3,'Q4':q4,'Q5':q5,'Q6':q6summary,'Q7':q7}
    audit['pandas']={'pivot_grand_total':round(float(pivot.loc['All','All']),2),'city_stats':citystats.to_dict(orient='records'),'amount_mean':float(vals.mean()),'amount_median':float(vals.median()),'amount_skew':float(vals.skew()),'q1':float(q1v),'q3':float(q3v),'lower_fence':float(low),'upper_fence':float(high),'outliers_below':int((vals<low).sum()),'outliers_above':int((vals>high).sum()),'correlation':corr}
    (OUT/'part_a_metrics.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': run()
