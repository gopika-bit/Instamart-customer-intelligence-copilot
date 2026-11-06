-- Q1
SELECT customer_id, SUM(order_amount) AS spend FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000 ORDER BY customer_id;

-- Q2
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;

-- Q3
SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4
SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5
WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM customers c JOIN orders o ON c.customer_id=o.customer_id WHERE o.status='completed' GROUP BY c.customer_id,c.city), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT spend.customer_id,spend.city,spend.total_spend FROM spend,avg_spend WHERE spend.total_spend>avg_spend.avg_total ORDER BY spend.customer_id;

-- Q6
WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT customer_id,city,ROUND(total_spend,2) total_spend,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rank_in_city,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rank_in_city,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) row_number_in_city FROM spend ORDER BY city,rank_in_city,customer_id;

-- Q7
WITH daily AS (SELECT substr(order_ts,1,10) day,ROUND(SUM(order_amount),2) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,ROUND(SUM(revenue) OVER(ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),2) cumulative_revenue FROM daily ORDER BY day;
