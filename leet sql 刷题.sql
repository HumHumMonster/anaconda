-- 197 找到比前一天温度高的数据
SELECT w1.id FROM Weather W1 , Weather W2
WHERE DATEDIFF(W1.recordDate , W2.recordDate) = 1
and W1.temperature > W2.temperature

-- 511 找每个用户登录时间最早的登录记录
select player_id , min(event_date) as first_login  from Activity
group by player_id

