from airflow import DAG 

from airflow.operators.bash_operator import BashOperator 

from datetime import datetime, timedelta 

 

default_args = { 

    'owner': myname, 

} 

 

dag = DAG('test', default_args=default_args, schedule_interval=timedelta(days=1)) 

 

t1 = BashOperator( 

    task_id='print_date', 

    bash_command='date', 

    dag=dag) 

 

t2 = BashOperator( 

    task_id='sleep', 

    bash_command='sleep 5', 

    retries=3, 

    dag=dag) 

 

t2.set_upstream(t1) 