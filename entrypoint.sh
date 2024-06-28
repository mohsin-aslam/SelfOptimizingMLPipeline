#!/bin/sh
. /.env

TRY_LOOP="20"
AIRFLOW_HOME="/usr/local/airflow"
export AIRFLOW_HOME

wait_for_port() {
  local name="$1" host="$2" port="$3"
  local j=0
  while ! nc -z "$host" "$port" >/dev/null 2>&1 < /dev/null; do
    j=$((j+1))
    if [ $j -ge $TRY_LOOP ]; then
      echo >&2 "$(date) - $host:$port still not reachable, giving up"
      exit 1
    fi
    echo "$(date) - waiting for $name... $j/$TRY_LOOP"
    sleep 5
  done
}

AIRFLOW__CODE_EDITOR__STRING_NORMALIZATION=True
export AIRFLOW__CODE_EDITOR__STRING_NORMALIZATION

wait_for_port "Postgres" "$POSTGRES_HOST" "$POSTGRES_PORT"

wait_for_port "Redis" "$REDIS_HOST" "$REDIS_PORT"

case "$1" in
  webserver)
    airflow db migrate
    Check if user "airflow" already exists in the list of users
    if ! airflow users list | grep -q "airflow"; then
        # User doesn't exist, so create it
        airflow users create \
            --username "${AIRFLOW_USERNAME}" \
            --password "${AIRFLOW_PASSWORD}" \
            --firstname Airflow \
            --lastname Admin \
            --role Admin \
            --email m.mohsin.aslam@gmail.com
    else
        echo "User 'airflow' already exists. Skipping creation."
    fi

    exec airflow webserver
    ;;
  worker)
    sleep 10
    exec airflow celery worker
    ;;
  scheduler)
    sleep 10
    exec airflow scheduler
    ;;
  flower)
    sleep 10
    exec airflow celery flower
    ;;
  *)
    exec "$@"
    ;;
esac
