import os
import redis
from rq import Worker, Queue, Connection

listen = ['default']
REDIS_HOST = os.environ.get('REDIS_SERVICE_HOST','localhost')
REDIS_PORT = os.environ.get('REDIS_SERVICE_PORT',6379)
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD','')
if len(REDIS_PASSWORD)==0:
    redis_url = f'redis://{REDIS_HOST}:{REDIS_PORT}/0'
else:
    redis_url = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work(with_scheduler=True)
