import os


class BaseConfig(object):
    SECRET_KEY = os.environ.get('GUIDEBOOK_CSRF_KEY', 'test_key_1293473')
