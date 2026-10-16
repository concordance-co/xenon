"""Defense-in-depth sentinel; the OS sandbox also covers native child programs."""
import os
import sys

if os.environ.get('XENON_LOCAL_SCRATCH') == '1':
    def _audit(event, args):
        if event in {'socket.connect', 'socket.connect_ex', 'socket.bind', 'socket.getaddrinfo', 'socket.sendto'}:
            raise PermissionError('local-scratch forbids network access')
        if event == 'import' and args and str(args[0]).split('.')[0] in {'modal', 'psycopg', 'psycopg2', 'openai', 'anthropic'}:
            raise PermissionError('local-scratch forbids cloud/database/API clients')
    sys.addaudithook(_audit)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
