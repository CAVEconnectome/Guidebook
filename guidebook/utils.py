from caveclient.tools.caching import CachedClient
from caveclient import CAVEclient
import flask


def make_client(datastack, server_address, **kwargs):
    """Build a framework client with appropriate auth token

    Parameters
    ----------
    datastack : str
        Datastack name for client
    config : dict
        Config dict for settings such as server address.
    server_address : str, optional
        Global server address for the client, by default None. If None, uses the config dict.

    """
    try:
        auth_token = flask.g.get("auth_token", None)
    except:
        auth_token = None

    client = CachedClient(
        datastack,
        server_address=server_address,
        auth_token=auth_token,
        write_server_cache=False,
        **kwargs,
    )
    return client

def make_global_client(server_address, **kwargs):
    try:
        auth_token = flask.g.get("auth_token", None)
        if auth_token == "AUTH_DISABLED":
            auth_token = None
    except:
        auth_token = None
    
    client = CAVEclient(
        datastack_name=None,
        server_address=server_address,
        auth_token=auth_token,
        global_only=True,
        write_server_cache=False,
        **kwargs,
    )
    return client
