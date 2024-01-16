from cachetools import cached, LRUCache
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

@cached(cache=LRUCache(maxsize=512))
def check_datastack(datastack, server_address):
    """Check that the datastack has an L2 cache

    Parameters
    ----------
    datastack : str
        Datastack name to check
    server_address : str
        Global server address for the client

    Returns
    -------
    bool
        True if the datastack is valid for guidebook
    """
    cl = make_client(datastack, server_address)
    return cl.l2cache.has_cache()

def get_datastacks(client):
    """Check that the datastacks in the list are backed by graphene segmentatoins

    Parameters
    ----------
    datastacks : list
        List of datastack names to check
    client : CAVEclient
        CAVEclient object to check against

    Returns
    -------
    list
        List of datastacks that are valid for the client
    """
    valid_datastacks = []
    ds = client.info.get_datastacks()
    for datastack_name in ds:
        if check_datastack(datastack_name, client.server_address):    
            valid_datastacks.append(datastack_name)
    return valid_datastacks