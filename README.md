# Guidebook

Tools to guide attention for proofreading neurons in a pychunkedgraph-backed segmentation.

## How to use

From the user's perspective, Guidebook takes a neuron and finds lists of points to look at.
At the moment, these are strictly topological points of interests: branch points and/or end points.
Because this structure is generated entirely from the current state of the segmentation, this can be completely dynamic and be run immediately after a proofreading event.

After submitting a neuron and waiting a short while (20-60 seconds, depending on size), you get back a collection of branch points, end points, or both.
An optional root point helps anchor the representation at a useful point, either the soma or perhaps the base of an axon.
The "root is soma" tag accounts for the fact that the soma is more like a large sphere than a linear neuronal process.
Branch points are grouped into collections of branches and ordered by distance from the root point.

### How it works

Guidebook has three parts:
    1) A Flask app that asks a user for a neuron root id
    2) A worker process that uses RQ to get jobs to do the neuron lookups
    3) A redis server to pass messages.

The docker-compose.yml file is configured to build a working application in docker.
You should be able to run `docker-compose up --build -d` from the base directory and get a working at at the `/guidebook/` endpoint.
For testing locally, you probably need to disable the `@authrequired` decorators on `guidebook/app/processing`.

For comments or questions, contact Casey Schneider-Mizell (caseys@alleninstitute.org) 