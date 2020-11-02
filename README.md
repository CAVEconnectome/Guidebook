# Guidebook

Tools to guide attention proofreading neurons.

## Super early testing instructions

1. Make a new conda environment from a terminal window.

```:bash
conda create --name guidebook_test python==3.7
```

2. Activate the new environment.

```:bash
conda activate guidebook_test
```

3. Clone the `guidebook` to your computer. In your terminal, find a good working directory (e.g. `~/Work`) and enter:

```:bash
git clone https://github.com/ceesem/guidebook.git
```

4. Go into the new guidebook directory and install it with pip. The develop flag means that it will use the code in the directory every time it is imported, so new changes can be incorporated with a simple `git pull`.

```:bash
cd guidebook
pip install -e .
```

**NOTE**: If you have not set up your computer for programmatic access to the annotation framework, get your computer set up by following the instructions at the [AnnotationFrameworkClient documentation](https://annotationframeworkclient.readthedocs.io/en/latest/guide/authentication.html)

5. Start the server by running:  

```:bash
python run_once.py
```

After several seconds, a new chrome window will open. There are three links. “Shutdown” will shutdown the flask server, which you can also do by going to the terminal and hitting control-C. “Skeletonize” is slow and involves downloading the full mesh. It might be okay for small cells, however., and currently has more features relating to root nodes, but that is not a fundamental constraint. “Fast Skeletonize” is the chunked-graph first approach and will be the fastest option for most cases. Click that link and enter a root id then hit skeletonize. You should a new tab open and after several seconds (maybe up to a minute) you’ll see a new neuroglancer link with the branch points for that neuron as an annotation layer.
