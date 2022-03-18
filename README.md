# PiNN-D

## Environments:

The code works well with [python](https://www.python.org/) 3.8.8, 
[pytorch](https://pytorch.org/) 1.8.1, 
and **[osmnx](https://github.com/gboeing/osmnx)  0.16.1**.

Some scripts (for map matching) require cygwin and python 2.7

## Pipeline:
1. Data preprocessing:
   
   (a) [Download the osm graph data](https://github.com/yang-mingzhou/PINN-D/blob/main/code/downloadGraph.py): download the osm graph within the bounding box to the folder '/data/bbox'. We are using a bounding box: (min Longitude, max Longitude , min Latitude , max Latitude: -95.35305 -91.544 43.1759 45.891999999999996)  which contains 722629 edges and 277704 nodes.
      
   (b) [Trajectory generation](https://github.com/yang-mingzhou/PINN-D/blob/main/code/trajectoryGeneration.py): extract and sample (by every 3 seconds) the trajectories of vehicles inside the bounding box for map matching (3646 trajectories in total).      

   (c) [Map matching](https://github.com/yang-mingzhou/PINN-D/blob/main/code/mapmatching.py)

   (d) Feature extraction

2. Model definition

3. Training

4. Evaluation

[comment]: <> (## File Folders:)

[comment]: <> (1. )
   
   
[comment]: <> (## Files)

[comment]: <> (1. )

Change Log
-----

### 2022/3/15
Version 1.0 Define the pipeline in readme.md, scripts for trajectory generation, graph download and mapmatching

