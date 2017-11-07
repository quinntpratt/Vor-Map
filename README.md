# Vor-Map

This program was developed to satisfy project requirements for MATH 380 - 'Computational and Discrete Geoemtry' at the University of San Diego, San Diego, CA.

We use mapping tools in python/matplotlib to display a Voronoi Diagram of a geographical data.

This progarm has two (really 3) modes of operation:

0. Google Maps Places API to create an 'interactive' map in MatplotLib
1. Google Maps Places API to create a static map in MatplotLib
2. Using pre-downloaded datasets form a GIS server (in shapefile form)

When first starting the program be sure to acquire your own Google Maps API Key for the Places API. Furthermore, you may wish to change the key-words and pre-loaded coordinates to ones of your own choosing. 

In 'mode 2' a path will need to be specified to a shapefile of points on your local computer.

Furthermore, the program will output shapefiles to a directory called 'Output' where the program is executed. This output directory will contain shapefiles for the Voronoi Edges, Voronoi verticies, and the bounding box of the map-region.

Using a GIS program (such as the open-source QGIS) one can easily crop the Voronoi edges to the bounding box and then merge them into a single shapefile, and then use that shapefile for laser-cutting.

An example dataset has been given for Gas Stations in the San Diego Area. We have also inclueded a gif of the interactive mode, as well as an image of the result for half-of the gas stations in the included dataset.
