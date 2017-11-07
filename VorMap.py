#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:22:03 2017
@author: Quinn T. Pratt
University of San Diego - Department of Mathematics

Before attemting to run the program in mode 0 or 1, i.e. the modes of operation which require the Google Maps Places API, one should be sure to acquire their own API key from the following location:
https://developers.google.com/places/

Furthermore, one should be sure to familiarize oneself with ArcGIS.

REMEMBER: LON = X ; LAT = Y
"""

# needed to make maps. Specifically version 1.0.7 or Basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt 
# needed to pull from API
from urllib.request import urlopen
# needed to manipulate JavaScript Object Notation
import json
# Needed to do math
import numpy as np
import scipy.spatial as sp
# Needed for reading/writing shapefiles
import shapefile as shp

def plotMap(lon_0,lat_0):
    # note, you need change the EPSG for different region, 
    # You have to google the region you want every time.
    # 2771 is nad83(harn) / california zone 6
    # 3499 is nad83(nsrs2007) / california zone 6
    # 4326 is the US in general.
    epsg_1 = 4326
    
    # Different ArcGIS Services:
    
    #service = 'ESRI_Imagery_World_2D'
    #service = 'World_Topo_Map'
    service = 'World_Street_Map'
    #service = 'World_Physical_Map'
    #service = 'World_Imagery'
    
    xpixels = 4000
    scale = 0.205
    
    llc_lon=lon_0-scale
    llc_lat=lat_0-scale
    urc_lon=lon_0+scale
    urc_lat=lat_0+scale
    bbox = [llc_lon,llc_lat,urc_lon,urc_lat]
    
    m = Basemap(llcrnrlon=lon_0-scale ,llcrnrlat=lat_0-scale,
        urcrnrlon=lon_0+scale,urcrnrlat=lat_0+scale, ellps='WGS84',
        resolution= 'l', epsg = epsg_1)
    
    print('Bounding Box of Map: \n'
          'LL_lon: {0}, LL_lat: {1} \n'
          'UR_lon: {2}, UR_lat: {3} \n'.format(*bbox))
    
    # xpixels controls the pixels in x direction, and if you leave ypixels
    # None, it will choose ypixels based on the aspect ratio
    
    try:
        serv='http://server.arcgisonline.com/ArcGIS'
        m.arcgisimage(server=serv,service=service, xpixels = xpixels,ypixels=xpixels, verbose= True)
        m.readshapefile('./SanDiego_MajorRoads/SanDiego_MajorRoads', 'MajorRoads', 
                        drawbounds = True, color='black',linewidth=1)
        print('Map Generated Successfully')
    except:
        print('Map Generation Failed - Backup Map')
        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color='coral')
        m.readshapefile('./SanDiego_MajorRoads/SanDiego_MajorRoads', 'MajorRoads', 
                                      drawbounds = True, color='black',linewidth=1)
    
    m.drawmapboundary() # draw a line around the map region
    return m,bbox
    
    
def nearbySearch(lon,lat,radius,type_place,keyword,placesAPIkey):
    
    lat = str(lat)
    lon = str(lon)
    rad = str(radius)
    
    f = urlopen('https://maps.googleapis.com/maps/api/' \
                'place/nearbysearch/json?key=' + placesAPIkey \
                + '&location=' + lat + ',' + lon \
                + '&radius=' + rad \
                + '&type=' + type_place \
                + '&keyword=' + keyword)
    
    json_response = f.read().decode('utf-8')
    f.close
     
    return json_response
    
def parseJSON(parsed_json):
    
    results = parsed_json['results']
    names = [None] * len(results)
    locations = np.zeros((len(results),2))
    
    for i in range(len(results)):
        r = results[i]    
        lon,lat = [r['geometry']['location']['lat'],r['geometry']['location']['lng']]
        
        names[i] = r['name']
        locations[i,:] = [lat,lon]
    return locations,names
    
def vorPlot(m,vor):
    m.plot(vor.vertices[:,0], vor.vertices[:,1], 'g.',markersize=18)

    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            m.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-',lw=2.5)

    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100000
            m.plot([vor.vertices[i,0], far_point[0]], [vor.vertices[i,1], far_point[1]], 
                   'k--',lw=2)
  
def fmt_names(name):
    return 'Location Name: ' + name
  
class FollowDotCursor(object):
    """Display the x,y location of the nearest data point.
    https://stackoverflow.com/a/4674445/190597 (Joe Kington)
    https://stackoverflow.com/a/13306887/190597 (unutbu)
    https://stackoverflow.com/a/15454427/190597 (unutbu)
    """
    def __init__(self, ax, x, y, names, tolerance=5, formatter=fmt_names, offsets=(-50, 50)):
        try:
            x = np.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = np.asarray(mdates.date2num(x), dtype='float')
        y = np.asarray(y, dtype='float')
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        self._index = 0
        self.names = names
        self.offsets = offsets
        y = y[np.abs(y-y.mean()) <= 3*y.std()]
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = sp.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.names[self._index])
        bbox = self.annotation.get_window_extent()
        self.fig.canvas.blit(bbox)
        self.fig.canvas.draw_idle()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='white', alpha=0.75),
            arrowprops = dict(
                width=10,headwidth=20,headlength=15, connectionstyle='arc3,rad=-0.2',fc='w'),
                    fontsize=25)
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            self._index = idx
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]


def shapefileData(shapefile):
    locations = np.zeros((shapefile.numRecords,2))
    for i in range(shapefile.numRecords):
        shape = shapefile.shape(i)
        latlon = shape.points[0]
        locations[i,0] =  latlon[0]
        locations[i,1] = latlon[1]
    return locations

def makeShapeFile_Points(m,vor_verts):
    # first convert the voronoi verticies to lon-lat:
    lat_lon = m(vor_verts[:,0],vor_verts[:,1],inverse=True)
    lat_lon = np.transpose(np.array(lat_lon))
    # now that they're in lon-lat we can write them to a multi-point shape file.
    w = shp.Writer(shapeType=shp.POINT)
    w.autoBalance = 1
    
    w.field('location')
    #w.field('Y','F',len(lat_lon),8)
    
    for i in range(len(lat_lon)):
        lon = lat_lon[i,0]
        lat = lat_lon[i,1]
        w.point(lon,lat)
        w.record(str(i), 'Point')
        
    w.save('./Output/VoronoiVerts')
    #np.savetxt("VoronoiVertices.csv", lat_lon, delimiter=",")
    
    return lat_lon

def makeShapeFile_Lines(m,vor):

    w = shp.Writer(shapeType=shp.POLYLINE)
    w.autoBalance = 1
    w.field('location')
    line = []
    #save the interconnected lines one region at a time:
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        line_parts = []
        if np.all(simplex >= 0):
            # converts voronoi line connected-ness to lat-lon notation
            lat_lon = m(vor.vertices[simplex,0], vor.vertices[simplex,1],inverse=True)
            locations = [[p[0],p[1]] for p in lat_lon]
            locations = np.array(locations)
            longitude = locations[0,:]
            latitude = locations[1,:]
            line_parts.append([float(longitude[0]),float(latitude[0])])
            line_parts.append([float(longitude[1]),float(latitude[1])])
            line.append(line_parts)
            
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        line_parts = []
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100000
            
            vertex = m(vor.vertices[i,0],vor.vertices[i,1],inverse=True)
            farpoint = m(far_point[0],far_point[1],inverse=True)
            
            line_parts.append([float(vertex[0]),float(vertex[1])])
            line_parts.append([float(farpoint[0]),float(farpoint[1])])
            line.append(line_parts)
            
    w.line(parts=line)
    w.field('Closed_Simplex')
    w.record('line','ClosedSimplex')
    w.save('./Output/VoronoiEdges')      
    return line

def makeShapeFile_BBox(m,bbox):
    
    w = shp.Writer(shapeType=shp.POLYLINE)
    w.autoBalance = 1
    w.field('bounding_box')
    line = []
    line_parts = []
    LL = [bbox[0],bbox[1]]
    LR = [bbox[2],bbox[1]]
    UR = [bbox[2],bbox[3]]
    UL = [bbox[0],bbox[3]]
    bbox = np.array([LL,LR,UR,UL,LL])
    for i in range(len(bbox)):
        line_parts.append([float(bbox[i,0]), float(bbox[i,1])])
    line.append(line_parts)
    w.line(parts=line)
    w.record('Box','ClosedSimplex')
    w.save('./Output/BoundingBox')    

    return line
    
   

'''
# This is the commercial center of Pacific Beach, San Diego:
lat_center = 32.801199
lon_center = -117.238430

# This is in 805/8 interchange in San Diego
lat_center = 32.771564
lon_center = -117.132270

# this is Balboa Park, San Diego
lat_center = 32.732265
lon_center = -117.149220

# University of San Diego
lat_center = 32.771195
lon_center = -117.189826
'''
# Oak Park, San Diego
lat_center = 32.737500
lon_center = -117.076700
'''
# this is Ocean Beach, San Diego
lat_center = 32.744742
lon_center = -117.247575

# this is central LA
lat_center = 34.039389
lon_center = -118.248869
'''

'''
This program features three modes of operation
0. Google Maps
1. Google Maps - Interactive Map
2. Manual GIS
'''

mode = 2

if mode == 0 or mode == 1:
    '''
    If we're using the google maps API we'll need to define the radius about which
    to search, as well as the keywords for the search.
    '''
    print(' - - - - - Google Maps Mode - - - - - ')
    
    radius = 1600 # in kilometers
    type_place = 'cafe'
    keyword = 'coffee'
    
    placesAPIkey = 'YOUR API KEY HERE'
    # Now we call the nearby-search features from google maps.
    json_response = nearbySearch(lon_center,lat_center,radius,
                                 type_place,keyword,placesAPIkey)
    parsed_json = json.loads(json_response)
    locations, names = parseJSON(parsed_json)
    
elif mode == 2:
    '''
    If we're performing the manual mode of this program we'll need to specify the 
    locations we want to plot by reading in a shape-file.
    '''
    print(' - - - - - Manual GIS Mode - - - - - ')
    try:
        path2shpfile = '~/MyShapeFile.shp'
        shape_import =  shp.Reader('./SanDiego_GasStations/SanDiego_GasStations.shp')
    except:
        print('--!-- You must give a valid path to shapefile data (line 383) --!--')
    locations = shapefileData(shape_import)
    type_place = 'Manual GIS Data'
    
else:
    print('Please Define Mode for VorMap Operation')
    
    
'''
Here we create the basemap, if we can get the data from the ArcGIS server, then 
great, but if we cannot we'll build a rudimentary map using the major roads, 
and a more basic map.
'''
m, bbox = plotMap(lon_center,lat_center)


'''
Now that we've acquired our map and our locations-of-interest, we will plot.
'''

# first we plot the center of our search - for reference purposes.
x,y = m(lon_center, lat_center)
m.plot(x, y, 'b*', markersize=20,label='Center of Search')  
x,y = m(locations[:,0],locations[:,1])

# it is sometimes desireable to down-sample the data returned... here we downsample 
# by a factor of 2.
if mode == 2:
    x = x[::2]
    y = y[::2]

m.plot(x,y,'r .',markersize=18)

# Then we simply compute the voronoi diagram using scipy and superimpose that.
points = np.transpose(np.array([x,y]))
vor = sp.Voronoi(points)
vorPlot(m,vor)

if mode == 1:
    cursor = FollowDotCursor(plt.gca(), x, y, names, tolerance=20)
    
elif mode == 2:
    
    lat_lon = makeShapeFile_Points(m,vor.vertices)
    line = makeShapeFile_Lines(m,vor)
    box = makeShapeFile_BBox(m,bbox)


plt.title('Voronoi Map - Results for: ' + type_place,fontsize=26)
plt.show()

