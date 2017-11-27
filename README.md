# blotchmap


blotchmap.py procedurally creates images of interlocking shapeless blobs that kind of resemble lines of a political map. These images can be saved to an image file (e.g. .png) or viewed in a window. Viewing in a window uses the Pygame package.

```
usage: blotchmap.py [-h] [-S S] [-D x y nnd] [-M M] [-J vdev ddev] [-I res]
                    [-O filename] [-V]
optional arguments:
  -h, --help            show this help message and exit
  -S S, --Seed S        Seed Number for rng.
  -D x y nnd, --Dim x y nnd
                        Dimensions of hexgrid. x and y are horizontal and
                        vertical number of hexes respectively. nnd is nearest
                        neighbour distance, which determines size of each hex.
  -M M, --Merge M       Merge hexes to have M cells remain.
  -J vdev ddev, --Jostle vdev ddev
                        Jostle the points that make up the hex grid. vdev and
                        ddev are standard deviation of the vertices in each
                        hex and the subdivisions of a side, respectively.
  -I res, --Interp res  Interpolate over subdivision using cubic spline. res
                        is resolution, number of points per hex side.
  -O filename, --Output filename
                        Save blotchmap as image to given filename. Use file
                        extensions .bmp, .tga, .png, or .jpeg.
  -V, --Visual          Show blotchmap as pygame drawing.

```

An image is created in a number of steps.
1. Build a grid of hexagons, defined by points with positions in 2-space, and lists of neighbours for connecting lines.
  * find the centers of the hexcells.
  * each hexcell gets six trinodes which represent the vertices.
  * trinodes usually have three neighbours each. (except for edges).
2. Merge some neighbouring hexecells together randomly, erasing the lines that divide them.
  * create `M` buckets for hexcells.
  * randomly pick `M` nodes and place one in each bucket.
  * other nodes get added to a bucket if they have a neighbour in it.
  * each bucket of nodes gets merged into one.
3. Trinodes get jostled around.
4. Subdivide trinode pairs
5. Subdivided points get jostled around a bit as well.
6. Interpolate using cubic spline to sort of smooth out the kinks.


Here's an example by adding each stage one at a time.

`python blotchmap.py -S 999 -D 12 8 60 -O bmapex1.png`
![example step 1](/example_images/bmapex1.png)
`python blotchmap.py -S 999 -D 12 8 60 -M 10 -O bmapex2.png`
![example step 2](/example_images/bmapex2.png)
`python blotchmap.py -S 999 -D 12 8 60 -M 10 -J 3 3 -O bmapex3.png`
![example step 3](/example_images/bmapex3.png)
`python blotchmap.py -S 999 -D 12 8 60 -M 10 -J 3 3 -I 100 -O bmapex4.png`
![example step 4](/example_images/bmapex4.png)


A mosaic tile, or jagged rock type pattern can be created by greatly turning down the interpolation resolution.

`python blotchmap.py -S 987654321 -D 30 20 25 -M 100 -J 5 0 -I 3 -O mosaicex.png`
![mosaic example](/example_images/mosaicex.png)
