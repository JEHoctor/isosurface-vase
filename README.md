# isosurface-vase
Use scalar fields to create surfaces for 3D printing in vase mode.

For now, this project generates a hard coded vase design.

![vase stl as of a recent commit](images/FreeCAD_review_578f0ef97baad504ec316ce757ef0d51548c1e57.png)

I scaled, stretched, and cut this model in my slicer to 3D print this:

![printed vase as of a recent commit](images/printed_578f0ef97baad504ec316ce757ef0d51548c1e57.jpg)

## Usage

To generate the vase and save it as `vase.stl`:

```
$ python isosurface_vase/vase.py --no-draft
```

Or, run the script in draft mode to see results much more quickly:

```
$ python isosurface_vase/vase.py
```

For more invocation options:

```
$ python isosurface_vase/vase.py --help
```