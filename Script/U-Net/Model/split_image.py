import rasterio
import numpy as np
import os
from rasterio.windows import Window
from rasterio.windows import transform

def split(image_path, window_size, stride, outfolder):
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        out_path = outfolder

        for y in range(0, height - window_size, stride):
            for x in range(0, width - window_size, stride):
                window2 = Window(x, y, window_size, window_size)
                transform2 = transform(window2, src.transform)
                image_data = src.read(window=window2)
                print(image_data.shape)
                out_meta = src.meta.copy()
                out_meta.update({'driver':'GTiff', 'width':image_data.shape[1], 'height':image_data.shape[1], 'count':3, 'crs': src.crs, 'transform':transform2})

                if out_meta is None:
                    print("Error: out_meta.update() returned None")
                else:      
                     with rasterio.open(os.path.join(out_path, "crop_{}_{}.tif".format(y,x)), 'w', **out_meta) as output:
                        output.write(image_data)
                        output.close()
    src.close()
                
    



