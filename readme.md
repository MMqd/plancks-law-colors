# Generating Colors of Hot Objects Using Planck's Law For Realtime Graphics
This project is a demo of how to generate and use colors of hot objects in realtime for games. It includes a Python script that generates a temperature-to-color gradient using Planck's Law and CIE XYZ color matching functions.

This is the repository for the following video from the MMqd channel.

[![A stylized vertical column of glowing cube-like particles transitioning from deep red at the top to bright yellow at the bottom, set against a soft sky background. Large text reading "Planck's Law" appears on the right, with a red arrow pointing toward the brightest part of the flame simulation.](https://img.youtube.com/vi/SyNikTi0Zm0/0.jpg)](http://www.youtube.com/watch?v=SyNikTi0Zm0 "Simulating Physically Accurate Colors of Fire")

## Key Features

* **Physically-based color:** Uses Planck's Law to compute radiance across the visible spectrum and converts the spectrum to sRGB colors using CIE XYZ color matching functions.
  
* **Exponential Temperature Mapping:** Maps a 0.0 - 1.0 input to a temperature range (e.g. 600 K to 60,000 K) using an exponential mapping function for a more uniform color distribution.
  
  ![A horizontal gradient that shifts from black through red and yellow to white, with white dominating most of the image.](extra/color_temperature.png)

  **DO NOT USE THIS IMAGE** it is for illustrative purposes only. Use the EXR texture in the project files. This images clips all the bright colors, and is tonemapped to SDR (the white parts are actually bluish-white), the EXR shows all the brightnesses and is not tonemapped. This image is also scaled up.

* **EXR Texture Output:** The Python script outputs an EXR texture that can be easily sampled from in any game or rendering engine.
  
* **Example Godot Integration:** A sample Godot project is provided that demonstrates how to use the temperature gradient in a shader and accurately apply temperature to cooling objects using Newton's Law of Cooling.

* **Provided gradient textures (Click to Download):**
    * [600K-60,000K, 256px](https://github.com/MMqd/plancks-law-colors/raw/main/color_temp_600_60000.exr)
    * [800K-6,000K, 128px](https://github.com/MMqd/plancks-law-colors/raw/main/extra/color_temp_800_6000.exr)
    * [600K-15,000K, 256px](https://github.com/MMqd/plancks-law-colors/raw/main/extra/color_temp_600_15000.exr)


## Workflow
1. **Reading Color Matching Functions:**  
   The process starts by reading the CIE XYZ color matching functions from a CSV file. These functions define how the human eye perceives color across different wavelengths.

2. **Computing Spectral Radiance:**  
   Using Planck's Law, the script calculates the spectral radiance (i.e., the energy emitted per wavelength) for a given temperature across the visible spectrum.

3. **Converting Spectrum to XYZ:**  
   The spectral radiance is integrated with the CIE XYZ color matching functions, converting the energy distribution into CIE XYZ color space values.

4. **XYZ to Linear sRGB Conversion:**  
   The obtained XYZ values are transformed into linear sRGB values, representing the physically accurate brightness and color.

5. **Mapping Temperature to Color:**  
   A normalized input value (from 0.0 to 1.0) is mapped to a temperature range using an exponential mapping function. Note that while this mapping function works well for clarity, a better mapping might exist, and further optimizations could be made. This approach was chosen to make the process easier to understand.
  
6. **Generating the EXR Texture:**  
   The Python script generates a 1D texture (saved as an EXR file) encoding the RGB values corresponding to different temperatures. This texture acts as a lookup table that can be sampled in realtime within game engines.

7. **Godot Shader Integration:**  
   Finally, the script outputs shader functions for Godot, which demonstrate how to sample the EXR texture—mapping a given temperature to the correct RGB color to be applied to objects in a scene.

## How To Run
The Godot project uses Godot 4.4, so Godot 4.4+ is recommended. The python script is in the `extra` folder. The Python script requires installing the dependencies:
```
pip install -r requirements.txt
```

Run the Python script as follows:
```
python plancks_law.py \
    --cie_csv CIE_xyz_1931_2deg.csv \
    --output_exr color_temperature.exr \
    --width 256 \
    --height 1 \
    --min_temp 600 \
    --max_temp 60000
```

**Basic Arguments:**
- `--help`: for a full list of arguments
- `--width`: Number of temperature samples.
- `--height`: Height of the output texture (usually 1 for a 1D lookup).
- `--min_temp`, `--max_temp`: Minimum and maximum temperatures in Kelvin.

## Additional Notes
* To use in your own projects, make sure to turn off mipmaps and anisotropy for the 1D temperature gradient texture. Mipmaps and anisotropy are designed for blending textures at different distances and angles, but since the temperature gradient is used for a single color lookup, downsampling may introduce imprecision.
* It is recommended to use an emission intensity multiplier of `0.1` in Godot. While the specific emission units in Godot shaders are not clearly documented, this value appears to produce visually accurate results when compared with reference images.
* **Mapping and Optimization:**  
  The current exponential mapping function is straightforward and easy to understand. However, a better mapping likely exists, and more optimizations can be made to improve precision or performance.

## License
* Content from Godot is under MIT
* The background in the video is `sunflowers_puresky_4k.exr` from [PolyHaven](https://polyhaven.com/a/sunflowers_puresky) under CC0, it is not included in the project, since it is too big.
* All other changes and content are under CC0 (Python script, gradient texture, new functions added to default Godot shader code).
