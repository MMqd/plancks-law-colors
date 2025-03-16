#!/usr/bin/python
import os
# Enable OpenEXR support in OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import argparse
import csv
import math
import numpy as np
from scipy.constants import h, c, k
import cv2

# Constants from scipy.constants
# h: Planck's constant (J·s)
# c: Speed of light (m/s)
# k: Boltzmann's constant (J/K)

def read_cie_xyz(csv_filename):
    """
    Read CIE XYZ color matching functions from a CSV file without headers.
    The CSV should have columns in the following order:
    wavelength (nm), x_bar, y_bar, z_bar

    Args:
        csv_filename (str): Path to the CSV file.

    Returns:
        tuple: Tuple containing numpy arrays for wavelengths (m), x_bar, y_bar, z_bar.
    """
    wavelengths = []
    x_bar = []
    y_bar = []
    z_bar = []
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row_num, row in enumerate(reader, start=1):
            if len(row) < 4:
                print(f"Warning: Row {row_num} has fewer than 4 columns. Skipping.")
                continue
            try:
                wavelength_nm = float(row[0])
                wavelengths.append(wavelength_nm * 1e-9)  # Convert nm to meters
                x_bar.append(float(row[1]))
                y_bar.append(float(row[2]))
                z_bar.append(float(row[3]))
            except ValueError as e:
                print(f"Error parsing row {row_num}: {e}. Skipping.")
                continue
    return (np.array(wavelengths),
            np.array(x_bar),
            np.array(y_bar),
            np.array(z_bar))

def planck_law(wavelength, temperature):
    """
    Calculate spectral radiance using Planck's Law.
    wavelength: in meters (numpy array)
    temperature: in Kelvin (float)
    Returns spectral radiance in W·sr⁻¹·m⁻³
    """
    # Avoid division by zero or negative temperatures
    temperature = max(temperature, 1.0)
    exponent = (h * c) / (wavelength * k * temperature)
    # To handle large exponents that cause overflow in exp
    with np.errstate(over='ignore', invalid='ignore'):
        spectral_radiance = (2.0 * h * c**2) / (wavelength**5) / (np.exp(exponent) - 1.0)
    # Replace infinities and NaNs with zero (radiance negligible)
    spectral_radiance = np.where(np.isfinite(spectral_radiance), spectral_radiance, 0.0)
    spectral_radiance = np.where(spectral_radiance > 0, spectral_radiance, 0.0)
    return spectral_radiance

def spectrum_to_xyz(spectral_radiance, x_bar, y_bar, z_bar, delta_lambda):
    """
    Convert spectral radiance to CIE XYZ using color matching functions.
    spectral_radiance: array of radiance values corresponding to wavelengths
    x_bar, y_bar, z_bar: CIE color matching functions
    delta_lambda: wavelength interval (m)
    Returns XYZ tuple
    """
    # Wavelength intervals must be consistent
    X = np.sum(spectral_radiance * x_bar) * delta_lambda
    Y = np.sum(spectral_radiance * y_bar) * delta_lambda
    Z = np.sum(spectral_radiance * z_bar) * delta_lambda
    return np.array([X, Y, Z])

def xyz_to_linear_rgb(xyz):
    """
    Convert CIE XYZ to linear RGB.
    Uses sRGB color space and D65 illuminant.
    """
    # Transformation matrix from CIE XYZ to sRGB
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    rgb = np.dot(M, xyz)
    return rgb

def temperature_to_rgb(wavelengths, x_bar, y_bar, z_bar, temperature):
    """
    Convert a given temperature in Kelvin to linear RGB using Planck's Law.
    wavelengths: array of wavelengths in meters
    x_bar, y_bar, z_bar: CIE color matching functions
    temperature: in Kelvin
    Returns RGB tuple with physically accurate brightness
    """
    spectral_radiance = planck_law(wavelengths, temperature)
    delta_lambda = wavelengths[1] - wavelengths[0]  # Assume uniform sampling
    xyz = spectrum_to_xyz(spectral_radiance, x_bar, y_bar, z_bar, delta_lambda)
    rgb = xyz_to_linear_rgb(xyz)
    # No normalization; raw RGB values represent physical brightness
    # Clamp negative values to zero
    rgb = np.maximum(rgb, 0)
    return rgb

def mapping_function(x, min_temp=600.0, max_temp=15000.0, use_linear=False):
    """
    Map a normalized input [0, 1] to temperature [min_temp, max_temp].

    By default, uses an exponential mapping:
        temperature = min_temp * (max_temp / min_temp) ** x
    If use_linear is True, uses linear interpolation:
        temperature = min_temp + x * (max_temp - min_temp)

    Args:
        x (float): Normalized input [0, 1].
        min_temp (float): Minimum temperature in Kelvin.
        max_temp (float): Maximum temperature in Kelvin.
        use_linear (bool): Flag to select linear mapping.

    Returns:
        float: Mapped temperature in Kelvin.
    """
    if x <= 0.0:
        return min_temp
    elif x >= 1.0:
        return max_temp
    else:
        if use_linear:
            temperature = min_temp + x * (max_temp - min_temp)
        else:
            temperature = min_temp * (max_temp / min_temp) ** x
        return temperature

def generate_temperature_texture(wavelengths, x_bar, y_bar, z_bar, width=1024, height=1,
                                 max_temp=15000.0, min_temp=600.0, mapping_linear=False):
    """
    Generate an EXR texture mapping temperature to RGB.
    width: number of temperature samples (e.g., 1024 for default)
    height: texture height (the 1D gradient is repeated across all rows)
    max_temp: maximum temperature in Kelvin
    min_temp: minimum temperature in Kelvin
    mapping_linear: use linear mapping if True; otherwise use exponential mapping.
    Returns R, G, B arrays.
    """
    # Generate gradient for one row
    R_row = np.zeros((width,), dtype=np.float32)
    G_row = np.zeros((width,), dtype=np.float32)
    B_row = np.zeros((width,), dtype=np.float32)

    for i in range(width):
        # Normalize x to [0, 1]
        x = i / (width - 1)
        # Apply mapping function to get the temperature
        temperature = mapping_function(x, min_temp, max_temp, use_linear=mapping_linear)
        rgb = temperature_to_rgb(wavelengths, x_bar, y_bar, z_bar, temperature)
        R_row[i] = rgb[0]
        G_row[i] = rgb[1]
        B_row[i] = rgb[2]

        if (i+1) % 100 == 0 or i == width-1:
            print(f"Processed {i+1}/{width} temperatures.")

    # Replicate the computed gradient across all height rows
    R = np.tile(R_row, (height, 1))
    G = np.tile(G_row, (height, 1))
    B = np.tile(B_row, (height, 1))

    return R, G, B

def save_exr(filename, R, G, B):
    """
    Save the RGB channels to an EXR file using OpenCV.
    """
    # Stack channels to form an image with shape (height, width, 3)
    exr_image = np.dstack((R, G, B))
    # OpenCV expects the channels in BGR order by default, so reorder if necessary
    exr_image = exr_image[:, :, ::-1]  # Convert RGB to BGR
    # Write the EXR file
    success = cv2.imwrite(filename, exr_image)
    if success:
        print(f"EXR file saved as '{filename}'")
    else:
        print(f"Failed to save EXR file as '{filename}'")

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a color temperature EXR texture using Planck's Law with selectable mapping.")
    parser.add_argument('--cie_csv', type=str, default='CIE_xyz_1931_2deg.csv',
                        help='Path to the CIE XYZ CSV file without headers.')
    parser.add_argument('--output_exr', type=str, default='color_temperature.exr',
                        help='Output EXR filename.')
    parser.add_argument('--width', type=int, default=256,
                        help='Width of the generated texture (number of temperature samples).')
    parser.add_argument('--height', type=int, default=1,
                        help='Height of the generated texture.')
    parser.add_argument('--max_temp', type=float, default=60000.0,
                        help='Maximum temperature in Kelvin.')
    parser.add_argument('--min_temp', type=float, default=600.0,
                        help='Minimum temperature in Kelvin.')
    parser.add_argument('--linear_mapping', action='store_true',
                        help='Use linear mapping instead of exponential (logarithmic) mapping.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Check if CIE CSV exists
    if not os.path.isfile(args.cie_csv):
        print(f"Error: CIE XYZ CSV file '{args.cie_csv}' not found.")
        return

    # Read CIE XYZ data from CSV
    print(f"Reading CIE XYZ data from '{args.cie_csv}'...")
    wavelengths, x_bar, y_bar, z_bar = read_cie_xyz(args.cie_csv)
    print("CIE XYZ data loaded.")

    # Generate temperature texture
    print("Generating temperature texture...")
    R, G, B = generate_temperature_texture(
        wavelengths, x_bar, y_bar, z_bar,
        width=args.width,
        height=args.height,
        max_temp=args.max_temp,
        min_temp=args.min_temp,
        mapping_linear=args.linear_mapping
    )

    # Save EXR file
    print(f"Saving EXR texture as '{args.output_exr}'...")
    save_exr(args.output_exr, R, G, B)

if __name__ == "__main__":
    main()
