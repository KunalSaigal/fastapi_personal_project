import numpy as np
import pandas as pd

# 1. CREATION & SHAPE
# Let's simulate 3 images, each 2x2 pixels.
# Pixel values range from 0 (black) to 255 (white).
images = np.array(
    [
        [[255, 100], [50, 0]],  # Image 0
        [[120, 130], [140, 150]],  # Image 1
        [[0, 255], [255, 0]],  # Image 2
    ]
)

print("Original Shape:", images.shape)
# Output: (3, 2, 2) -> (Batch Size, Height, Width)


# 2. RESHAPING
# Most traditional ML algorithms (like Scikit-Learn's Logistic Regression)
# expect a 2D matrix: (Number of samples, Number of features).
# We need to "flatten" each 2x2 image into a 4-pixel row.

flattened_images = images.reshape(3, 4)
print("\nFlattened Shape:", flattened_images.shape)
# Output: (3, 4) -> 3 images, each has 4 features (pixels)
print("Flattened Data:\n", flattened_images)


# 3. SLICING
# Syntax: array[batch, row, column] or array[sample, feature]

# Example A: Get the entire second image (index 1) from the original data
second_image = images[1]
# or  images[1, :, :] which is - [2nd index, all rows , all columns]
print("\nSecond Image:\n", second_image)

# Example B: Get the first two features (pixels) for all images in the flattened data
first_two_pixels = flattened_images[:, 0:2]
print("\nFirst two pixels of all images:\n", first_two_pixels)


# 4. BROADCASTING
# Neural networks perform much better when inputs are scaled between 0 and 1.
# Instead of looping through every pixel, we can divide the entire array by 255.
# NumPy "broadcasts" the single scalar (255) across every element in the array.

normalized_images = flattened_images / 255.0
print("\nNormalized Data (Broadcasting):\n", normalized_images)


# 5. PANDAS COMPARISON
# Let's compare NumPy with Pandas for a simple task.
# We'll create a DataFrame with image data and calculate the mean pixel value for each image.

# Create a DataFrame from our NumPy array
df = pd.DataFrame(flattened_images, columns=["Pixel1", "Pixel2", "Pixel3", "Pixel4"])
print("\nDataFrame:\n", df)

# Calculate the mean pixel value for each image
mean_pixel_values = df.mean(axis=1)
print("\nMean Pixel Values:\n", mean_pixel_values)

# Filter rows where Pixel2 is equal to 130
corrupt_data = df[df["Pixel2"] == 130]

print("\n--- Corrupt Rows Detected ---")
print(corrupt_data)

# Calculate the average value across columns (axis=1) for each row
df["Average_Brightness"] = df[["Pixel1", "Pixel2", "Pixel3", "Pixel4"]].mean(axis=1)

# Assign a label: If Average_Brightness > 100, label it 'Bright' (1), else 'Dark' (0)
df["Label"] = np.where(df["Average_Brightness"] > 100, "Bright", "Dark")

print("\n--- Processed DataFrame with Labels ---")
print(df)
