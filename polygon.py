from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image_path, text_coordinates):
    """
    Adds text to an image at specified coordinates.

    Args:
        image_path: Path to the image file.
        text_coordinates: A list of tuples, where each tuple contains:
            - The text to be added (string).
            - The x-coordinate (integer).
            - The y-coordinate (integer).
    """

    try:
        img = Image.open(image_path).convert("RGB") # Ensure it's RGB for text compatibility
        draw = ImageDraw.Draw(img)

        # Choose a font (you might need to adjust the path if the font isn't in a standard location)
        try:
            font = ImageFont.truetype("arial.ttf", size=400)  # Try Arial first
        except IOError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=20) #linux
            except IOError:
                font = ImageFont.load_default() # Fallback to default font if Arial is not found.

        for text, x, y in text_coordinates:
            draw.text((x, y), text, fill=(0, 0, 0), font=font) # Black text

        # Save the modified image (or display it)
        new_image_path = "image_with_text.jpg"  # Or any other name/path you want.
        img.save(new_image_path)
        print(f"Image saved to {new_image_path}")
        #img.show() # to display the image directly

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
image_file = "people.jpg"  # **REPLACE with the actual path to your image file**
text_coords = [
    ("a", 1100, 10),
    ("b", 1300, 10),
    ("c", 1110, 910),
    ("d", 720, 910),
]

add_text_to_image(image_file, text_coords)