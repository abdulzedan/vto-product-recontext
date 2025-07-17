import sys
import os
from dotenv import load_dotenv

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictResponse

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "main-env-demo")  # @param {type:"string"}
LOCATION = os.getenv("LOCATION", "us-central1")  # @param ["us-central1"]

aiplatform.init(project=PROJECT_ID, location=LOCATION)

api_regional_endpoint = f"{LOCATION}-aiplatform.googleapis.com"
client_options = {"api_endpoint": api_regional_endpoint}
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

model_endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/virtual-try-on-exp-05-31"
print(f"Prediction client initiated on project {PROJECT_ID} in {LOCATION}.")

# @title Import libraries and define utilities
# @markdown Run this cell before proceeding to import libraries and define utility functions.
import base64
import io
import re
import timeit

from PIL import Image
from google.cloud import storage
import argparse
import matplotlib.pyplot as plt


# Parses the generated image bytes from the response and converts it
# to a PIL Image object.
def prediction_to_pil_image(
    prediction: PredictResponse, size=(640, 640)
) -> Image.Image:
    encoded_bytes_string = prediction["bytesBase64Encoded"]
    decoded_image_bytes = base64.b64decode(encoded_bytes_string)
    image_pil = Image.open(io.BytesIO(decoded_image_bytes))
    image_pil.thumbnail(size)
    return image_pil


# Displays images and predictions in a horizontal row.
def display_row(items: list, figsize: tuple[int, int] = (15, 15)):
    count = len(items)

    if count == 0:
        print("No items to display.")
        return

    fig, ax = plt.subplots(1, count, figsize=figsize)
    if count == 1:
        axes = [ax]
    else:
        axes = ax

    for i in range(count):
        item = items[i]
        current_ax = axes[i]

        if isinstance(item, Image.Image):
            current_ax.imshow(item, None)
            current_ax.axis("off")
        elif "bytesBase64Encoded" in item:
            pil_image = prediction_to_pil_image(item)
            current_ax.imshow(pil_image, None)
            current_ax.axis("off")
        elif "raiFilteredReason" in item:
            rai_reason = item["raiFilteredReason"]
            current_ax.text(
                0.5,
                0.5,
                rai_reason,
                horizontalalignment="center",
                verticalalignment="center",
                transform=current_ax.transAxes,
                fontsize=12,
                wrap=True,
            )
            current_ax.set_xlim(0, 1)
            current_ax.set_ylim(0, 1)
            current_ax.axis("off")

    plt.tight_layout()
    plt.show()


# Download image bytes from a GCS URI.
def download_gcs_image_bytes(uri: str) -> bytes:
    matched = re.match(r"gs://(.*?)/(.*)", uri)

    if matched:
        bucket_name = matched.group(1)
        object_name = matched.group(2)
    else:
        raise ValueError(f"Invalid GCS URI format: {uri}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    return blob.download_as_bytes()


# Constructs a Vertex AI PredictRequest and uses it to call Virtual Try-On.


def call_virtual_try_on(
    person_image_bytes=None,
    product_image_bytes=None,
    person_image_uri=None,
    product_image_uri=None,
    sample_count: int = 1,
    base_steps=None,
    safety_setting=None,
    person_generation=None,
) -> PredictResponse:
    instances = []

    if person_image_uri and product_image_uri:
        instance = {
            "personImage": {"image": {"gcsUri": person_image_uri}},
            "productImages": [{"image": {"gcsUri": product_image_uri}}],
        }
        instances.append(instance)
    elif person_image_bytes and product_image_bytes:
        instance = {
            "personImage": {"image": {"bytesBase64Encoded": person_image_bytes}},
            "productImages": [{"image": {"bytesBase64Encoded": product_image_bytes}}],
        }
        instances.append(instance)
    else:
        raise ValueError(
            "Both person_image_bytes and product_image_bytes or both person_image_uri and product_image_uri must be set."
        )

    parameters = {"sampleCount": sample_count}

    if base_steps:
        parameters["baseSteps"] = base_steps

    if safety_setting:
        parameters["safetySetting"] = safety_setting

    if person_generation:
        parameters["personGeneration"] = person_generation

    start = timeit.default_timer()

    response = client.predict(
        endpoint=model_endpoint, instances=instances, parameters={}
    )
    end = timeit.default_timer()
    print(f"Virtual Try-On took {end - start:.2f}s.")

    return response



# Load person image from local file
def load_person_image(person_image_path):
    with open(person_image_path, 'rb') as f:
        RAW_PERSON_IMAGE_BYTES = f.read()
    
    # Process the image first - convert to RGB
    PERSON_IMAGE_PIL = Image.open(io.BytesIO(RAW_PERSON_IMAGE_BYTES)).convert("RGB")
    original_size = PERSON_IMAGE_PIL.size
    
    # COMMENTED OUT: Don't downscale person image to preserve original resolution
    # # Apply thumbnail to maintain aspect ratio within 1024x1024
    # PERSON_IMAGE_PIL.thumbnail((1024, 1024))
    # print(f"Person image size: {original_size} -> {PERSON_IMAGE_PIL.size}")
    
    print(f"Person image size: {PERSON_IMAGE_PIL.size}")
    
    # Encode the full resolution image
    buffer = io.BytesIO()
    PERSON_IMAGE_PIL.save(buffer, format='PNG')
    ENCODED_PERSON_IMAGE_BYTES = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return ENCODED_PERSON_IMAGE_BYTES, PERSON_IMAGE_PIL

# Load product image from local file
def load_product_image(product_image_path):
    with open(product_image_path, 'rb') as f:
        RAW_PRODUCT_IMAGE_BYTES = f.read()
    
    # Process the image first - convert to RGB and ensure max 1024x1024
    PRODUCT_IMAGE_PIL = Image.open(io.BytesIO(RAW_PRODUCT_IMAGE_BYTES)).convert("RGB")
    original_size = PRODUCT_IMAGE_PIL.size
    
    # Apply thumbnail to maintain aspect ratio within 1024x1024
    PRODUCT_IMAGE_PIL.thumbnail((1024, 1024))
    print(f"Product image size: {original_size} -> {PRODUCT_IMAGE_PIL.size}")
    
    # Encode the PROCESSED image, not the original
    buffer = io.BytesIO()
    PRODUCT_IMAGE_PIL.save(buffer, format='PNG')
    ENCODED_PRODUCT_IMAGE_BYTES = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return ENCODED_PRODUCT_IMAGE_BYTES, PRODUCT_IMAGE_PIL



def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On using Google Cloud AI')
    parser.add_argument('--person', required=True, help='Path to person image')
    parser.add_argument('--product', required=True, help='Path to product/clothing image')
    parser.add_argument('--output', default='output.png', help='Path to save output image (default: output.png)')
    parser.add_argument('--sample-count', type=int, default=1, help='Number of samples to generate (1-4)')
    parser.add_argument('--safety-setting', default='block_low_and_above', 
                       choices=['block_low_and_above', 'block_medium_and_above', 'block_only_high', 'block_none'],
                       help='Safety setting for content filtering')
    parser.add_argument('--person-generation', default='allow_adult',
                       choices=['dont_allow', 'allow_adult', 'allow_all'],
                       help='Person generation setting')
    
    args = parser.parse_args()
    
    # Load images
    print("Loading person image...")
    ENCODED_PERSON_IMAGE_BYTES, PERSON_IMAGE_PIL = load_person_image(args.person)
    
    print("Loading product image...")
    ENCODED_PRODUCT_IMAGE_BYTES, PRODUCT_IMAGE_PIL = load_product_image(args.product)
    
    # Show input images
    print("\nInput images:")
    display_row([PERSON_IMAGE_PIL, PRODUCT_IMAGE_PIL])
    
    # Call Virtual Try-On
    print("\nCalling Virtual Try-On API...")
    response = call_virtual_try_on(
        person_image_bytes=ENCODED_PERSON_IMAGE_BYTES,
        product_image_bytes=ENCODED_PRODUCT_IMAGE_BYTES,
        sample_count=args.sample_count,
        base_steps=None,
        safety_setting=args.safety_setting,
        person_generation=args.person_generation,
    )
    
    # Display and save results
    print("\nResults:")
    predictions = list(response.predictions)
    # Skip display for non-interactive mode
    # display_row(predictions)
    
    # Save the first result at original API response size
    if predictions and 'bytesBase64Encoded' in predictions[0]:
        # Don't thumbnail the saved image - keep original API response quality
        result_image = prediction_to_pil_image(predictions[0], size=(1024, 1024))
        result_image.save(args.output)
        print(f"\nOutput saved to: {args.output}")
    else:
        print("\nNo valid predictions returned.")

if __name__ == "__main__":
    main()
     


     

     
