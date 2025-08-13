# Comprehensive Guide: Bulk Image Processor

This document provides an exhaustive, in-depth analysis of the Bulk Image Processor application. It is intended for developers who need a deep understanding of the system's architecture, data flow, components, and the rationale behind its design.

## 1. Philosophy and High-Level Architecture

The Bulk Image Processor is engineered for **robustness, scalability, and observability**. It is not just a script but a resilient pipeline designed to handle large-scale, real-world image processing tasks where failures are expected and must be handled gracefully.

### Core Architectural Principles:

*   **Asynchronous First:** The entire application is built on Python's `asyncio` framework. This allows for high-throughput I/O operations (downloading images, making API calls) without the overhead of traditional multithreading, leading to significant performance gains.
*   **Stateful, Staged Pipeline:** The process is broken down into distinct, sequential stages (Download -> Classify -> Process). This makes the logic easier to follow, debug, and manage. The output of one stage becomes the input for the next.
*   **AI-Driven Quality Control:** The system uses a "human-in-the-loop" pattern, where the Gemini model acts as an AI agent to perform tasks that would typically require human judgment: classifying images, assessing the quality of generated content, and even providing creative direction.
*   **Resilience Through Intelligent Retries:** The application anticipates two types of failures: transient network/API errors and poor-quality AI generations. It uses a two-pronged retry strategy:
    1.  **API Retries (`tenacity`):** Simple, exponential-backoff retries for network-level failures.
    2.  **Logic-Based Model Retries:** A more sophisticated loop within the VTO processor that can detect a *bad result* (e.g., the garment wasn't applied) and try again with a completely different model, overcoming fundamental generation failures.
*   **Structured Configuration:** All settings are managed through a `.env` file and parsed by Pydantic, providing type safety, validation, and clear separation of configuration from code.
*   **Comprehensive, Structured Logging:** Using `structlog`, all log entries are produced in JSON format, making them machine-readable and easy to ingest, query, and visualize in modern logging platforms.

### System Flow Diagram

Here is a detailed visual representation of the application's workflow:

```
(Start)
   |
   V
[CLI: `__main__.py`]
   |  - Parse CLI args (--csv, --max-workers)
   |  - Override .env settings if needed
   |
   V
[Orchestrator: `main.BulkImageProcessor`]
   |  - Initialize all components (Analyzer, Processors, Storage)
   |  - Create Run Directory (e.g., output/2025/08/13/run_001/)
   |
   +------------------ STAGE 1: DOWNLOAD ------------------+
   |                                                       |
   |  [Downloader: `downloader.ImageDownloader`]           |
   |     | - Reads `img_conversion_table.csv`              |
   |     | - Creates 100 `ImageRecord` objects             |
   |     |                                                 |
   |     V                                                 |
   |  (Async Task Pool - `max_workers`)                    |
   |     - Downloads 100 images concurrently               |
   |     - Handles retries for network errors              |
   |     - Validates and saves images to `downloads/`      |
   |                                                       |
   +-------------------------------------------------------+
   |
   V
   +------------------ STAGE 2: CLASSIFY ------------------+
   |                                                       |
   |  [Analyzer: `analyzer.GeminiAnalyzer`]                |
   |     |                                                 |
   |     V                                                 |
   |  (Async Task Pool - `max_workers`)                    |
   |     - For each image, calls Gemini API                |
   |     - Prompt: "Is this apparel or product?"           |
   |     - Returns `ClassificationResult` (e.g., APPAREL)  |
   |                                                       |
   +-------------------------------------------------------+
   |
   V
[Image Sorting]
   | - Apparel Images -> VTO Processor
   | - Product/Unknown Images -> Recontext Processor
   |
   +------------------ STAGE 3: PROCESS -------------------+
   |                                                       |
   |  [Processors: `VTO` & `ProductRecontext`]             |
   |     |                                                 |
   |     V                                                 |
   |  (Async Task Pool - `max_workers`)                    |
   |     |                                                 |
   |     +--> [IF VTO]                                     |
   |     |      1. AI-Select Model (`recommend_fashion...`) |
   |     |      2. Call VTO API                           |
   |     |      3. AI-Analyze Quality (`analyze_vto...`)    |
   |     |      4. **Retry with new model if needed**       |
   |     |                                                 |
   |     +--> [IF Recontext]                               |
   |     |      1. AI-Generate Prompt (`generate_prompt...`)|
   |     |      2. Call Recontext API                     |
   |     |      3. AI-Analyze Quality (`analyze_recontext...`)|
   |                                                       |
   +-------------------------------------------------------+
   |
   V
   +-------------------- STAGE 4: STORE -------------------+
   |                                                       |
   |  [Storage: `storage.StorageManager`]                  |
   |     - Saves final image to `virtual_try_on/` or `...` |
   |     - Saves detailed `metadata.json` for each result  |
   |     - Saves `statistics.json` for the entire run      |
   |     - Uploads entire run directory to GCS (optional)  |
   |                                                       |
   +-------------------------------------------------------+
   |
   V
(End)
```

---

## 2. Module-by-Module Deep Dive

### 2.1. Entrypoint: `src/bulk_image_processor/__main__.py`

This module is the front door to the application.

*   **Purpose:** To provide a user-friendly command-line interface (CLI) and to initialize the application environment based on user input.
*   **Key Functions:**
    *   `parse_arguments()`: Defines all the CLI flags using Python's `argparse` library. This is where flags like `--csv`, `--max-workers`, and `--dry-run` are specified. The `epilog` provides helpful examples directly in the `--help` output.
    *   `main_cli()`: The core asynchronous logic. It first parses arguments, then **dynamically sets environment variables** if the user provided overrides. This is a clever way to inject user settings before the main `Settings` object is created, ensuring a single source of truth for configuration.
    *   `dry_run(csv_path)`: A crucial utility function. It simulates the setup process by initializing components and validating inputs (like the CSV file and GCS connection) without performing any actual processing. This allows users to quickly check if their environment is configured correctly.

### 2.2. Configuration: `src/bulk_image_processor/config.py`

This module ensures that the application's configuration is robust, type-safe, and easy to manage.

*   **Purpose:** To load, validate, and provide access to all application settings.
*   **Core Components:**
    *   **`Settings(BaseSettings)`:** The central Pydantic settings class.
        *   `model_config`: Specifies that settings should be loaded from a `.env` file and are case-insensitive.
        *   `validation_alias`: This allows the `.env` file to use standard environment variable naming conventions (e.g., `PROJECT_ID`) while the Python code uses more idiomatic snake\_case (`project_id`).
    *   **Nested Pydantic Models (`ProcessingConfig`, `GoogleCloudConfig`, etc.):** These are not just for organization. They provide **data validation**. For example, `max_workers: int = Field(default=10, ge=1, le=50)` ensures that the number of workers is always an integer between 1 and 50, preventing invalid configurations from ever running.
    *   **`@property` methods:** The `settings.processing`, `settings.google_cloud`, etc., properties provide a clean, organized way to access related groups of settings throughout the application.

### 2.3. Main Orchestrator: `src/bulk_image_processor/main.py`

This is the heart of the application, where the entire pipeline is coordinated.

*   **Purpose:** To manage the high-level flow of the image processing pipeline.
*   **`BulkImageProcessor` Class:**
    *   `__init__()`: Initializes all the major components (Analyzer, Processors, StorageManager) and sets up signal handlers for graceful shutdown.
    *   `_setup_signal_handlers()`: Catches `SIGINT` (Ctrl+C) and `SIGTERM` to set a `self.shutdown_requested` flag. This flag is checked within the processing loops, allowing the application to stop gracefully instead of crashing.
    *   `process_from_csv()`: The main execution method. It orchestrates the entire multi-stage process:
        1.  Calls `download_images_from_csv`.
        2.  Calls `_classify_images` on the results.
        3.  Calls `_process_classified_images` to dispatch to the correct processors.
        4.  Calls `generate_run_statistics` to create a final report.
    *   `_classify_images()` & `_process_with_processor()`: These methods demonstrate the core concurrency pattern. They create a list of `asyncio` tasks and use `asyncio.as_completed` to process them as they finish, all while respecting the `max_workers` limit via an `asyncio.Semaphore`.

### 2.4. The AI Brain: `src/bulk_image_processor/analyzer.py`

This module centralizes all interactions with the Gemini AI, effectively acting as the application's "judgment" layer.

*   **Purpose:** To abstract away the complexities of prompting and parsing responses from the Gemini API for various analytical tasks.
*   **`GeminiAnalyzer` Class:**
    *   `_setup_client()`: Initializes the Gemini client with the correct API key and model, and configures the safety settings to block harmful content.
    *   `classify_image()`:
        *   **Logic:** Takes an image and an optional context string.
        *   **Prompt Engineering:** The prompt is highly specific, defining "apparel" vs. "product" and instructing the model to identify the target gender. It also demands a JSON output, making the response easy to parse reliably.
        *   **Concurrency Handling:** The actual API call `self.client.generate_content` is **synchronous (blocking)**. To prevent it from freezing the entire `asyncio` event loop, it is wrapped in `loop.run_in_executor(None, ...)`. This runs the blocking code in a separate thread managed by `asyncio`, allowing other tasks to proceed. This is a critical pattern for mixing async and sync code.
    *   `analyze_virtual_try_on_quality()`:
        *   **Multi-modal Prompting:** This method sends **three images** (result, original apparel, original model) along with a detailed text prompt to Gemini.
        *   **Critical Check:** The prompt's most important instruction is the `CRITICAL FIRST CHECK` which asks Gemini to set `"garment_applied": false` if the try-on failed completely. This boolean flag is the linchpin of the VTO processor's intelligent retry logic.
    *   `recommend_fashion_coordination()`:
        *   **AI as a Stylist:** This is the most advanced use of AI in the system. The prompt provides Gemini with the apparel image and a "catalog" of available models and their outfits. It then gives Gemini a set of fashion rules (garment compatibility, color harmony) and asks it to choose the best model. This offloads a complex decision-making process to the AI.
    *   **Response Parsing (`_parse_*_response`)**: Each method has a corresponding private parsing method. These methods are responsible for cleaning the raw text from the API, finding the JSON block, and parsing it into a strongly-typed Pydantic model (`ClassificationResult`, `FeedbackResult`), which makes the data easy and safe to use elsewhere in the application.

### 2.5. The Workhorses: `src/bulk_image_processor/processors/`

These modules perform the core image generation tasks.

#### `base.py`

*   **`BaseProcessor` (Abstract Base Class):** Defines the contract that all processors must follow. This ensures that the main orchestrator can treat them interchangeably.
*   `process_with_retry()`: This is a crucial method inherited by all processors. It wraps the main `process_image` call in a `for` loop that handles retries with exponential backoff. This makes every processor instantly resilient to transient API errors without duplicating code.

#### `virtual_try_on.py`

*   **`VirtualTryOnProcessor`:**
    *   `_load_model_images()`: This method is key to the fashion coordination feature. It scans the `image_folder/image_models` directory and parses the filenames (e.g., `1_woman_black_trousers_white_shirt.jpg`) to create a structured database of available models and their outfits.
    *   `process_image()`: This method contains the **most complex logic in the application**. It's not just a simple API call; it's a stateful loop that manages retries on two levels:
        1.  **API Failure:** If the API call itself fails, the `process_with_retry` in the base class will handle it.
        2.  **Poor Quality Generation:** If the API succeeds but Gemini analysis says the `"garment_applied"` is `false`, this method will **add the failed model to an `exclude_models` list** and then call `_process_with_model` again, forcing it to select a *different* model for the next attempt. This is an incredibly robust way to handle unreliable AI generations.
    *   `_save_successful_result()`: This method centralizes the logic for saving the final output image, its metadata, and uploading everything to GCS.

#### `product_recontext.py`

*   **`ProductRecontextProcessor`:**
    *   **AI-driven Creativity:** Unlike the VTO processor, which is more deterministic, this processor's workflow is a creative loop:
        1.  It first calls `GeminiAnalyzer.generate_product_recontext_prompt()` to get a creative scene description.
        2.  It then feeds this AI-generated prompt into the Product Recontext API.
        3.  Finally, it sends the output back to `GeminiAnalyzer.analyze_product_recontext_quality()` for a final quality check.

### 2.6. Error Handling: `src/bulk_image_processor/exceptions.py`

This module defines a suite of custom exceptions, which is a best practice for building robust applications.

*   **Purpose:** To provide specific, meaningful error types for different failure scenarios, making debugging and error handling far more precise than catching generic `Exception`s.
*   **Hierarchy:** All custom exceptions inherit from `BulkImageProcessorError`. This allows for catching specific errors (e.g., `DownloadError`) or any application-specific error (`BulkImageProcessorError`) as needed.
*   **Contextual Information:** Each exception is designed to carry relevant context. For example, a `DownloadError` includes the `url` and `status_code`, which is invaluable for logging and debugging.

---

## 3. Setup and Usage

### 3.1. Environment Setup (`scripts/setup_env.sh`)

This script fully automates the environment setup. Running `./scripts/setup_env.sh` will:
1.  Create a Python virtual environment in `.venv/`.
2.  Install all necessary dependencies from `pyproject.toml`, including development tools.
3.  Set up pre-commit hooks to enforce code quality.
4.  Create the required directory structure (`output/`, `logs/`).
5.  Create a `.env` file from the example template for you to fill in.
6.  Run tests to confirm the setup was successful.

### 3.2. Running the Application

*   **Directly:**
    ```bash
    # Activate the environment
    source .venv/bin/activate

    # Run with default settings
    python -m bulk_image_processor

    # Run with a specific CSV and more workers
    python -m bulk_image_processor --csv /path/to/my_images.csv --max-workers 30
    ```
*   **Via the Example Script:**
    ```bash
    python scripts/run_example.py
    ```

### 3.3. Output Directory Structure

The application creates a highly organized output structure to make tracking runs easy.

```
output/
└── 2025/                  # Year
    └── 08/                # Month
        └── 13/            # Day
            ├── run_001/
            │   ├── downloads/
            │   │   └── img_sample_001_..._.jpg
            │   ├── virtual_try_on/
            │   │   └── 20250813_143000_123456_sample_001/
            │   │       ├── result.jpg
            │   │       ├── selected_model.jpg
            │   │       └── metadata.json
            │   ├── manifest.json
            │   └── statistics.json
            └── run_002/
                └── ...
└── latest -> ./2025/08/13/run_002/  # A symlink to the most recent run
```

*   **`latest` symlink:** Provides a convenient, stable path to access the results of the most recent run.
*   **`manifest.json`:** Records the initial parameters of the run.
*   **`statistics.json`:** Contains a detailed breakdown of performance and quality metrics for the completed run.
*   **`metadata.json`:** The most detailed file, containing everything about a single processed image: the original data, the full Gemini analysis, quality scores, and more.
