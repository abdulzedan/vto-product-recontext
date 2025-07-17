"""Entry point for the bulk image processor module."""

import argparse
import asyncio
import sys
from pathlib import Path

from .main import main


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bulk image processor with Virtual Try-On and Product Recontext",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images from default CSV location
  python -m bulk_image_processor
  
  # Process images from specific CSV file
  python -m bulk_image_processor --csv /path/to/images.csv
  
  # Process with custom settings
  python -m bulk_image_processor --csv /path/to/images.csv --max-workers 20
        """,
    )
    
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("./image_folder/img_conversion_table.csv"),
        help="Path to CSV file containing image URLs (default: ./image_folder/img_conversion_table.csv)",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of concurrent workers (overrides .env setting)",
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of retry attempts (overrides .env setting)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local output directory (overrides .env setting)",
    )
    
    parser.add_argument(
        "--disable-gcs",
        action="store_true",
        help="Disable Google Cloud Storage upload",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (overrides .env setting)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual processing",
    )
    
    return parser.parse_args()


async def main_cli():
    """Main CLI entry point."""
    args = parse_arguments()
    
    # Validate CSV file
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Override settings if provided
    if args.max_workers or args.max_retries or args.output_dir or args.disable_gcs or args.log_level:
        import os
        from .config import get_settings
        
        # Load existing settings
        settings = get_settings()
        
        # Apply overrides
        if args.max_workers:
            os.environ["MAX_WORKERS"] = str(args.max_workers)
        if args.max_retries:
            os.environ["MAX_RETRIES"] = str(args.max_retries)
        if args.output_dir:
            os.environ["LOCAL_OUTPUT_DIR"] = str(args.output_dir)
        if args.disable_gcs:
            os.environ["ENABLE_GCS_UPLOAD"] = "false"
        if args.log_level:
            os.environ["LOG_LEVEL"] = args.log_level
        
        # Reload settings with overrides
        settings = get_settings()
    
    # Perform dry run if requested
    if args.dry_run:
        print("Performing dry run...")
        await dry_run(args.csv)
        return
    
    # Run main processing
    await main(args.csv)


async def dry_run(csv_path: Path):
    """Perform a dry run to validate setup."""
    from .config import get_settings
    from .downloader import ImageDownloader
    
    try:
        # Test configuration
        settings = get_settings()
        print(f"✓ Configuration loaded successfully")
        print(f"  Project ID: {settings.google_cloud.project_id}")
        print(f"  Location: {settings.google_cloud.location}")
        print(f"  Max workers: {settings.processing.max_workers}")
        print(f"  GCS enabled: {settings.storage.enable_gcs_upload}")
        
        # Test CSV loading
        async with ImageDownloader(settings) as downloader:
            records = downloader.load_csv(csv_path)
            print(f"✓ CSV loaded successfully: {len(records)} records")
            
            # Show sample records
            if records:
                print("  Sample records:")
                for i, record in enumerate(records[:3]):
                    print(f"    {i+1}. ID: {record.id}, URL: {record.image_url[:50]}...")
        
        # Test Google Cloud connectivity
        from .analyzer import GeminiAnalyzer
        analyzer = GeminiAnalyzer(settings)
        print("✓ Gemini analyzer initialized successfully")
        
        # Test storage
        from .storage import StorageManager
        storage_manager = StorageManager(settings)
        print("✓ Storage manager initialized successfully")
        
        stats = storage_manager.get_storage_stats()
        print(f"  Local storage: {stats['local_storage']}")
        print(f"  GCS storage: {stats['gcs_storage']}")
        
        print("\n✓ Dry run completed successfully. System is ready for processing.")
        
    except Exception as e:
        print(f"✗ Dry run failed: {e}")
        sys.exit(1)


def cli_main():
    """Synchronous CLI entry point."""
    asyncio.run(main_cli())


if __name__ == "__main__":
    cli_main()