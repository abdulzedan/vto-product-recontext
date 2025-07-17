#!/usr/bin/env python3
"""Example script to run the bulk image processor."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bulk_image_processor.main import BulkImageProcessor


async def run_example():
    """Run example processing."""
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("Error: .env file not found. Please copy .env.example to .env and configure it.")
        return
    
    # Check if CSV file exists
    csv_file = Path("image_folder/img_conversion_table.csv")
    if not csv_file.exists():
        print("Error: CSV file not found. Please create image_folder/img_conversion_table.csv with your image URLs.")
        return
    
    print("Starting bulk image processing example...")
    
    try:
        # Initialize processor
        processor = BulkImageProcessor()
        
        # Show system status
        status = processor.get_system_status()
        print(f"System Status: {status}")
        
        # Run processing
        summary = await processor.process_from_csv(csv_file)
        
        print("\nProcessing Summary:")
        print(f"Total processing time: {summary['total_processing_time']} seconds")
        print(f"Download stats: {summary['download_stats']}")
        print(f"Classification stats: {summary['classification_stats']}")
        print(f"Processing stats: {summary['processing_stats']}")
        print(f"Quality stats: {summary['quality_stats']}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_example())