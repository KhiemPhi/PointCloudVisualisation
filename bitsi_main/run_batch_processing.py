#!/usr/bin/env python3
"""
Simple script to run batch processing for ShapeNetPart segmentation.
"""

import subprocess
import sys
import os

def main():
    """Run the batch processing script."""
    
    print("🚀 Starting ShapeNetPart batch processing...")
    print("=" * 60)
    
    # Check if the batch processing script exists
    script_path = "/home/khiem/bitsi/batch_shapenetpart_segmentation.py"
    if not os.path.exists(script_path):
        print(f"❌ Batch processing script not found: {script_path}")
        return
    
    # Run the batch processing
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print("✅ Batch processing completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch processing failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return
    
    print("\n" + "=" * 60)
    print("🔍 Running verification...")
    
    # Run verification
    verify_script = "/home/khiem/bitsi/verify_segmentation_results.py"
    if os.path.exists(verify_script):
        try:
            result = subprocess.run([sys.executable, verify_script], 
                                  capture_output=True, text=True, check=True)
            print("✅ Verification completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Verification failed with error code {e.returncode}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
    else:
        print(f"⚠️ Verification script not found: {verify_script}")
    
    print("\n🎉 All processing completed!")

if __name__ == "__main__":
    main()
