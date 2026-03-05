# FILE: test_pipeline.py
# ROLE: Verifies that AI + MongoDB are working together.

from vision_engine import analyze_image
from database_manager import save_analysis
import os

# 1. Pick an image you have in your folder
TEST_IMG = "test.jpg" 

if os.path.exists(TEST_IMG):
    # AI Analysis
    type_label, type_conf, detail_label, detail_conf = analyze_image(TEST_IMG)
    print(
        f"AI says type={type_label} ({type_conf*100:.1f}%), "
        f"label={detail_label} ({detail_conf*100:.1f}%)."
    )

    # Save to MongoDB
    file_size = os.path.getsize(TEST_IMG)
    db_id = save_analysis(TEST_IMG, file_size, type_label, type_conf, detail_label, detail_conf)
    print(f"Saved to MongoDB with ID: {db_id}")
else:
    print("Please put a test.jpg in your folder to run the pipeline test!")
