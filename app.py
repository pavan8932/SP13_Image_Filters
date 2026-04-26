# ---------------- SP13: IMAGE SMOOTHING & SHARPENING ----------------
# Web App for Mean, Gaussian, and Laplacian Filters
# Deployable on Render.com

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
import io
import sys

# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.secret_key = 'sp13-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Folder setup
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# ---------------- SP13: FILTER FUNCTIONS ----------------

def apply_mean_filter(image, kernel_size=5):
    """
    MEAN FILTER - Smoothing/Blurring
    Replaces each pixel with average of neighbors
    """
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = min(kernel_size, min(image.shape[:2]) // 2)
    result = cv2.blur(image, (kernel_size, kernel_size))
    return result

def apply_gaussian_filter(image, kernel_size=5, sigma=1.5):
    """
    GAUSSIAN FILTER - Smoothing with weighted average
    Preserves edges better than mean filter
    """
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = min(kernel_size, min(image.shape[:2]) // 2)
    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return result

def apply_median_filter(image, kernel_size=3):
    """
    MEDIAN FILTER - Excellent for salt-and-pepper noise
    """
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    result = cv2.medianBlur(image, kernel_size)
    return result

def apply_laplacian_sharpening(image, strength=1.3):
    """
    LAPLACIAN FILTER - Edge detection and sharpening
    Enhances edges and details
    """
    # Convert to grayscale for edge detection if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Noise reduction before edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian_abs = np.absolute(laplacian)
    laplacian_uint8 = np.uint8(np.clip(laplacian_abs, 0, 255))
    
    # Sharpen: original + strength * edges
    if len(image.shape) == 3:
        edges_colored = cv2.cvtColor(laplacian_uint8, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(image, 1.0, edges_colored, strength, 0)
    else:
        sharpened = cv2.addWeighted(image, 1.0, laplacian_uint8, strength, 0)
    
    return sharpened, laplacian_uint8

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    BILATERAL FILTER - Edge-preserving smoothing
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# ---------------- HISTOGRAM AND VISUALIZATION ----------------

def create_histogram(image, title, filename):
    """Create and save histogram plot"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    plt.figure(figsize=(10, 5))
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, edgecolor='black')
    plt.title(f'{title} - Pixel Intensity Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, 255])
    
    # Add mean line
    mean_val = np.mean(gray)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}')
    plt.legend()
    
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path

def create_comparison_plot(original, mean_img, gaussian_img, laplacian_img, filename):
    """Create 2x2 comparison grid"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert to grayscale for display if needed
    def to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    images = [
        (to_gray(original), "Original Image", 0, 0),
        (to_gray(mean_img), "Mean Filter (Smoothing)", 0, 1),
        (to_gray(gaussian_img), "Gaussian Filter (Smoothing)", 1, 0),
        (to_gray(laplacian_img), "Laplacian Filter (Sharpening)", 1, 1)
    ]
    
    for img, title, row, col in images:
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.suptitle("SP13: Image Smoothing vs Sharpening Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def calculate_sharpness(image):
    """Calculate sharpness metric using Laplacian variance"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ---------------- FLASK ROUTES ----------------

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and apply filters"""
    
    if 'image' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Use PNG, JPG, or JPEG.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save original image
        filename = secure_filename(file.filename)
        timestamp = get_timestamp()
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            flash('Could not read image file', 'error')
            return redirect(url_for('index'))
        
        # Get image info
        height, width = image.shape[:2]
        image_type = "Grayscale" if len(image.shape) == 2 else "Color"
        
        # Apply filters
        mean_result = apply_mean_filter(image, kernel_size=5)
        gaussian_result = apply_gaussian_filter(image, kernel_size=5, sigma=1.5)
        median_result = apply_median_filter(image, kernel_size=3)
        laplacian_result, laplacian_edges = apply_laplacian_sharpening(image, strength=1.3)
        bilateral_result = apply_bilateral_filter(image)
        
        # Save results
        base_name = f"{timestamp}_{name}"
        
        mean_path = os.path.join(OUTPUT_FOLDER, f'mean_{base_name}.png')
        gaussian_path = os.path.join(OUTPUT_FOLDER, f'gaussian_{base_name}.png')
        median_path = os.path.join(OUTPUT_FOLDER, f'median_{base_name}.png')
        laplacian_path = os.path.join(OUTPUT_FOLDER, f'laplacian_{base_name}.png')
        edges_path = os.path.join(OUTPUT_FOLDER, f'edges_{base_name}.png')
        bilateral_path = os.path.join(OUTPUT_FOLDER, f'bilateral_{base_name}.png')
        
        cv2.imwrite(mean_path, mean_result)
        cv2.imwrite(gaussian_path, gaussian_result)
        cv2.imwrite(median_path, median_result)
        cv2.imwrite(laplacian_path, laplacian_result)
        cv2.imwrite(edges_path, laplacian_edges)
        cv2.imwrite(bilateral_path, bilateral_result)
        
        # Create histograms
        hist_original = create_histogram(image, "Original Image", f'hist_original_{base_name}.png')
        hist_mean = create_histogram(mean_result, "Mean Filter", f'hist_mean_{base_name}.png')
        hist_gaussian = create_histogram(gaussian_result, "Gaussian Filter", f'hist_gaussian_{base_name}.png')
        hist_laplacian = create_histogram(laplacian_result, "Laplacian Filter", f'hist_laplacian_{base_name}.png')
        
        # Create comparison grid
        comparison_grid = create_comparison_plot(image, mean_result, gaussian_result, laplacian_result, f'grid_{base_name}.png')
        
        # Calculate sharpness metrics
        metrics = {
            'original': round(calculate_sharpness(image), 2),
            'mean': round(calculate_sharpness(mean_result), 2),
            'gaussian': round(calculate_sharpness(gaussian_result), 2),
            'laplacian': round(calculate_sharpness(laplacian_result), 2),
            'median': round(calculate_sharpness(median_result), 2)
        }
        
        return render_template('index.html',
            input_image='/' + input_path,
            mean_image='/' + mean_path,
            gaussian_image='/' + gaussian_path,
            median_image='/' + median_path,
            laplacian_image='/' + laplacian_path,
            laplacian_edge_image='/' + edges_path,
            bilateral_image='/' + bilateral_path,
            hist_original='/' + hist_original,
            hist_mean='/' + hist_mean,
            hist_gaussian='/' + hist_gaussian,
            hist_laplacian='/' + hist_laplacian,
            comparison_grid='/' + comparison_grid,
            image_width=width,
            image_height=height,
            image_type=image_type,
            metrics=metrics
        )
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/clear')
def clear_all():
    """Clear all uploaded and processed images"""
    try:
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for f in os.listdir(folder):
                os.unlink(os.path.join(folder, f))
        flash('All images cleared successfully', 'success')
    except Exception as e:
        flash('Error clearing files', 'error')
    return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check for Render"""
    return jsonify({"status": "healthy", "app": "SP13 Image Filters"}), 200

# ---------------- MAIN ----------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)