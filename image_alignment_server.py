#!/usr/bin/env python3

import sys
import cv2
import numpy as np
import json
import base64
import tempfile
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import requests
from datetime import datetime, timedelta
import uuid
import hashlib
import threading
from urllib.parse import parse_qs, urlparse




# ============================================================================
# CONFIGURATION AND GLOBAL STATE
# ============================================================================


config = None
DEBUG_DIR = "debug_output"
KEYS_DB_PATH = "api_keys.json"
ADMIN_DB_PATH = "admin_creds.json"
API_USAGE_DB_PATH = "api_usage.json"


API_KEYS_LOCK = threading.RLock()
API_USAGE_LOCK = threading.RLock()
API_KEYS = {}
ADMIN_CREDENTIALS = {}
SESSION_TOKENS = {}  # {token: {'username': ..., 'expires': ...}}
API_USAGE = {} # {api_key: [timestamp1, timestamp2, ...]}




# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================


def load_api_keys():
    """Load API keys from local storage."""
    global API_KEYS
    if os.path.exists(KEYS_DB_PATH):
        try:
            with open(KEYS_DB_PATH, 'r') as f:
                data = json.load(f)
                API_KEYS = data.get('keys', {})
                print(f"[DB] Loaded {len(API_KEYS)} API keys from {KEYS_DB_PATH}")
        except Exception as e:
            print(f"[DB] Error loading API keys: {e}")
            API_KEYS = {}




def save_api_keys():
    """Save API keys to local storage."""
    with API_KEYS_LOCK:
        try:
            with open(KEYS_DB_PATH, 'w') as f:
                json.dump({'keys': API_KEYS}, f, indent=2, default=str)
                print(f"[DB] Saved {len(API_KEYS)} API keys to {KEYS_DB_PATH}")
        except Exception as e:
            print(f"[DB] Error saving API keys: {e}")


def load_api_usage():
    """Load API usage data from local storage."""
    global API_USAGE
    if os.path.exists(API_USAGE_DB_PATH):
        try:
            with open(API_USAGE_DB_PATH, 'r') as f:
                API_USAGE = json.load(f)
                count = sum(len(v) for v in API_USAGE.values())
                print(f"[DB] Loaded {count} usage records for {len(API_USAGE)} keys from {API_USAGE_DB_PATH}")
        except Exception as e:
            print(f"[DB] Error loading API usage data: {e}")
            API_USAGE = {}




def save_api_usage():
    """Save API usage data to local storage."""
    with API_USAGE_LOCK:
        try:
            with open(API_USAGE_DB_PATH, 'w') as f:
                json.dump(API_USAGE, f, indent=2)
        except Exception as e:
            print(f"[DB] Error saving API usage data: {e}")


def record_api_usage(api_key):
    """Record a successful API call for the given key."""
    with API_USAGE_LOCK:
        timestamp = datetime.now().isoformat()
        if api_key not in API_USAGE:
            API_USAGE[api_key] = []
        API_USAGE[api_key].append(timestamp)
    # Saving on every request can be I/O intensive.
    # For high-load, consider batching saves.
    save_api_usage()


def load_admin_credentials():
    """Load admin credentials from storage."""
    global ADMIN_CREDENTIALS
    if os.path.exists(ADMIN_DB_PATH):
        try:
            with open(ADMIN_DB_PATH, 'r') as f:
                data = json.load(f)
                ADMIN_CREDENTIALS = data.get('users', {})
                print(f"[DB] Loaded {len(ADMIN_CREDENTIALS)} admin accounts from {ADMIN_DB_PATH}")
        except Exception as e:
            print(f"[DB] Error loading admin credentials: {e}")




def save_admin_credentials():
    """Save admin credentials to storage."""
    try:
        with open(ADMIN_DB_PATH, 'w') as f:
            json.dump({'users': ADMIN_CREDENTIALS}, f, indent=2)
            print(f"[DB] Saved {len(ADMIN_CREDENTIALS)} admin accounts to {ADMIN_DB_PATH}")
    except Exception as e:
        print(f"[DB] Error saving admin credentials: {e}")




def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()




def verify_api_key(api_key):
    """Verify if API key is valid and not expired. Returns (is_valid, message)"""
    with API_KEYS_LOCK:
        if api_key not in API_KEYS:
            return False, "Invalid API key"


        key_data = API_KEYS[api_key]


        # Check expiration
        if key_data.get('expires_at'):
            try:
                expires_at = datetime.fromisoformat(key_data['expires_at'])
                if datetime.now() > expires_at:
                    return False, "API key has expired"
            except ValueError:
                return False, "Invalid expiration date format"


        return True, "Valid"




def detect_request_type_from_headers(headers):
    """
    Detect request type from HTTP headers.
    Returns 'gemini' if x-goog-api-key header exists, 'flux' if Authorization header exists.
    """
    if 'x-goog-api-key' in headers:
        return 'gemini'
    elif 'authorization' in headers:
        return 'flux'
    return None




def extract_api_key_from_headers(headers, request_type):
    """
    Extract API key from HTTP headers based on request type.
    Gemini: x-goog-api-key header
    FLUX: Authorization header (Bearer token)
    """
    if request_type == 'gemini':
        api_key = headers.get('x-goog-api-key', '')
        if api_key:
            return api_key
    elif request_type == 'flux':
        auth_header = headers.get('authorization', '')
        if auth_header.startswith('Bearer '):
            return auth_header[7:]  # Remove "Bearer " prefix
    return None




# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================


def load_config():
    global config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_host = config.get('api_host', '127.0.0.1')
        api_port = config.get('api_port', 8080)
        admin_host = config.get('admin_host', '127.0.0.1')
        admin_port = config.get('admin_port', 8081)
        print(f"Config loaded: API on {api_host}:{api_port}, Admin on {admin_host}:{admin_port}")


        if config.get('debug_mode', False):
            print("DEBUG MODE ENABLED")
            if not os.path.exists(DEBUG_DIR):
                os.makedirs(DEBUG_DIR)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)




def get_debug_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]




def save_debug_image(image, suffix):
    if not config.get('debug_mode', False):
        return None
    timestamp = get_debug_timestamp()
    filename = f"{timestamp}_{suffix}.png"
    filepath = os.path.join(DEBUG_DIR, filename)
    cv2.imwrite(filepath, image)
    print(f"[DEBUG] Saved: {filename}")
    return filepath




def debug_print(message):
    """Print message only in debug mode."""
    if config.get('debug_mode', False):
        print(f"[DEBUG] {message}")




def round_to_multiple_of_8(value):
    return int(round(value / 8.0)) * 8




def scale_to_fit_max_dimension(image, max_dim=1792):
    height, width = image.shape[:2]
    if width <= max_dim and height <= max_dim:
        return image
    scale = min(max_dim / width, max_dim / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    debug_print(f"Scaling image from {width}x{height} to {new_width}x{new_height} (max_dim={max_dim})")
    print(f"Scaling image from {width}x{height} to {new_width}x{new_height}")
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)




def add_border(image, border_ratio=0.05, ensure_multiple_of_8=False, max_dimension=None):
    height, width = image.shape[:2]
    original_dimensions = (width, height)


    debug_print(f"add_border() called: border_ratio={border_ratio}, ensure_multiple_of_8={ensure_multiple_of_8}, max_dimension={max_dimension}")
    debug_print(f"  Input image dimensions: {width}x{height}")


    if not config.get('enable_border', True):
        debug_print("Border disabled in config")
        return image, original_dimensions, original_dimensions


    border_color = config.get('border_color', [0, 255, 0])
    avg_dim = (height + width) / 2
    border_width = int(avg_dim * border_ratio)
    debug_print(f"  Border color: {border_color}, border_width: {border_width}")


    if ensure_multiple_of_8:
        tentative_width = width + 2 * border_width
        tentative_height = height + 2 * border_width
        target_width = round_to_multiple_of_8(tentative_width)
        target_height = round_to_multiple_of_8(tentative_height)
        total_width_padding = target_width - width
        total_height_padding = target_height - height
        border_left = total_width_padding // 2
        border_right = total_width_padding - border_left
        border_top = total_height_padding // 2
        border_bottom = total_height_padding - border_top


        debug_print(f"  Multiple of 8 mode: tentative {tentative_width}x{tentative_height} -> target {target_width}x{target_height}")
        debug_print(f"  Padding: top={border_top}, bottom={border_bottom}, left={border_left}, right={border_right}")


        bordered_image = cv2.copyMakeBorder(
            image, border_top, border_bottom, border_left, border_right,
            cv2.BORDER_CONSTANT, value=border_color
        )
        pre_scaled_height, pre_scaled_width = bordered_image.shape[:2]
        pre_scaled_dimensions = (pre_scaled_width, pre_scaled_height)
        debug_print(f"  After border: {pre_scaled_width}x{pre_scaled_height}")


        if max_dimension:
            bordered_image = scale_to_fit_max_dimension(bordered_image, max_dimension)
            h, w = bordered_image.shape[:2]
            new_w = round_to_multiple_of_8(w)
            new_h = round_to_multiple_of_8(h)
            if new_w != w or new_h != h:
                debug_print(f"  Adjusting to multiple of 8: {w}x{h} -> {new_w}x{new_h}")
                bordered_image = cv2.resize(bordered_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        debug_print(f"  Simple border mode: adding {border_width}px on all sides")
        bordered_image = cv2.copyMakeBorder(
            image, border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT, value=border_color
        )
        pre_scaled_height, pre_scaled_width = bordered_image.shape[:2]
        pre_scaled_dimensions = (pre_scaled_width, pre_scaled_height)
        debug_print(f"  After border: {pre_scaled_width}x{pre_scaled_height}")


    debug_print(f"  Final output: {bordered_image.shape[1]}x{bordered_image.shape[0]}")
    return bordered_image, original_dimensions, pre_scaled_dimensions




def upscale_to_match_height(image, target_height):
    height, width = image.shape[:2]
    debug_print(f"upscale_to_match_height() called: current={width}x{height}, target_height={target_height}")
    
    if height >= target_height:
        debug_print(f"  No upscaling needed: current height {height} >= target {target_height}")
        return image
    
    scale = target_height / height
    new_width = int(width * scale)
    debug_print(f"  Upscaling with factor {scale:.4f}: {width}x{height} -> {new_width}x{target_height}")
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)




def estimateScaleTranslateTransform(points_b, points_a, ransacReprojThreshold=5.0, maxIters=2000, minMatches=4):
    debug_print(f"estimateScaleTranslateTransform() called: points_b={len(points_b)}, points_a={len(points_a)}, threshold={ransacReprojThreshold}, maxIters={maxIters}, minMatches={minMatches}")
    
    if len(points_a) < minMatches:
        debug_print(f"  Insufficient points: {len(points_a)} < {minMatches}")
        return None, None


    best_inlier_count = 0
    best_inlier_mask = None


    for iteration in range(maxIters):
        indices = np.random.choice(len(points_a), 2, replace=False)
        sample_a = points_a[indices]
        sample_b = points_b[indices]


        if abs(sample_b[0, 0] - sample_b[1, 0]) < 1e-6 or abs(sample_b[0, 1] - sample_b[1, 1]) < 1e-6:
            continue


        s_x = (sample_a[0, 0] - sample_a[1, 0]) / (sample_b[0, 0] - sample_b[1, 0])
        t_x = sample_a[0, 0] - s_x * sample_b[0, 0]
        s_y = (sample_a[0, 1] - sample_a[1, 1]) / (sample_b[0, 1] - sample_b[1, 1])
        t_y = sample_a[0, 1] - s_y * sample_b[0, 1]


        current_transform = np.array([[s_x, 0, t_x], [0, s_y, t_y]], dtype=np.float32)
        points_b_h = np.hstack([points_b, np.ones((points_b.shape[0], 1))])
        transformed_points = (current_transform @ points_b_h.T).T
        errors = np.sum((points_a - transformed_points)**2, axis=1)
        current_inlier_mask = errors < ransacReprojThreshold**2
        current_inlier_count = np.sum(current_inlier_mask)


        if current_inlier_count > best_inlier_count:
            best_inlier_count = current_inlier_count
            best_inlier_mask = current_inlier_mask


    if best_inlier_mask is None or np.sum(best_inlier_mask) < minMatches:
        debug_print(f"  RANSAC failed: inliers={np.sum(best_inlier_mask) if best_inlier_mask is not None else 0} < {minMatches}")
        return None, None


    inlier_points_a = points_a[best_inlier_mask]
    inlier_points_b = points_b[best_inlier_mask]
    best_inlier_count = np.sum(best_inlier_mask)


    debug_print(f"  RANSAC succeeded: {best_inlier_count} inliers found")


    A_x = np.hstack([inlier_points_b[:, 0:1], np.ones((best_inlier_count, 1))])
    B_x = inlier_points_a[:, 0:1]
    params_x, _, _, _ = np.linalg.lstsq(A_x, B_x, rcond=None)
    s_x, t_x = params_x.flatten()


    A_y = np.hstack([inlier_points_b[:, 1:2], np.ones((best_inlier_count, 1))])
    B_y = inlier_points_a[:, 1:2]
    params_y, _, _, _ = np.linalg.lstsq(A_y, B_y, rcond=None)
    s_y, t_y = params_y.flatten()


    final_transform = np.array([[s_x, 0, t_x], [0, s_y, t_y]], dtype=np.float32)
    debug_print(f"  Final transform: sx={s_x:.4f}, sy={s_y:.4f}, tx={t_x:.2f}, ty={t_y:.2f}")
    return final_transform, best_inlier_mask




def align_images_orb(img_a_path, img_b_path, output_path, max_features=None):
    if not config.get('enable_alignment', True):
        debug_print("Alignment disabled in config")
        img_b = cv2.imread(img_b_path)
        if img_b is None: return False
        cv2.imwrite(output_path, img_b)
        return True


    if max_features is None:
        max_features = config.get('max_features', 5000)


    lowe_ratio_threshold = config.get('lowe_ratio_threshold', 0.75)
    ransac_threshold = config.get('ransac_threshold', 5.0)
    min_matches_required = config.get('min_matches_required', 4)


    debug_print(f"align_images_orb() called: max_features={max_features}, lowe_ratio_threshold={lowe_ratio_threshold}, ransac_threshold={ransac_threshold}, min_matches_required={min_matches_required}")


    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)


    if img_a is None or img_b is None:
        debug_print(f"  Failed to load images: img_a={img_a is not None}, img_b={img_b is not None}")
        return False


    debug_print(f"  Image A: {img_a.shape[1]}x{img_a.shape[0]}, Image B: {img_b.shape[1]}x{img_b.shape[0]}")


    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)


    orb_scale_factor = config.get('orb_scale_factor', 1.2)
    orb_nlevels = config.get('orb_nlevels', 8)
    orb_edge_threshold = config.get('orb_edge_threshold', 31)


    debug_print(f"  ORB parameters: scale_factor={orb_scale_factor}, nlevels={orb_nlevels}, edge_threshold={orb_edge_threshold}")


    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=orb_scale_factor,
        nlevels=orb_nlevels,
        edgeThreshold=orb_edge_threshold
    )


    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)


    debug_print(f"  Keypoints found: img_a={len(keypoints_a)}, img_b={len(keypoints_b)}")


    if len(keypoints_a) < min_matches_required or len(keypoints_b) < min_matches_required:
        debug_print(f"  Insufficient keypoints")
        return False


    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    good_matches = [m for m, n in matches if len(matches[0])==2 and m.distance < lowe_ratio_threshold * n.distance]


    debug_print(f"  Matches: total={len(matches)}, good={len(good_matches)} (threshold={lowe_ratio_threshold})")


    if len(good_matches) < min_matches_required:
        debug_print(f"  Insufficient good matches: {len(good_matches)} < {min_matches_required}")
        return False


    points_b = np.float32([keypoints_b[m.queryIdx].pt for m in good_matches])
    points_a = np.float32([keypoints_a[m.trainIdx].pt for m in good_matches])


    transform_matrix, inliers = estimateScaleTranslateTransform(
        points_b, points_a,
        ransacReprojThreshold=ransac_threshold,
        minMatches=min_matches_required
    )


    if transform_matrix is None:
        debug_print(f"  Transform estimation failed")
        return False


    height_a, width_a = img_a.shape[:2]
    debug_print(f"  Applying affine transform to warp image B to A dimensions: {width_a}x{height_a}")
    aligned_img = cv2.warpAffine(img_b, transform_matrix, (width_a, height_a))
    cv2.imwrite(output_path, aligned_img)
    debug_print(f"  Alignment completed successfully")
    return True




# ============================================================================
# REQUEST TYPE DETECTION
# ============================================================================


def detect_request_type_from_body(data):
    """
    Detect request type from request body.
    Gemini requests have 'contents' field.
    FLUX requests have 'model' field with 'FLUX' string.
    """
    if 'contents' in data:
        return 'gemini'
    elif 'model' in data and 'FLUX' in data.get('model', ''):
        return 'flux'
    return None




def handle_gemini_request(data, orig_path):
    print("=== Processing Gemini Request ===")
    debug_print(f"Gemini request data keys: {list(data.keys())}")
    
    if 'contents' not in data or not data['contents'] or 'parts' not in data['contents'][0]:
        raise ValueError('Invalid Gemini request format')


    original_b64 = data['contents'][0]['parts'][0]['inline_data']['data']
    debug_print(f"Original image base64 length: {len(original_b64)}")
    
    with open(orig_path, 'wb') as f:
        f.write(base64.b64decode(original_b64))


    orig_img = cv2.imread(orig_path)
    debug_print(f"Loaded original image: {orig_img.shape[1]}x{orig_img.shape[0]}")
    
    bordered_img, _, pre_scaled_dims = add_border(orig_img, config.get('border_ratio', 0.05))
    save_debug_image(bordered_img, "01_bordered_original")


    bordered_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(bordered_path, bordered_img)
    target_upscale_height = pre_scaled_dims[1]
    debug_print(f"Target upscale height: {target_upscale_height}")


    with open(bordered_path, 'rb') as f:
        bordered_b64 = base64.b64encode(f.read()).decode('utf-8')


    data['contents'][0]['parts'][0]['inline_data']['data'] = bordered_b64


    debug_print(f"Forwarding request to Gemini API: {config['gemini_endpoint']}")
    headers = {"x-goog-api-key": config['gemini_api_key'], "Content-Type": "application/json"}
    response = requests.post(config['gemini_endpoint'], headers=headers, json=data)
    os.unlink(bordered_path)


    if response.status_code != 200:
        debug_print(f"Gemini API returned status {response.status_code}")
        print(f"[ERROR] Gemini API response ({response.status_code}):")
        try:
            error_json = response.json()
            print(json.dumps(error_json, indent=2))
        except:
            print(response.text)
        raise Exception(f"Gemini API Error {response.status_code}")


    api_data = response.json()
    debug_print(f"Gemini API response keys: {list(api_data.keys())}")
    
    if 'candidates' not in api_data or not api_data['candidates']:
        debug_print(f"Gemini API response JSON: {json.dumps(api_data, indent=2)}")
        raise Exception('No candidates in Gemini response')


    parts = api_data['candidates'][0].get('content', {}).get('parts', [])
    result_b64 = next((p['inlineData']['data'] for p in parts if 'inlineData' in p), None)


    if not result_b64:
        debug_print(f"Gemini API response JSON: {json.dumps(api_data, indent=2)}")
        raise Exception('No inline data in response')


    return api_data, result_b64, parts, target_upscale_height




def handle_flux_request(data, orig_path):
    print("=== Processing FLUX Request ===")
    debug_print(f"FLUX request data keys: {list(data.keys())}")
    
    # Validate steps field: must be a positive integer, cap at 30
    steps = data.get('steps')
    if steps is None:
        raise ValueError('Invalid FLUX request: missing steps')
    try:
        steps_int = int(steps)
        if steps_int <= 0:
            raise ValueError('Invalid FLUX request: steps must be positive')
        if steps_int > 30:
            steps_int = 30
        data['steps'] = steps_int  # normalize to bounded value
    except (ValueError, TypeError):
        raise ValueError('Invalid FLUX request: steps must be a positive integer')


    if 'image_url' not in data or not data['image_url'].startswith('data:image/'):
        raise ValueError('Invalid FLUX request')



    original_b64 = data['image_url'].split(',', 1)[1]
    debug_print(f"Original image base64 length: {len(original_b64)}")
    
    with open(orig_path, 'wb') as f:
        f.write(base64.b64decode(original_b64))



    orig_img = cv2.imread(orig_path)
    debug_print(f"Loaded original image: {orig_img.shape[1]}x{orig_img.shape[0]}")
    
    bordered_img, _, pre_scaled_dims = add_border(
        orig_img, config.get('border_ratio', 0.05),
        ensure_multiple_of_8=True,
        max_dimension=config.get('max_flux_dimension', 1792)
    )
    save_debug_image(bordered_img, "01_bordered_original")



    bordered_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(bordered_path, bordered_img)
    target_upscale_height = pre_scaled_dims[1]
    debug_print(f"Target upscale height: {target_upscale_height}")



    bordered_height, bordered_width = bordered_img.shape[:2]
    with open(bordered_path, 'rb') as f:
        bordered_b64 = base64.b64encode(f.read()).decode('utf-8')



    data.update({
        'image_url': f"data:image/png;base64,{bordered_b64}",
        'width': bordered_width,
        'height': bordered_height
    })



    debug_print(f"Forwarding request to FLUX API: {config['flux_endpoint']}")
    debug_print(f"Request image dimensions: {bordered_width}x{bordered_height}")
    headers = {"Authorization": f"Bearer {config['flux_api_key']}", "Content-Type": "application/json"}
    response = requests.post(config['flux_endpoint'], headers=headers, json=data)
    os.unlink(bordered_path)



    if response.status_code != 200:
        debug_print(f"FLUX API returned status {response.status_code}")
        print(f"[ERROR] FLUX API response ({response.status_code}):")
        try:
            error_json = response.json()
            print(json.dumps(error_json, indent=2))
        except:
            print(response.text)
        raise Exception(f"FLUX API Error {response.status_code}")



    api_data = response.json()
    debug_print(f"FLUX API response keys: {list(api_data.keys())}")
    
    if 'data' not in api_data or not api_data['data']:
        debug_print(f"FLUX API response JSON: {json.dumps(api_data, indent=2)}")
        raise Exception('No image in FLUX response')



    image_url = api_data['data'][0]['url']
    debug_print(f"Downloaded image URL: {image_url[:80]}...")
    
    image_response = requests.get(image_url)
    if not image_response.ok:
        debug_print(f"Image download failed: {image_response.status_code}")
        raise Exception(f"Failed to download FLUX result")



    result_b64 = base64.b64encode(image_response.content).decode('utf-8')
    return api_data, result_b64, None, target_upscale_height





# ============================================================================
# HTTP REQUEST HANDLERS
# ============================================================================


class APIRequestHandler(BaseHTTPRequestHandler):
    """Handler for API port (Port A) - client requests"""


    def do_POST(self):
        if self.path != '/':
            return self.send_error(404)


        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return self.send_error(400, 'No content')


            request_data = json.loads(self.rfile.read(content_length))


            # Step 1: Detect request type from headers first
            request_type = detect_request_type_from_headers(self.headers)
            if not request_type:
                print("[AUTH] Could not detect request type from headers")
                return self.send_error(400, 'Invalid request type')


            print(f"[AUTH] Detected request type: {request_type.upper()}")


            # Step 2: Extract API key from headers based on detected type
            api_key = extract_api_key_from_headers(self.headers, request_type)
            if not api_key:
                print(f"[AUTH] No API key found in {request_type.upper()} headers")
                return self.send_error(401, f'No API key in {request_type.upper()} request')


            # Step 3: Verify API key
            is_valid, message = verify_api_key(api_key)
            if not is_valid:
                print(f"[AUTH] API key verification failed: {message}")
                return self.send_error(401, message)


            print(f"[AUTH] Valid API key accepted")
            record_api_usage(api_key) # Record successful, authenticated request


            # Step 4: Verify request body matches detected type
            body_request_type = detect_request_type_from_body(request_data)
            if body_request_type != request_type:
                print(f"[VALIDATION] Request type mismatch: headers say {request_type}, body says {body_request_type}")
                return self.send_error(400, 'Request type mismatch between headers and body')


            print(f"[API] Processing {request_type.upper()} request")


            orig_path = result_path = aligned_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    orig_path = f.name


                if request_type == 'gemini':
                    api_data, result_b64, parts, target_height = handle_gemini_request(request_data, orig_path)
                else:
                    api_data, result_b64, parts, target_height = handle_flux_request(request_data, orig_path)


                result_bytes = base64.b64decode(result_b64)
                result_img = cv2.imdecode(np.frombuffer(result_bytes, np.uint8), cv2.IMREAD_COLOR)
                if result_img is None:
                    raise Exception('Failed to decode result')


                debug_print(f"Decoded result image: {result_img.shape[1]}x{result_img.shape[0]}")
                save_debug_image(result_img, "02_result_raw")
                
                result_img = upscale_to_match_height(result_img, target_height)
                save_debug_image(result_img, "03_result_upscaled")


                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    result_path = f.name
                cv2.imwrite(result_path, result_img)


                aligned_path = tempfile.mktemp(suffix='.png')
                if not align_images_orb(orig_path, result_path, aligned_path):
                    raise Exception('Alignment failed')


                with open(aligned_path, 'rb') as f:
                    aligned_b64 = base64.b64encode(f.read()).decode('utf-8')


                save_debug_image(cv2.imread(aligned_path), "04_final_aligned")


                if request_type == 'gemini':
                    for p in parts:
                        if 'inlineData' in p:
                            p['inlineData']['data'] = aligned_b64
                            break
                else:
                    api_data['data'][0]['url'] = f"data:image/png;base64,{aligned_b64}"


            finally:
                for path in [orig_path, result_path, aligned_path]:
                    if path and os.path.exists(path):
                        os.unlink(path)


            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(api_data).encode('utf-8'))
            print(f"[API] {request_type.upper()} request completed successfully\n")


        except Exception as e:
            print(f"[API] Error: {e}")
            self.send_error(500, str(e))


    def log_message(self, format, *args):
        pass




class AdminRequestHandler(BaseHTTPRequestHandler):
    """Handler for Admin port (Port B) - admin panel"""


    def do_GET(self):
        if self.path == '/':
            if self.verify_session():
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(self.get_admin_panel_html().encode('utf-8'))
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(self.get_login_html().encode('utf-8'))


        elif self.path == '/api/keys':
            if not self.verify_session():
                return self.send_error(401, 'Unauthorized')


            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            with API_KEYS_LOCK:
                keys_list = []
                for key, data in API_KEYS.items():
                    is_valid, _ = verify_api_key(key)
                    keys_list.append({
                        'key': key[:8] + '...',
                        'full_key': key,
                        'name': data.get('name', 'Unnamed'),
                        'created_at': data.get('created_at'),
                        'expires_at': data.get('expires_at') or 'Never',
                        'status': 'valid' if is_valid else 'expired'
                    })
                self.wfile.write(json.dumps(keys_list).encode('utf-8'))

        elif self.path.startswith('/api/usage'):
            if not self.verify_session():
                return self.send_error(401, 'Unauthorized')
            
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            period_days = query_params.get('days', ['7'])[0]

            try:
                days = int(period_days)
                start_time = datetime.now() - timedelta(days=days)
            except ValueError:
                return self.send_error(400, 'Invalid days parameter')

            usage_stats = []
            with API_USAGE_LOCK, API_KEYS_LOCK:
                for key, timestamps in API_USAGE.items():
                    recent_timestamps = [t for t in timestamps if datetime.fromisoformat(t) >= start_time]
                    count = len(recent_timestamps)
                    if count > 0:
                        key_info = API_KEYS.get(key)
                        key_name = key_info.get('name', 'Unnamed Key') if key_info else '[DELETED KEY]'
                        usage_stats.append({
                            'name': key_name,
                            'key_prefix': key[:8] + '...',
                            'count': count
                        })
            
            # Sort by count descending
            usage_stats.sort(key=lambda x: x['count'], reverse=True)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(usage_stats).encode('utf-8'))


        else:
            self.send_error(404)


    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return self.send_error(400, 'No content')


        request_data = json.loads(self.rfile.read(content_length))


        if self.path == '/api/login':
            username = request_data.get('username', '')
            password = request_data.get('password', '')


            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == hash_password(password):
                session_token = str(uuid.uuid4())
                SESSION_TOKENS[session_token] = {
                    'username': username,
                    'expires': (datetime.now() + timedelta(hours=24)).isoformat()
                }


                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Set-Cookie', f'session={session_token}; Path=/; HttpOnly; Max-Age=86400')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True}).encode('utf-8'))
                print(f"[ADMIN] Login successful: {username}")
            else:
                print(f"[ADMIN] Login failed: {username}")
                self.send_error(401, 'Invalid credentials')


        elif self.path == '/api/keys/create':
            if not self.verify_session():
                return self.send_error(401, 'Unauthorized')


            name = request_data.get('name', 'Unnamed')
            expires_days = request_data.get('expires_days')


            new_key = str(uuid.uuid4())
            key_data = {'created_at': datetime.now().isoformat(), 'name': name}


            if expires_days:
                expires_at = datetime.now() + timedelta(days=int(expires_days))
                key_data['expires_at'] = expires_at.isoformat()


            with API_KEYS_LOCK:
                API_KEYS[new_key] = key_data
            save_api_keys()


            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'api_key': new_key}).encode('utf-8'))
            print(f"[ADMIN] API key created: {name}")


        elif self.path == '/api/keys/delete':
            if not self.verify_session():
                return self.send_error(401, 'Unauthorized')


            api_key = request_data.get('api_key', '')
            with API_KEYS_LOCK:
                if api_key in API_KEYS:
                    del API_KEYS[api_key]
                    save_api_keys()
                    # We don't delete usage data, it will show up as [DELETED KEY]
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': True}).encode('utf-8'))
                    print(f"[ADMIN] API key deleted")
                else:
                    self.send_error(404, 'Key not found')


        else:
            self.send_error(404)


    def verify_session(self):
        """Verify if user has valid session."""
        cookies = self.headers.get('Cookie', '')
        for cookie in cookies.split(';'):
            if 'session=' in cookie:
                token = cookie.split('=')[1].strip()
                if token in SESSION_TOKENS:
                    try:
                        expires = datetime.fromisoformat(SESSION_TOKENS[token]['expires'])
                        if datetime.now() < expires:
                            return True
                    except:
                        pass
        return False


    def get_login_html(self):
        return """<!DOCTYPE html>
<html>
<head><title>Admin Login</title>
<style>
body{font-family:Arial;background:#f0f0f0;margin:0;padding:0}
.login-container{max-width:300px;margin:100px auto;background:#fff;padding:30px;border-radius:5px;box-shadow:0 0 10px rgba(0,0,0,0.1)}
h2{color:#333;text-align:center}
input{width:100%;padding:10px;margin:10px 0;box-sizing:border-box;border:1px solid #ddd;border-radius:3px}
button{width:100%;padding:10px;background:#4CAF50;color:#fff;border:0;cursor:pointer;border-radius:3px;font-size:16px}
button:hover{background:#45a049}
.error{color:red;margin-top:10px;text-align:center;font-size:14px}
</style></head>
<body>
<div class="login-container">
<h2>Admin Login</h2>
<form onsubmit="login(event)">
<input type="text" id="username" placeholder="Username" required>
<input type="password" id="password" placeholder="Password" required>
<button type="submit">Login</button>
</form>
<div id="error" class="error"></div>
</div>
<script>
function login(e){e.preventDefault();
fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({username:document.getElementById('username').value,password:document.getElementById('password').value})})
.then(r=>r.json()).then(d=>{if(d.success)location.reload();else document.getElementById('error').textContent='Invalid credentials'});}
</script>
</body></html>"""


    def get_admin_panel_html(self):
        return """<!DOCTYPE html>
<html>
<head><title>API Key Management</title>
<style>
body{font-family:Arial;background:#f5f5f5;margin:0;padding:20px}
.container{max-width:1000px;margin:0 auto;background:#fff;padding:30px;border-radius:5px;box-shadow:0 0 5px rgba(0,0,0,0.1)}
h1{color:#333;border-bottom:2px solid #4CAF50;padding-bottom:10px}
h3{margin-top:40px;border-bottom:1px solid #ccc;padding-bottom:8px}
.control-panel{background:#f9f9f9;padding:20px;border-radius:5px;margin-bottom:30px;border:1px solid #e0e0e0; display: flex; align-items: center; gap: 10px;}
.control-panel h3 { margin: 0; padding: 0; border: 0;}
input,select{padding:8px;border:1px solid #ddd;border-radius:3px}
button{padding:8px 15px;background:#4CAF50;color:#fff;border:0;cursor:pointer;border-radius:3px}
button:hover{background:#45a049}
table{width:100%;border-collapse:collapse; margin-top: 15px;}
th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}
th{background:#4CAF50;color:#fff}
tr:hover{background:#f5f5f5}
.copy-btn{background:#2196F3;padding:5px 10px;font-size:0.9em}
.delete-btn{background:#f44336;padding:5px 10px;font-size:0.9em}
.status-valid{color:green;font-weight:bold}
.status-expired{color:red;font-weight:bold}
code{background:#f0f0f0;padding:2px 6px;border-radius:3px;font-family:monospace}
</style></head>
<body>
<div class="container">
<h1>API Key Management</h1>

<div class="control-panel">
<h3>Generate New Key</h3>
<input type="text" id="keyName" placeholder="Key Name (e.g., Client A)">
<select id="expiresDays">
<option value="">No Expiration</option>
<option value="7">7 Days</option>
<option value="30">30 Days</option>
<option value="90">90 Days</option>
<option value="365">1 Year</option>
</select>
<button onclick="createKey()">Generate Key</button>
</div>

<h3>Active API Keys</h3>
<table>
<thead><tr><th>Name</th><th>Key</th><th>Created</th><th>Expires</th><th>Status</th><th>Actions</th></tr></thead>
<tbody id="keysList"></tbody>
</table>

<h3>API Usage Statistics</h3>
<div class="control-panel">
<label for="usagePeriod">Show usage for past:</label>
<select id="usagePeriod" onchange="loadUsage()">
<option value="1">24 Hours</option>
<option value="7" selected>7 Days</option>
<option value="30">30 Days</option>
<option value="180">180 Days</option>
<option value="365">365 Days</option>
</select>
</div>
<table>
<thead><tr><th>Name</th><th>Key Prefix</th><th>Request Count</th></tr></thead>
<tbody id="usageList"></tbody>
</table>

</div>
<script>
function loadKeys(){
fetch('/api/keys').then(r=>r.json()).then(keys=>{
const tbody = document.getElementById('keysList');
tbody.innerHTML='';
keys.forEach(k=>{
const row=tbody.insertRow();
row.innerHTML=`<td>${k.name}</td><td><code>${k.key}</code></td><td>${new Date(k.created_at).toLocaleString()}</td><td>${k.expires_at==='Never'?'Never':new Date(k.expires_at).toLocaleString()}</td><td class="status-${k.status}">${k.status}</td><td><button class="copy-btn" onclick="copyKey('${k.full_key}')">Copy</button><button class="delete-btn" onclick="deleteKey('${k.full_key}')">Delete</button></td>`;});});}

function loadUsage(){
const days = document.getElementById('usagePeriod').value;
fetch(`/api/usage?days=${days}`).then(r=>r.json()).then(usage=>{
const tbody = document.getElementById('usageList');
tbody.innerHTML='';
if (usage.length === 0) {
    const row = tbody.insertRow();
    row.innerHTML = `<td colspan="3" style="text-align:center;">No usage data for this period.</td>`;
} else {
    usage.forEach(u=>{
    const row=tbody.insertRow();
    row.innerHTML=`<td>${u.name}</td><td><code>${u.key_prefix}</code></td><td>${u.count}</td>`;});
}});
}

function createKey(){
const name=document.getElementById('keyName').value||'Unnamed';
const days=document.getElementById('expiresDays').value;
fetch('/api/keys/create',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({name:name,expires_days:days?parseInt(days):null})})
.then(r=>r.json()).then(d=>{if(d.success){alert('Key Generated: '+d.api_key);document.getElementById('keyName').value='';
document.getElementById('expiresDays').value='';loadKeys();}});}

function deleteKey(key){if(confirm('Are you sure you want to delete this key? This cannot be undone.'))
fetch('/api/keys/delete',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({api_key:key})}).then(r=>r.json()).then(d=>{if(d.success){loadKeys(); loadUsage();}});
}

function copyKey(key){navigator.clipboard.writeText(key).then(() => alert('API Key copied to clipboard!'));}

// Initial load
loadKeys();
loadUsage();

// Refresh data periodically
setInterval(() => {
    loadKeys();
    loadUsage();
}, 60000);
</script>
</body></html>"""


    def log_message(self, format, *args):
        pass




class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass




# ============================================================================
# SERVER MANAGEMENT
# ============================================================================


def start_api_server():
    """Start API server on Port A"""
    host = config.get('host', '127.0.0.1')
    port = config.get('port', 8080)
    server = ThreadingHTTPServer((host, port), APIRequestHandler)
    print(f"[API] Server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[API] Shutting down...")




def start_admin_server():
    """Start Admin server on Port B"""
    host = config.get('admin_host', '127.0.0.1')
    port = config.get('admin_port', 8081)
    server = ThreadingHTTPServer((host, port), AdminRequestHandler)
    print(f"[ADMIN] Server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[ADMIN] Shutting down...")




def start_servers():
    """Start both servers in separate threads"""
    load_config()
    load_api_keys()
    load_admin_credentials()
    load_api_usage()


    # Initialize default admin if none exists
    if not ADMIN_CREDENTIALS:
        ADMIN_CREDENTIALS['admin'] = hash_password('admin123')
        save_admin_credentials()
        print("[INIT] Default admin account created (admin / admin123)")


    api_thread = threading.Thread(target=start_api_server, daemon=False)
    admin_thread = threading.Thread(target=start_admin_server, daemon=False)


    api_thread.start()
    admin_thread.start()


    try:
        api_thread.join()
        admin_thread.join()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Gracefully shutting down...")


if __name__ == "__main__":
    start_servers()
