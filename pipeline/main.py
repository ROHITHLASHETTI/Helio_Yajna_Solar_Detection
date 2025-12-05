import os
import json
import pandas as pd
import requests
from ultralytics import YOLO
from shapely.geometry import Point, box
from math import sqrt, pi
from dotenv import load_dotenv
load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
INPUT_FILE = os.getenv("INPUT_FILE", "input_data/input.xlsx")
ARTEFACT_DIR = os.getenv("ARTEFACT_DIR", "output_data/artefacts")
PREDICTION_DIR = os.getenv("PREDICTION_DIR", "output_data/predictions")
os.makedirs(ARTEFACT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

ZOOM_LEVEL = int(os.getenv("ZOOM_LEVEL", "20"))
GSD = float(os.getenv("GSD", "0.15"))
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment. Please create a .env with GOOGLE_API_KEY.")

# Derived constants (areas/radii)
AREA_1200_SQFT_IN_M2 = 1200 * 0.092903
RADIUS_1200_M = sqrt(AREA_1200_SQFT_IN_M2 / pi)  # in meters

AREA_2400_SQFT_IN_M2 = 2400 * 0.092903
RADIUS_2400_M = sqrt(AREA_2400_SQFT_IN_M2 / pi)  # in meters


def fetch_image(sample_id, lat, lon, output_dir, zoom=ZOOM_LEVEL, timeout=12):
    """Downloads image from Google Static Maps and returns local path (or None)."""
    filename = f"{sample_id}.jpg"
    path = os.path.join(output_dir, filename)

    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "size": "640x640",
        "maptype": "satellite",
        "key": GOOGLE_API_KEY,
    }

    print(f"Fetching image for {sample_id} at ({lat}, {lon}) — zoom {zoom} ...")
    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image for {sample_id}: {e}")
        return None


def process_detection(results, gsd=GSD):
    """
    Applies the Challenge Logic:
      - Check 1200 sqft buffer (center of image)
      - If fails, check 2400 sqft buffer
      - Compute estimated PV area in m^2
    Expects results returned by ultralytics YOLO predict.
    Returns: has_solar (bool), detected_buffer (0/1200/2400), pv_area_sqm (float), best_bbox (list or None)
    """
    # Image center for 640x640 static map
    center_x, center_y = 320, 320

    # Convert meters radius to pixels: radius_pixels = radius_meters / GSD
    radius_1200_px = RADIUS_1200_M / gsd
    radius_2400_px = RADIUS_2400_M / gsd

    buffer_1200 = Point(center_x, center_y).buffer(radius_1200_px)
    buffer_2400 = Point(center_x, center_y).buffer(radius_2400_px)

    best_panel = None
    max_overlap = 0
    detected_buffer = 0

    # If there are no boxes, results[0].boxes may be empty
    boxes = []
    try:
        
        if len(results) == 0:
            boxes = []
        else:
            
            xyxy = results[0].boxes.xyxy.tolist() if hasattr(results[0].boxes, "xyxy") else []
            confs = results[0].boxes.conf.tolist() if hasattr(results[0].boxes, "conf") else []
            
            for i, b in enumerate(xyxy):
                c = float(confs[i]) if i < len(confs) else 0.0
                boxes.append((b[0], b[1], b[2], b[3], c))
    except Exception:
        boxes = []
    for b in boxes:
        x1, y1, x2, y2, conf = b
        panel_box = box(x1, y1, x2, y2)
        if panel_box.intersects(buffer_1200):
            intersection = panel_box.intersection(buffer_1200).area
            if intersection > 0 and intersection > max_overlap:
                max_overlap = intersection
                best_panel = panel_box
                detected_buffer = 1200
    if best_panel is None:
        for b in boxes:
            x1, y1, x2, y2, conf = b
            panel_box = box(x1, y1, x2, y2)
            if panel_box.intersects(buffer_2400):
                intersection = panel_box.intersection(buffer_2400).area
                if intersection > max_overlap:
                    max_overlap = intersection
                    best_panel = panel_box
                    detected_buffer = 2400

    has_solar = best_panel is not None
    pv_area_sqm = 0.0
    best_bbox = None
    if has_solar:
        pv_area_sqm = best_panel.area * (gsd ** 2)
        minx, miny, maxx, maxy = best_panel.bounds
        best_bbox = [minx, miny, maxx, maxy]

    return has_solar, detected_buffer, round(pv_area_sqm, 2), best_bbox


def main():
    print("Loading YOLO model from:", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please check MODEL_PATH in .env or place model there.")
        return

    model = YOLO(MODEL_PATH)

  
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}. Please provide input Excel at this path.")
        return

    df = pd.read_excel(INPUT_FILE)

  
    if not {'sample_id', 'latitude', 'longitude'}.issubset(set(df.columns)):
        print("Input file must contain 'sample_id', 'latitude', 'longitude' columns.")
        return

    for _, row in df.iterrows():
        s_id = row['sample_id']
        lat, lon = row['latitude'], row['longitude']
        print(f"\nProcessing sample: {s_id}")

        img_path = fetch_image(s_id, lat, lon, ARTEFACT_DIR)
        if not img_path:
            print(f"Skipping {s_id}: could not download image.")
            continue

        
        try:
            results = model.predict(source=img_path, conf=0.1, verbose=False)
        except Exception as e:
            print(f"Prediction error for {s_id}: {e}")
            continue

        # Logic and area calculation
        has_solar, buffer_sz, area, best_bbox = process_detection(results, gsd=GSD)

       
        try:
            results[0].save(filename=os.path.join(ARTEFACT_DIR, f"{s_id}_overlay.jpg"))
        except Exception:
            
            pass
        confidence_pct = 0.0
        try:
            if has_solar and hasattr(results[0].boxes, "conf"):
                # find closest box by bbox overlap
                confs = results[0].boxes.conf.tolist()
                xyxy = results[0].boxes.xyxy.tolist()
                best_conf = 0.0
                for i, b in enumerate(xyxy):
                    
                    bx = [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
                    if best_bbox:
                        bbx = [(best_bbox[0] + best_bbox[2]) / 2.0, (best_bbox[1] + best_bbox[3]) / 2.0]
                       
                        d = ((bx[0] - bbx[0])**2 + (bx[1] - bbx[1])**2) ** 0.5
                        if d < 50:  
                            best_conf = max(best_conf, float(confs[i]))
                confidence_pct = float(best_conf) * 100.0
        except Exception:
            confidence_pct = 0.0

   
        output_json = {
            "sample_id": s_id,
            "lat": float(lat),
            "lon": float(lon),
            "has_solar": bool(has_solar),
            "confidence_pct": round(confidence_pct, 2),
            "pv_area_sqm_est": area,
            "buffer_radius_sqft": buffer_sz,
            "qc_status": "VERIFIABLE",
            "best_bbox": best_bbox or [],
            "image_metadata": {"source": "Google Static Maps", "zoom": ZOOM_LEVEL},
        }

        out_json_path = os.path.join(PREDICTION_DIR, f"{s_id}.json")

        with open(out_json_path, "w", encoding="utf-8") as jf:
            json.dump(output_json, jf, indent=2)

        print(f"Saved results to {out_json_path}")

    print("\n✅ Batch Processing Complete. Check output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
