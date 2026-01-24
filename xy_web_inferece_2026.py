import os, shutil, tempfile, re, sys
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np

from agcem_test_affectnet_ROI_web import (
    process_image, BASE_OUTPUT_DIR
)

def crop_face_to_224_inplace(img_path: str, size: int = 224, margin: float = 0.25) -> None:
    """
    Load image at img_path, crop a square region around the largest detected face,
    expand by `margin`, clamp to image bounds, resize to `size`×`size`, and
    overwrite img_path with the cropped image. If no face is detected, do a
    center square crop then resize to `size`.
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    h, w = img.shape[:2]

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        side = int(max(fw, fh) * (1.0 + margin))
        x1 = int(cx - side / 2)
        y1 = int(cy - side / 2)
        x2 = x1 + side
        y2 = y1 + side
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        cw = x2 - x1
        ch = y2 - y1
        side = max(cw, ch)
        x1 = int(max(0, min(w - side, int(cx - side / 2))))
        y1 = int(max(0, min(h - side, int(cy - side / 2))))
        x2 = x1 + side
        y2 = y1 + side
        crop = img[y1:y2, x1:x2]
    else:
        side = min(w, h)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        crop = img[y1:y1 + side, x1:x1 + side]
    crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ext = os.path.splitext(img_path)[1].lower()
    ext = ".jpg" if ext not in [".jpg", ".jpeg", ".png", ".bmp"] else ext
    success, buf = cv2.imencode(ext, crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise RuntimeError("Failed to encode cropped image")
    buf.tofile(img_path)

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
def favicon():
    ico_path = "static/favicon.ico"
    if os.path.exists(ico_path):
        return FileResponse(ico_path)
    return HTMLResponse(content="", status_code=204)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>AGCM Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background:#f8f9fa; color:#333; }
            .container { max-width: 900px; margin:auto; background:#fff; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.1); }
            input[type=file] { margin-top:10px; }
            button { background:#007bff; color:#fff; border:none; padding:10px 20px; margin-top:10px; border-radius:6px; cursor:pointer; }
            button:hover { background:#0056b3; }
            .result { margin-top:20px; padding:20px; border:1px solid #ddd; border-radius:8px; background:#fafafa; }
            .muted { color:#777; font-size:0.95em; }
            pre { background:#f0f0f0; padding:10px; border-radius:6px; overflow-x:auto; }
            .block { margin:18px 0; }
            .title { margin:6px 0 10px 0; font-weight:600; }
            .img-full { width: 320px; max-width:100%; border-radius:8px; display:block; margin:0 auto; } /* smaller weighted map */
            .au-item { margin:14px 0; text-align:center; }
            .au-img { width:200px; border-radius:8px; display:block; margin:0 auto 6px auto; }
            .au-label { font-size:0.9em; color:#555; }
            .warn { color:#a33; font-size:0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">Interpretable Concept-based Deep Learning Framework for Multimodal Human Behavior Modeling</h1>
            <p style="text-align: center;"> Xinyu Li, Marwa Mahmoud, School of Computing Science, University of Glasgow</p>
            <h2>Attention-Guided Concept Model</h2>
            <p>
              Please upload an AffectNet-style cropped aligned facial image; otherwise it will be automatically cropped to a 224×224 face (using OpenCV). 
              Then, we show the AGCM task prediction, concept prediction, the weighted concept map, and AU heatmaps.  
              <br><br>
              Please feel free to try any facial images from or not from FER datasets (e.g., AffectNet). 
              Here we provide some <a href="https://github.com/xinyuli-uofg/agcm/tree/main/AffectNet" target="_blank"><strong>sample images from AffectNet</strong></a> 
              to help you reproduce the results of our paper.  
              <br>
              For access to the official AffectNet dataset, please visit the 
              <a href="https://www.mohammadmahoor.com/pages/databases/affectnet/" target="_blank"><strong>AffectNet website</strong></a>.
              <br><br>
              You can also try one of the following sample images (random examples from the internet; <strong>not from the FER datasets</strong>):
              <ul>
                <li>
                  <a href="https://media.istockphoto.com/id/1045297442/photo/frustrated-and-worried-young-man-portrait-in-grey-t-shirt.jpg?s=612x612&w=0&k=20&c=7bE3TZ0L-rMYe3h3n0bFJkSmu8KgPIkM-qajf0claDI=" download target="_blank">
                    Download sample image (male, frustrated expression)
                  </a>
                </li>
                <li>
                  <a href="https://www.freepik.com/free-photo/pretty-beautiful-woman-with-blonde-long-hair-having-excited-happy-facial-expression_9116613.htm#fromView=keyword&page=1&position=0&uuid=4d376c6a-b79c-4187-b1f5-c2e31b71baeb&query=Happy+woman" target="_blank">
                    Download sample image (female, happy expression)
                  </a>
                </li>
              </ul>
            </p>
            <input type="file" id="fileInput" accept="image/*"/>
            <button onclick="send()">Run Inference</button>

            <div id="result" class="result" style="display:none;">
                <div class="block">
                    <h3 class="title">Predicted Emotion</h3>
                    <p id="pred" class="muted"></p>
                </div>

                <div class="block">
                    <h3 class="title">AU Scores</h3>
                    <pre id="auTable"></pre>
                </div>

                <div class="block" id="weightedBlock" style="display:none;">
                    <h3 class="title">Weighted Concept Attention Map</h3>
                    <img id="weightedImg" class="img-full" alt="Weighted concept attention map"/>
                    <div id="weightedWarn" class="warn" style="display:none;"></div>
                </div>

                <div class="block">
                    <h3 class="title">Action Unit Heatmaps</h3>
                    <div id="heatmaps"></div>
                </div>
            </div>
        </div>

        <script>
        async function send() {
            const fileInput   = document.getElementById('fileInput');
            const resultDiv   = document.getElementById('result');
            const predText    = document.getElementById('pred');
            const auTable     = document.getElementById('auTable');
            const weightedBlk = document.getElementById('weightedBlock');
            const weightedImg = document.getElementById('weightedImg');
            const weightedWarn= document.getElementById('weightedWarn');
            const heatmaps    = document.getElementById('heatmaps');

            if (!fileInput.files.length) {
                alert('Please select an image first!');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            // reset UI
            predText.innerText   = 'Processing…';
            auTable.textContent  = '';
            heatmaps.innerHTML   = '';
            weightedWarn.style.display = 'none';
            weightedWarn.textContent   = '';
            weightedBlk.style.display  = 'none';
            weightedImg.src            = '';
            resultDiv.style.display    = 'block';

            try {
                const res  = await fetch('/predict/', { method: 'POST', body: formData });
                const text = await res.text();   // raw first (handles non-JSON server errors)
                let data;
                try { data = JSON.parse(text); } catch (e) { data = null; }

                if (!res.ok) throw new Error((data && data.error) || text || 'Server error');
                if (!data) throw new Error('Empty response');

                // 1) Predicted emotion
                predText.innerText = 'Predicted Emotion: ' + (data.prediction || 'unknown');

                // 2) AU table
                auTable.textContent = (data.au_table && data.au_table.length)
                    ? data.au_table.join('\\n')
                    : 'No AU table found.';

                // 3) Weighted concept attention map — reveal on load only
                if (data.weighted_map) {
                    console.log('[weighted_map URL]', data.weighted_map);
                    weightedImg.onload = () => {
                        weightedWarn.style.display = 'none';
                        weightedBlk.style.display  = 'block';
                        console.log('[weighted_map] loaded');
                    };
                    weightedImg.onerror = () => {
                        weightedWarn.textContent   = 'Could not load the weighted map image.';
                        weightedWarn.style.display = 'block';
                        weightedBlk.style.display  = 'none';
                        console.warn('[weighted_map] failed to load:', data.weighted_map);
                    };
                    weightedImg.src = data.weighted_map;  // cache-busting param included from backend
                } else {
                    console.warn('[weighted_map] not provided by backend');
                    weightedBlk.style.display = 'none';
                }

                // 4) Each AU heatmap one-by-one
                if (Array.isArray(data.heatmap_rows)) {
                    console.log('[heatmap_rows]', data.heatmap_rows.length);
                    data.heatmap_rows.forEach(row => {
                        if (!row || !row.path) return;
                        const wrap = document.createElement('div');
                        wrap.className = 'au-item';

                        const label = document.createElement('div');
                        label.className = 'au-label';
                        label.textContent = (row.aus && row.aus.length) ? row.aus[0] : '';
                        wrap.appendChild(label);

                        const img = document.createElement('img');
                        img.src = row.path;
                        img.className = 'au-img';
                        img.alt = label.textContent || 'AU heatmap';
                        wrap.appendChild(img);

                        heatmaps.appendChild(wrap);
                    });
                }

            } catch (err) {
                predText.innerText = '❌ Error: ' + (err && err.message ? err.message : err);
                console.error(err);
            }
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        tmp_dir  = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, file.filename)
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        crop_face_to_224_inplace(tmp_path, size=224, margin=0.25)
        stem = os.path.splitext(os.path.basename(tmp_path))[0]
        process_image(tmp_path, k=5)
        matches = [d for d in os.listdir(BASE_OUTPUT_DIR) if d.startswith(stem)]
        if not matches:
            return JSONResponse({"error": f"No result folder found for {stem}."}, status_code=404)
        pred_dir = max(matches, key=lambda d: os.path.getmtime(os.path.join(BASE_OUTPUT_DIR, d)))
        out_dir  = os.path.join(BASE_OUTPUT_DIR, pred_dir)
        img_dir  = os.path.join(out_dir, "images")
        result_txt = os.path.join(out_dir, "results.txt")
        pred_label = "unknown"
        au_lines = []
        if os.path.exists(result_txt):
            with open(result_txt, "r") as f:
                lines = f.readlines()
            pred_indices = [i for i, l in enumerate(lines) if l.strip().startswith("Pred class")]
            if pred_indices:
                start = pred_indices[-1]
                pred_label = lines[start].split(":")[-1].strip()
                for l in lines[start + 1:]:
                    stripped = l.strip()
                    if re.match(r"^(Upper|Lower):", stripped):
                        au_lines.append(stripped)
                    elif stripped.startswith(("AU name", "----", "")):
                        continue
                    elif stripped.startswith("Images saved") or stripped.startswith("Pred class"):
                        break
        weighted_map_url = None
        weighted_map_file = f"{stem}_norm_pred.jpg"
        weighted_map_path = os.path.join(img_dir, weighted_map_file)
        if os.path.exists(weighted_map_path):
            static_map = f"static/{weighted_map_file}"
            shutil.copy(weighted_map_path, static_map)
            weighted_map_url = f"/{static_map}?t={int(os.path.getmtime(weighted_map_path))}"
            print("[weighted_map] ->", weighted_map_url)
        else:
            print("[weighted_map] not found at", weighted_map_path)
        CONCEPT_MAP = [
            "Upper:FAU1","Upper:FAU2","Upper:FAU4","Upper:FAU5","Upper:FAU6","Upper:FAU7",
            "Lower:FAU9","Lower:FAU10","Lower:FAU12","Lower:FAU14","Lower:FAU15","Lower:FAU17",
            "Lower:FAU20","Lower:FAU23","Lower:FAU25","Lower:FAU26","Lower:FAU28","Upper:FAU45",
        ]
        heatmap_rows = []
        for au_name in CONCEPT_MAP:
            au_tag  = au_name.split(":")[-1]
            au_file = f"{stem}_au_pred_{au_tag.replace('FAU','AU')}.jpg"
            au_path = os.path.join(img_dir, au_file)
            if os.path.exists(au_path):
                static_file = f"static/{au_file}"
                shutil.copy(au_path, static_file)
                heatmap_rows.append({
                    "path": f"/{static_file}?t={int(os.path.getmtime(au_path))}",
                    "aus": [au_name]
                })
        return {
            "prediction": pred_label,
            "au_table": au_lines,
            "weighted_map": weighted_map_url,
            "heatmap_rows": heatmap_rows
        }
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
