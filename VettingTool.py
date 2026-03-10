import sys
import pandas as pd
import os
import re
import json
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u
from PIL import Image

# CONFIGURATION 
BASE_DIR = "/home/mar930/Documents/EMUSE BT Galaxies/FINAL CATALOGUE OF BT GALAXIES"
CSV_PATH = os.path.join(BASE_DIR, "combined_class_1s.csv")
HARD_DRIVE_PATH = "/media/mar930/Internship/images"
OUTPUT_HTML = os.path.join(BASE_DIR, "morphology_vetting_tool.html")

# KEEP SAME KEY TO PRESERVE PROGRESS
STORAGE_KEY = "morphology_vetting_class1_binning_v2_RESET"

IMAGE_DIR = os.path.join(BASE_DIR, "vetting_images")

PROCESS_LIMIT = None
CUTOUT_SIZE = (150, 150)
CLIP_LEVELS = [99.9, 99.5, 99.0]

warnings.filterwarnings('ignore')

def remove_duplicates(df):
    print("\n--- DUPLICATE CHECK (COORDINATES ONLY) ---", flush=True)
    if len(df) == 0:
        print("CSV is empty.", flush=True)
        return df
        
    coords = SkyCoord(ra=df['RA'].values*u.deg, dec=df['Dec'].values*u.deg)
    idx1, idx2, d2d, _ = coords.search_around_sky(coords, 2*u.arcsec)
    coord_drop_indices = set()
    for i, j in zip(idx1, idx2):
        if i < j: coord_drop_indices.add(j)
    if coord_drop_indices:
        df_final = df.drop(df.index[list(coord_drop_indices)])
    else:
        df_final = df
    print(f"Removed {len(coord_drop_indices)} physical duplicates.", flush=True)
    return df_final

def get_file_map(directory):
    print(f"--- Deep Scanning Hard Drive: {directory} ---", flush=True)
    if not os.path.exists(directory):
        print(f"CRITICAL ERROR: Hard drive path does not exist: {directory}", flush=True)
        return {}
        
    file_map = {}
    sb_pattern = re.compile(r"SB(\d+)")
    
    count = 0
    # Add a progress counter
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".fits"):
                count += 1
                if count % 2000 == 0:
                    print(f"Scanned {count} files...", end='\r', flush=True)
                    
                match = sb_pattern.search(f)
                if match:
                    sb_id_str = match.group(1)
                    full_path = os.path.join(root, f)
                    if sb_id_str in file_map: file_map[sb_id_str].append(full_path)
                    else: file_map[sb_id_str] = [full_path]
                    
    print(f"\nScan complete. Found {count} FITS files in {len(file_map)} SB groups.", flush=True)
    return file_map

def find_correct_file(sb_id_str, ra, dec, file_map):
    if sb_id_str not in file_map: return None
    candidate_files = file_map[sb_id_str]
    if len(candidate_files) == 1: return candidate_files[0]
    for f_path in candidate_files:
        try:
            with fits.open(f_path) as hdul:
                header = hdul[0].header
                wcs = WCS(header).celestial if WCS(header).naxis > 2 else WCS(header)
                px, py = wcs.all_world2pix(ra, dec, 1)
                naxis1, naxis2 = header.get('NAXIS1'), header.get('NAXIS2')
                if (0 <= px <= naxis1) and (0 <= py <= naxis2): return f_path
        except: continue
    return candidate_files[0]

def create_rgb_png(fits_path, output_png, ra, dec):
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header).celestial if WCS(header).naxis > 2 else WCS(header)
            data = data.squeeze()
            while data.ndim > 2: data = data[0]
            position = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
            cutout = Cutout2D(data, position, CUTOUT_SIZE, wcs=wcs)
            r_data = np.nan_to_num(cutout.data)
            img = np.zeros((r_data.shape[0], r_data.shape[1], 3), dtype=np.uint8)
            for clipid, maxclip in enumerate(CLIP_LEVELS):
                vmin, vmax = np.nanpercentile(r_data, [50, maxclip])
                dataclips = np.clip(r_data, vmin, vmax)
                if vmax - vmin == 0: data_chnl = np.zeros_like(dataclips)
                else: data_chnl = (dataclips - vmin) / (vmax - vmin)
                img[:, :, len(CLIP_LEVELS) - clipid - 1] = (255 * data_chnl).astype(np.uint8)
            Image.fromarray(np.flipud(img), "RGB").save(output_png)
            return True
    except: return False

def generate_html(valid_records, sb_col_name):
    print(f"\nGenerating Interface for {len(valid_records)} sources...", flush=True)
    all_data_json = json.dumps(valid_records)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Morphology Binning Tool</title>
        <style>
            body {{ background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; height: 100vh; overflow: hidden; }}

            /* SIDEBAR */
            .sidebar {{ width: 300px; background: #1e1e1e; padding: 20px; border-right: 1px solid #333; display: flex; flex-direction: column; overflow-y: auto; }}
            .sidebar h2 {{ color: #e91e63; margin-top: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 1.2em; }}

            .stats {{ margin-bottom: 20px; background: #2c2c2c; padding: 15px; border-radius: 8px; border-left: 4px solid #e91e63; }}
            .stat-item {{ display: flex; justify-content: space-between; margin: 5px 0; font-size: 0.9em; }}

            .controls-help {{ font-size: 0.85em; color: #aaa; margin-bottom: 15px; background: #252525; padding: 10px; border-radius: 6px; }}
            .controls-help p {{ margin: 4px 0; display: flex; align-items: center; }}
            
            /* Reference Images Section */
            .example-section {{ margin-bottom: 20px; }}
            .example-section h3 {{ font-size: 0.9em; color: #888; text-transform: uppercase; margin-bottom: 8px; border-bottom: 1px solid #444; padding-bottom: 5px; }}
            .btn-ex {{ 
                width: 100%; 
                padding: 6px 10px; 
                margin-bottom: 6px; 
                background: #252525; 
                border: 1px solid #444; 
                color: #ccc; 
                text-align: left; 
                cursor: pointer; 
                border-radius: 4px;
                font-size: 0.85em;
                display: flex;
                align-items: center;
                transition: background 0.2s;
            }}
            .btn-ex:hover {{ background: #333; color: white; }}
            .color-tag {{ width: 8px; height: 8px; display: inline-block; margin-right: 8px; border-radius: 50%; }}

            /* Push buttons to bottom */
            .spacer {{ flex-grow: 1; }}

            .download-section {{ margin-top: 10px; }}
            button.main-btn {{ width: 100%; padding: 10px; margin-bottom: 8px; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; text-align: left; transition: all 0.2s; font-size: 0.85em; }}
            button.main-btn:hover {{ filter: brightness(1.2); }}

            /* NEW COLOR SCHEME FOR 5 CLASSES */
            .btn-1 {{ background: #29b6f6; color: black; }} /* WAT - Light Blue */
            .btn-2 {{ background: #66bb6a; color: black; }} /* NAT - Green */
            .btn-3 {{ background: #ffa726; color: black; }} /* S/Z - Orange */
            .btn-4 {{ background: #ab47bc; color: white; }} /* X - Purple */
            .btn-0 {{ background: #ef5350; color: white; }} /* Unsure - Red */

            .btn-master {{ background: #d81b60; color: white; padding: 15px; font-size: 1em; text-align: center; }}
            .btn-undo {{ background: transparent; color: #aaa; border: 1px solid #e91e63; margin-bottom: 20px; text-align: center; width: 100%; padding: 10px; cursor: pointer; }}
            .btn-undo:hover {{ background: #e91e631a; color: #e91e63; }}

            .btn-reset {{ background: transparent; color: #777; border: 1px solid #444; margin-top: 10px; text-align: center; font-size: 0.8em; width: 100%; padding: 8px; cursor: pointer; }}
            .btn-reset:hover {{ border-color: #ff5252; color: #ff5252; }}

            /* MAIN VIEW */
            .main-content {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; }}

            .image-box {{
                position: relative;
                width: 800px;
                height: 800px;
                background: #000;
                border: 2px solid #444;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 0 20px rgba(0,0,0,0.5);
            }}
            .image-box img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                image-rendering: pixelated;
                image-rendering: crisp-edges;
            }}

            .metadata {{ position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 15px; border-radius: 4px; pointer-events: none; border-left: 3px solid #e91e63; }}
            .feedback-overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 80px; font-weight: bold; opacity: 0; transition: opacity 0.2s; pointer-events: none; text-shadow: 0 0 10px black; text-align: center; width: 100%; }}

            .progress-bar {{ position: absolute; top: 0; left: 0; height: 5px; background: #e91e63; width: 0%; transition: width 0.3s; }}

            /* EXTERNAL LINKS SECTION */
            .links-container {{
                margin-top: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 10px;
            }}

            .external-link {{ 
                color: #e91e63; 
                font-size: 1.1em; 
                text-decoration: none; 
                border-bottom: 1px solid #e91e63; 
                font-weight: bold; 
                padding-bottom: 2px;
                transition: color 0.2s;
            }}
            .external-link:hover {{ color: #ff80ab; border-color: #ff80ab; }}

            .emu-link {{
                color: #29b6f6; /* Light Blue to distinguish from Aladin */
                border-color: #29b6f6;
            }}
            .emu-link:hover {{ color: #81d4fa; border-color: #81d4fa; }}

            .copy-btn {{
                background: #333;
                border: 1px solid #555;
                color: #ddd;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9em;
                margin-left: 10px;
                transition: background 0.2s;
            }}
            .copy-btn:hover {{ background: #555; }}
            .copy-btn:active {{ background: #29b6f6; color: black; }}

            /* MODAL STYLES */
            .modal-overlay {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.85); align-items: center; justify-content: center; }}
            .modal-content {{ position: relative; background-color: #1e1e1e; margin: auto; padding: 20px; border: 1px solid #444; width: auto; max-width: 80%; max-height: 80%; box-shadow: 0 4px 8px rgba(0,0,0,0.5); text-align: center; border-radius: 8px; }}
            .modal-content img {{ max-width: 100%; max-height: 70vh; width: auto; height: auto; object-fit: contain; margin-top: 15px; border: 1px solid #333; }}
            .close-btn {{ position: absolute; top: 10px; right: 15px; color: #aaa; font-size: 28px; font-weight: bold; cursor: pointer; transition: color 0.2s; line-height: 20px; }}
            .close-btn:hover {{ color: #e91e63; }}
            .modal-title {{ margin: 0; color: #e0e0e0; font-weight: normal; letter-spacing: 1px; }}

        </style>
    </head>
    <body>

    <div id="exampleModal" class="modal-overlay">
        <div class="modal-content">
            <span class="close-btn" onclick="closeModal()">&times;</span>
            <h3 id="modal-title" class="modal-title">Example</h3>
            <img id="modal-image" src="" alt="Example Image">
        </div>
    </div>

    <div class="sidebar">
        <h2>Morphology Binning</h2>

        <div class="stats">
            <div class="stat-item"><span>Total:</span> <span id="stat-total">{len(valid_records)}</span></div>
            <div class="stat-item"><span>Done:</span> <span id="stat-done">0</span></div>
            <hr style="border-color: #444">
            <div class="stat-item" style="color:#29b6f6"><span>[1] WAT:</span> <span id="count-1">0</span></div>
            <div class="stat-item" style="color:#66bb6a"><span>[2] NAT:</span> <span id="count-2">0</span></div>
            <div class="stat-item" style="color:#ffa726"><span>[3] S/Z:</span> <span id="count-3">0</span></div>
            <div class="stat-item" style="color:#ab47bc"><span>[4] X:</span> <span id="count-4">0</span></div>
            <div class="stat-item" style="color:#ef5350"><span>[0] Unsure:</span> <span id="count-0">0</span></div>
        </div>

        <div class="controls-help">
            <p><strong>Keyboard Controls:</strong></p>
            <p><span style="color:#29b6f6">[1]</span> Wide-Angle Tailed</p>
            <p><span style="color:#66bb6a">[2]</span> Narrow-Angle Tailed</p>
            <p><span style="color:#ffa726">[3]</span> S or Z Shaped</p>
            <p><span style="color:#ab47bc">[4]</span> X Shaped</p>
            <p><span style="color:#ef5350">[0]</span> Unsure / Not Qual</p>
            <p><span>[Bksp]</span> Undo Last</p>
        </div>

        <div class="example-section">
            <h3>Reference Images</h3>
            <button class="btn-ex" onclick="showExample('WAT')">
                <span class="color-tag" style="background:#29b6f6"></span> Show WAT Example
            </button>
            <button class="btn-ex" onclick="showExample('NAT')">
                <span class="color-tag" style="background:#66bb6a"></span> Show NAT Example
            </button>
            <button class="btn-ex" onclick="showExample('S_and_Z')">
                <span class="color-tag" style="background:#ffa726"></span> Show S/Z Example
            </button>
            <button class="btn-ex" onclick="showExample('X_Shaped')">
                <span class="color-tag" style="background:#ab47bc"></span> Show X-Shape Example
            </button>
        </div>

        <button class="btn-undo" onclick="undoLast()">⟲ Undo Last</button>
        <button class="btn-reset" onclick="resetProgress()">⚠ Reset All Progress</button>

        <div class="spacer"></div>

        <div class="download-section">
            <button class="main-btn btn-master" id="btn-master" onclick="downloadMaster()">Download Completed</button>
            <button class="main-btn btn-1" onclick="downloadCSV(1)">Download WAT (1)</button>
            <button class="main-btn btn-2" onclick="downloadCSV(2)">Download NAT (2)</button>
            <button class="main-btn btn-3" onclick="downloadCSV(3)">Download S/Z (3)</button>
            <button class="main-btn btn-4" onclick="downloadCSV(4)">Download X (4)</button>
            <button class="main-btn btn-0" onclick="downloadCSV(0)">Download Unsure (0)</button>
        </div>
    </div>

    <div class="main-content">
        <div class="progress-bar" id="progress-bar"></div>
        <div class="image-box">
            <img id="main-image" src="" alt="Candidate">
            <div class="metadata" id="metadata"></div>
            <div class="feedback-overlay" id="feedback"></div>
        </div>
        
        <div class="links-container">
            <a id="aladin-link" href="#" target="_blank" class="external-link">View in Aladin</a>
            
            <div style="display: flex; align-items: center;">
                <a href="https://emu-survey.org/progress/aladin.html" target="_blank" class="external-link emu-link">View in EMU</a>
                <button class="copy-btn" onclick="copyCoords()">Copy Coords</button>
            </div>
        </div>
    </div>

    <script>
        const allData = {all_data_json};
        const sbKey = "{sb_col_name}";
        const storageKey = "{STORAGE_KEY}";

        let currentIndex = 0;
        let decisions = JSON.parse(localStorage.getItem(storageKey)) || {{}};

        window.onload = function() {{ findFirstUnvetted(); updateUI(); }};

        function findFirstUnvetted() {{
            for(let i=0; i<allData.length; i++) {{
                let rowId = String(allData[i].Original_Row_ID);
                if(!decisions[rowId]) {{ currentIndex = i; return; }}
            }}
            currentIndex = allData.length - 1;
        }}

        function loadCandidate(index) {{
            if(index < 0 || index >= allData.length) return;
            const item = allData[index];
            const ra = parseFloat(item.RA).toFixed(4);
            const dec = parseFloat(item.Dec).toFixed(4);
            document.getElementById('main-image').src = item.vet_img_path;
            
            // Auto-fill Aladin URL if present
            if(item.aladin_url) {{
                document.getElementById('aladin-link').href = item.aladin_url;
            }} else {{
                document.getElementById('aladin-link').href = '#';
            }}
            
            document.getElementById('metadata').innerHTML = `
                <strong>SB: ${{item[sbKey]}}</strong><br>
                Row: ${{item.Original_Row_ID}}<br>
                Prob: ${{item.Probability || item.Prob || 'N/A'}}<br>
                RA: ${{ra}} <br> Dec: ${{dec}}
            `;
        }}

        function copyCoords() {{
            if(currentIndex < 0 || currentIndex >= allData.length) return;
            const item = allData[currentIndex];
            const text = `${{item.RA}} ${{item.Dec}}`; // Space separated for easy pasting
            
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.querySelector('.copy-btn');
                const originalText = btn.innerText;
                btn.innerText = "Copied!";
                btn.style.background = "#66bb6a";
                btn.style.color = "black";
                setTimeout(() => {{
                    btn.innerText = originalText;
                    btn.style.background = "#333";
                    btn.style.color = "#ddd";
                }}, 1500);
            }}).catch(err => {{
                console.error('Failed to copy: ', err);
            }});
        }}

        function makeDecision(value) {{
            if(currentIndex >= allData.length) return;
            const item = allData[currentIndex];
            const rowId = String(item.Original_Row_ID);
            decisions[rowId] = value;
            localStorage.setItem(storageKey, JSON.stringify(decisions));
            showFeedback(value);
            currentIndex++;
            if(currentIndex >= allData.length) alert("End of list!");
            updateUI();
        }}

        function undoLast() {{
            if(currentIndex > 0) {{
                currentIndex--;
                const item = allData[currentIndex];
                delete decisions[String(item.Original_Row_ID)];
                localStorage.setItem(storageKey, JSON.stringify(decisions));
                updateUI();
            }}
        }}

        function resetProgress() {{
            if(confirm("Are you sure you want to RESET ALL progress? This cannot be undone.")) {{
                decisions = {{}};
                localStorage.removeItem(storageKey);
                currentIndex = 0;
                updateUI();
                alert("Progress reset.");
            }}
        }}

        function updateUI() {{ loadCandidate(currentIndex); updateStats(); }}

        function updateStats() {{
            const total = allData.length;
            const doneCount = Object.keys(decisions).length;
            let c0=0, c1=0, c2=0, c3=0, c4=0;
            Object.values(decisions).forEach(v => {{
                if(v == 1) c1++;
                else if(v == 2) c2++;
                else if(v == 3) c3++;
                else if(v == 4) c4++;
                else if(v == 0) c0++;
            }});
            document.getElementById('stat-done').innerText = doneCount;
            document.getElementById('count-1').innerText = c1;
            document.getElementById('count-2').innerText = c2;
            document.getElementById('count-3').innerText = c3;
            document.getElementById('count-4').innerText = c4;
            document.getElementById('count-0').innerText = c0;

            const pct = (doneCount / total) * 100;
            document.getElementById('progress-bar').style.width = pct + "%";
        }}

        function showFeedback(val) {{
            const fb = document.getElementById('feedback');
            let text = "", color = "";

            if(val == 1) {{ text = "WAT"; color = "#29b6f6"; }}
            else if(val == 2) {{ text = "NAT"; color = "#66bb6a"; }}
            else if(val == 3) {{ text = "S/Z Shape"; color = "#ffa726"; }}
            else if(val == 4) {{ text = "X Shape"; color = "#ab47bc"; }}
            else if(val == 0) {{ text = "Unsure / Not Qual"; color = "#ef5350"; }}

            fb.innerText = text; fb.style.color = color; fb.style.opacity = 1;
            setTimeout(() => {{ fb.style.opacity = 0; }}, 500);
        }}

        document.addEventListener('keydown', function(event) {{
            if(event.key === '1') makeDecision(1);
            if(event.key === '2') makeDecision(2);
            if(event.key === '3') makeDecision(3);
            if(event.key === '4') makeDecision(4);
            if(event.key === '0') makeDecision(0);
            if(event.key === 'Backspace') undoLast();
            if(event.key === 'Escape') closeModal();
        }});

        function convertToCSV(objArray) {{
            const array = typeof objArray != 'object' ? JSON.parse(objArray) : objArray;
            let str = Object.keys(array[0]).join(',') + '\\r\\n';
            for (let i = 0; i < array.length; i++) {{
                let line = '';
                for (let index in array[i]) {{
                    if (line != '') line += ',';
                    let item = array[i][index];
                    if (typeof item === 'string' && item.includes(',')) item = '"' + item + '"';
                    line += item;
                }}
                str += line + '\\r\\n';
            }}
            return str;
        }}

        function downloadCSV(classVal) {{
            const rows = allData.filter(row => decisions[String(row.Original_Row_ID)] == classVal);
            if(rows.length === 0) {{ alert("No candidates found for Class " + classVal); return; }}
            const exportRows = rows.map(row => ({{ ...row, Morphology_Class: classVal }}));
            triggerDownload(exportRows, `morphology_class_${{classVal}}.csv`);
        }}

        function downloadMaster() {{
            const vettedRows = allData.filter(row => decisions[String(row.Original_Row_ID)] !== undefined);
            if(vettedRows.length === 0) {{ alert("No candidates vetted yet!"); return; }}
            const masterRows = vettedRows.map(row => ({{
                ...row,
                Morphology_Class: decisions[String(row.Original_Row_ID)]
            }}));
            triggerDownload(masterRows, 'MASTER_Morphology_Catalogue.csv');
        }}

        function triggerDownload(data, filename) {{
            const a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([convertToCSV(data)], {{type: 'text/csv'}}));
            a.download = filename;
            a.click();
        }}

        // MODAL FUNCTIONS
        function showExample(type) {{
            const modal = document.getElementById('exampleModal');
            const modalImg = document.getElementById('modal-image');
            const modalTitle = document.getElementById('modal-title');
            let filename = '';

            if (type === 'WAT') {{
                filename = 'WAT.png';
                modalTitle.innerText = 'Wide-Angle Tailed (WAT) Example';
                modalImg.style.filter = 'grayscale(100%)'; 
            }} else if (type === 'NAT') {{
                filename = 'NAT.png';
                modalTitle.innerText = 'Narrow-Angle Tailed (NAT) Example';
                modalImg.style.filter = 'grayscale(100%)'; 
            }} else if (type === 'S_and_Z') {{
                filename = 'S and Z.png';
                modalTitle.innerText = 'S or Z Shaped Example';
                modalImg.style.filter = 'none'; 
            }} else if (type === 'X_Shaped') {{
                filename = 'X Shaped.png';
                modalTitle.innerText = 'X Shaped Example';
                modalImg.style.filter = 'none'; 
            }}

            modalImg.src = filename;
            modal.style.display = 'flex'; 
        }}

        function closeModal() {{
            document.getElementById('exampleModal').style.display = 'none';
        }}
    </script>
    </body>
    </html>
    """

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Done! Open '{OUTPUT_HTML}' to start rapid vetting.", flush=True)

def main():
    print("Script starting...", flush=True)
    try:
        if not os.path.exists(CSV_PATH): 
            print(f"CSV missing at {CSV_PATH}", flush=True)
            return
            
        print("Reading CSV...", flush=True)
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
        sb_col = next((c for c in df.columns if 'sb' in c.lower()), None)
        prob_col = next((c for c in df.columns if 'prob' in c.lower()), None)
        
        if not sb_col: 
            print("No SB column found.", flush=True)
            return

        df = remove_duplicates(df)
        if prob_col: df = df.sort_values(by=prob_col, ascending=False)
        df['Original_Row_ID'] = df.index + 2

        # Check Hard Drive before scanning
        print("Checking Hard Drive Access...", flush=True)
        try:
            if os.path.exists(HARD_DRIVE_PATH):
                print("Drive found. Scanning...", flush=True)
                file_map = get_file_map(HARD_DRIVE_PATH)
            else:
                print("DRIVE NOT FOUND. Using empty map.", flush=True)
                file_map = {}
        except Exception as e:
            print(f"Drive Error: {e}", flush=True)
            file_map = {}

        valid_records = []
        print("\n--- Generating Images ---", flush=True)
        for i, row in df.iterrows():
            raw_sb = str(row[sb_col])
            clean_sb = re.sub(r"[^0-9]", "", raw_sb)
            correct_fits = find_correct_file(clean_sb, row['RA'], row['Dec'], file_map)

            if correct_fits:
                png_name = f"img_{clean_sb}_ra{row['RA']:.2f}.png"
                png_path = os.path.join(IMAGE_DIR, png_name)
                if not os.path.exists(png_path):
                    success = create_rgb_png(correct_fits, png_path, row['RA'], row['Dec'])
                else: success = True

                # For the HTML, we need the path relative to the HTML file itself
                relative_png_path = os.path.join("vetting_images", png_name)
                
                # Generate Aladin URL here
                aladin_url = f"https://aladin.u-strasbg.fr/AladinLite/?target={row['RA']}+{row['Dec']}&fov=0.08&survey=P/DSS2/color"

                if success:
                    if len(valid_records) % 50 == 0: print(".", end="", flush=True)
                    rec = row.to_dict()
                    rec['vet_img_path'] = relative_png_path 
                    rec['aladin_url'] = aladin_url # Store URL in the record
                    rec[sb_col] = str(row[sb_col])
                    valid_records.append(rec)

        generate_html(valid_records, sb_col)

    except KeyboardInterrupt:
        print("\nEXITING (KeyboardInterrupt)", flush=True)
    except Exception as e:
        print(f"\nCRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()